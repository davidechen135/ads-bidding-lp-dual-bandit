"""Budget pacing / bidding strategies.

All strategies share one interface used by :func:`adbid.engine.run_day`:

* ``reset(B, R, forecast)`` - start of the day
* ``bid(ctx, v) -> (bids, participate)`` - bids for every request of the slot
* ``update(ctx, spend, value)`` - feedback after the slot (spend & predicted
  value actually won in this slot)
* ``trace()`` - scalars to log (lambda, mu, multiplier ...)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlotContext:
    t: int
    T: int
    B: float
    remaining: float  # budget remaining at start of slot
    forecast: np.ndarray  # forecast traffic share per slot (sums to 1)
    n_requests: int

    @property
    def remaining_traffic_share(self) -> float:
        return float(self.forecast[self.t:].sum())

    def adaptive_target(self) -> float:
        """rho_t: re-plan remaining budget proportionally to remaining
        expected traffic."""
        rem = self.remaining_traffic_share
        if rem <= 0:
            return self.remaining
        return self.remaining * self.forecast[self.t] / rem

    def static_target(self) -> float:
        return self.B * self.forecast[self.t]

    def pace_ratio(self) -> float:
        """(remaining budget share) / (remaining traffic share); >1 means we are
        behind plan (under-spending), <1 means ahead (over-spending)."""
        return (self.remaining / self.B) / max(self.remaining_traffic_share, 1e-9)


class Strategy:
    name = "base"

    def reset(self, B: float, R: float, forecast: np.ndarray) -> None:
        self.B, self.R, self.forecast = B, R, forecast

    def bid(self, ctx: SlotContext, v: np.ndarray, rng: np.random.Generator):
        raise NotImplementedError

    def update(self, ctx: SlotContext, spend: float, value: float) -> None:
        pass

    def trace(self) -> dict:
        return {}


# ----------------------------------------------------------------------------
class NoPacing(Strategy):
    """Bid the ROI-break-even price v/R on every request until the budget is
    exhausted (typical oCPX bidding without pacing)."""

    name = "NoPacing"

    def bid(self, ctx, v, rng):
        return v / self.R, np.ones(len(v), bool)

    def trace(self):
        return {"alpha": 1.0 / self.R}


class Throttling(Strategy):
    """Probabilistic throttling: participate in each auction with probability
    p_t, bid v/R; p_t follows the plan via a multiplicative controller."""

    name = "Throttling"

    def __init__(self, p0: float = 1.0, k: float = 0.4, p_min: float = 0.01):
        self.p0, self.k, self.p_min = p0, k, p_min

    def reset(self, B, R, forecast):
        super().reset(B, R, forecast)
        self.p = self.p0

    def bid(self, ctx, v, rng):
        return v / self.R, rng.random(len(v)) < self.p

    def update(self, ctx, spend, value):
        rho = ctx.adaptive_target()
        e = np.clip((rho - spend) / max(rho, 1e-9), -2.0, 1.0)
        self.p = float(np.clip(self.p * np.exp(self.k * e), self.p_min, 1.0))

    def trace(self):
        return {"p": self.p, "alpha": self.p / self.R}


class PIDPacing(Strategy):
    """PID controller on the spend error that shades the ROI bid:
    bid = m_t * v / R, m_t = clip(exp(u_t), m_min, 1)."""

    name = "PID"

    def __init__(self, kp=0.30, ki=0.05, kd=0.10, m_min=0.05):
        self.kp, self.ki, self.kd, self.m_min = kp, ki, kd, m_min

    def reset(self, B, R, forecast):
        super().reset(B, R, forecast)
        self.I = 0.0
        self.prev_e = 0.0
        self.m = 1.0

    def bid(self, ctx, v, rng):
        return self.m * v / self.R, np.ones(len(v), bool)

    def update(self, ctx, spend, value):
        rho = ctx.adaptive_target()
        e = float(np.clip((rho - spend) / max(rho, 1e-9), -2.0, 1.0))
        self.I = float(np.clip(self.I + e, -20.0, 2.0))  # anti wind-up
        u = self.kp * e + self.ki * self.I + self.kd * (e - self.prev_e)
        self.prev_e = e
        self.m = float(np.clip(self.m * np.exp(u), self.m_min, 1.0))

    def trace(self):
        return {"alpha": self.m / self.R}


class DualPacing(Strategy):
    """Lagrangian-dual pacing (online projected sub-gradient / mirror descent
    on the dual of the ROI + budget LP).

    bid_i = alpha_t * v_i,   alpha_t = (1 + mu_t) / (lambda_t + mu_t R)

    After every slot:
        rho_t   = remaining_budget * f_t / sum_{s>=t} f_s            (re-plan)
        g_lam   = (spend_t - rho_t) / rho_t                           (normalised)
        g_mu    = (R spend_t - value_t) / (R rho_t)
        eta_t   = eta0 * (1 + kappa |log(pace_ratio_t)|)              (adaptive)
        lambda <- max(0, lambda + eta_t * s_lam * g_lam)
        mu     <- max(0, mu     + eta_t * s_mu  * g_mu)

    which is exactly  lambda <- max(0, lambda - eta (rho_t - spend_t))  with a
    per-slot normalisation, i.e. a dual sub-gradient step (the sub-gradient
    of the dual function w.r.t. lambda is rho_t - spend_t).
    """

    name = "DualPacing"

    def __init__(
        self,
        lam0: float | None = None,
        mu0: float | None = None,
        eta0: float = 0.04,
        eta_mu: float = 0.04,
        kappa: float = 1.0,
        adaptive: bool = True,
        update: bool = True,
        alpha_max_mult: float = 4.0,
        name: str | None = None,
    ):
        self.lam0, self.mu0 = lam0, mu0
        self.eta0, self.eta_mu, self.kappa = eta0, eta_mu, kappa
        self.adaptive, self.do_update = adaptive, update
        self.alpha_max_mult = alpha_max_mult
        if name:
            self.name = name

    def reset(self, B, R, forecast):
        super().reset(B, R, forecast)
        # cold default: no budget pressure, ROI multiplier giving bid = 2 v / (R+1)... ~ v/R
        self.lam = 0.0 if self.lam0 is None else float(self.lam0)
        self.mu = 1.0 if self.mu0 is None else float(self.mu0)
        self.alpha_max = self.alpha_max_mult / R
        self.s_lam = R  # natural scale of lambda (value per unit cost)
        self.s_mu = 1.0
        self.eta_t = self.eta0

    @property
    def alpha(self) -> float:
        denom = self.lam + self.mu * self.R
        if denom <= 1e-12:
            return self.alpha_max
        return float(min(self.alpha_max, (1.0 + self.mu) / denom))

    def bid(self, ctx, v, rng):
        return self.alpha * v, np.ones(len(v), bool)

    def update(self, ctx, spend, value):
        if not self.do_update:
            return
        rho = max(ctx.adaptive_target(), 1e-9)
        g_lam = float(np.clip((spend - rho) / rho, -1.0, 3.0))
        g_mu = float(np.clip((self.R * spend - value) / (self.R * rho), -1.0, 3.0))
        if self.adaptive:
            r = ctx.pace_ratio()
            self.eta_t = self.eta0 * (1.0 + self.kappa * abs(np.log(max(r, 1e-3))))
        else:
            self.eta_t = self.eta0
        self.lam = max(0.0, self.lam + self.eta_t * self.s_lam * g_lam)
        self.mu = max(0.0, self.mu + self.eta_mu * self.s_mu * g_mu * (self.eta_t / self.eta0))

    def trace(self):
        return {"lam": self.lam, "mu": self.mu, "alpha": self.alpha, "eta": self.eta_t}
