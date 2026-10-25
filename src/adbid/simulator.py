"""Vectorised RTB auction simulator.

A day is split into ``n_slots`` equal time slots (default 288 x 5 minutes).
For every campaign we generate a stream of bid requests (impression
opportunities) whose arrival intensity follows a diurnal traffic curve with a
late-morning, a lunch and an evening peak.  Every request carries

* a context feature vector ``x`` (user / page / time features),
* the *true* click and conversion probabilities (``ctr``, ``cvr``),
* the *predicted* pCTR / pCVR produced by an imperfect model
  (multiplicative log-normal noise, mean-preserving),
* the competing market price ``d`` = highest competitor bid, drawn from a
  log-normal whose location depends on the hour (more competition at peaks)
  and on the impression quality.

Everything is seeded and generated with numpy in one shot, so a full day with
~30k requests per campaign takes milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Traffic & market curves
# ----------------------------------------------------------------------------


def _gauss(h: np.ndarray, mu: float, sd: float) -> np.ndarray:
    # circular distance on a 24h clock
    d = np.minimum(np.abs(h - mu), 24.0 - np.abs(h - mu))
    return np.exp(-0.5 * (d / sd) ** 2)


def traffic_curve(n_slots: int) -> np.ndarray:
    """Expected relative traffic per slot (normalised to sum to 1).

    Night trough (2am-6am), small morning commute peak (~9h),
    lunch peak (~12.5h) and a dominant evening prime-time peak (~20.5h).
    """
    h = (np.arange(n_slots) + 0.5) * 24.0 / n_slots
    f = (
        0.12
        + 0.35 * _gauss(h, 9.0, 1.2)
        + 0.70 * _gauss(h, 12.5, 1.6)
        + 1.00 * _gauss(h, 20.5, 2.0)
    )
    return f / f.sum()


def competition_curve(n_slots: int) -> np.ndarray:
    """Relative competition intensity in [0, 1] (drives market price level)."""
    f = traffic_curve(n_slots)
    return (f - f.min()) / (f.max() - f.min())


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------


@dataclass
class CampaignConfig:
    name: str
    value_per_conv: float = 100.0  # advertiser value of one conversion
    target_roi: float = 1.5  # required value / spend
    budget_frac: float = 0.5  # daily budget as fraction of unconstrained spend
    n_requests: int = 30_000  # expected bid requests per day
    base_ctr: float = 0.02
    base_cvr: float = 0.05


@dataclass
class MarketConfig:
    n_slots: int = 288
    n_features: int = 6
    price_median: float = 0.07  # off-peak median market price per impression
    price_peak_uplift: float = 0.45  # log-uplift of median price at the peak
    price_sigma: float = 0.55  # log-normal sigma of competitor price
    price_quality_coef: float = 0.35  # correlation of price with true CTR
    pred_noise_sigma: float = 0.25  # model error (log-normal) on pCTR and pCVR
    traffic_day_sigma: float = 0.08  # day-level traffic volume shock
    traffic_hour_sigma: float = 0.10  # hour-level forecast error
    price_day_sigma: float = 0.10  # day-level shift of the market price level
    auction: str = "second"  # 'second' or 'first' price


@dataclass
class DayData:
    """All bid requests of one campaign for one day (sorted by time)."""

    slot: np.ndarray  # (N,) int slot index
    x: np.ndarray  # (N, d) context features
    ctr: np.ndarray  # true CTR
    cvr: np.ndarray  # true CVR (post-click)
    pctr: np.ndarray  # predicted CTR
    pcvr: np.ndarray  # predicted CVR
    price: np.ndarray  # market price (highest competing bid)
    value_per_conv: float
    n_slots: int
    forecast: np.ndarray  # (n_slots,) forecast traffic share used for the plan
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.slot)

    @property
    def p_conv(self) -> np.ndarray:
        """True conversion probability per impression."""
        return self.ctr * self.cvr

    @property
    def p_conv_hat(self) -> np.ndarray:
        return self.pctr * self.pcvr

    @property
    def v_true(self) -> np.ndarray:
        return self.p_conv * self.value_per_conv

    @property
    def v_hat(self) -> np.ndarray:
        return self.p_conv_hat * self.value_per_conv

    def slot_bounds(self) -> np.ndarray:
        """Start offsets of each slot (len n_slots+1) - requests are sorted."""
        return np.searchsorted(self.slot, np.arange(self.n_slots + 1))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p / (1 - p))


class AuctionSimulator:
    """Generates days of bid requests for a set of campaigns."""

    def __init__(self, market: MarketConfig | None = None, model_seed: int = 0):
        self.m = market or MarketConfig()
        rng = np.random.default_rng(model_seed)
        d = self.m.n_features
        # fixed "world" weights for CTR / CVR (shared across days)
        # feature 0 is intercept 1.0; set w[0] = 0.0 so camp.base_ctr and camp.base_cvr govern baseline
        self.w_ctr = rng.normal(0, 0.45, d)
        self.w_ctr[0] = 0.0
        self.w_cvr = rng.normal(0, 0.35, d)
        self.w_cvr[0] = 0.0
        self.base_curve = traffic_curve(self.m.n_slots)
        self.comp = competition_curve(self.m.n_slots)

    # ------------------------------------------------------------------
    def generate_day(self, camp: CampaignConfig, seed: int) -> DayData:
        m = self.m
        rng = np.random.default_rng(seed)
        T = m.n_slots
        # realised traffic deviates from the forecast curve
        hour_shock = np.exp(rng.normal(0, m.traffic_hour_sigma, 24))
        hour_of_slot = (np.arange(T) * 24) // T
        day_shock = np.exp(rng.normal(0, m.traffic_day_sigma))
        lam = camp.n_requests * self.base_curve * hour_shock[hour_of_slot] * day_shock
        counts = rng.poisson(lam)
        slot = np.repeat(np.arange(T), counts)
        n = len(slot)

        # Context features with explicit column 0 as constant 1.0 (intercept)
        x_raw = rng.normal(0, 1, (n, m.n_features - 1))
        x = np.column_stack([np.ones(n), x_raw])
        hour = (slot + 0.5) * 24.0 / T
        # evening users convert a bit better, night users a bit worse
        hour_cvr = 0.15 * np.sin((hour - 14.0) / 24.0 * 2 * np.pi)
        z_ctr = _logit(camp.base_ctr) + x @ self.w_ctr
        z_cvr = _logit(camp.base_cvr) + x @ self.w_cvr + hour_cvr
        ctr = _sigmoid(z_ctr)
        cvr = _sigmoid(z_cvr)

        s = m.pred_noise_sigma
        pctr = np.clip(ctr * np.exp(rng.normal(-0.5 * s * s, s, n)), 1e-6, 0.999)
        pcvr = np.clip(cvr * np.exp(rng.normal(-0.5 * s * s, s, n)), 1e-6, 0.999)

        log_med = (
            np.log(m.price_median)
            + m.price_peak_uplift * self.comp[slot]
            + m.price_quality_coef * (z_ctr - _logit(camp.base_ctr))
        )
        log_med = log_med + rng.normal(0, m.price_day_sigma)
        price = np.exp(log_med + rng.normal(0, m.price_sigma, n))

        return DayData(
            slot=slot,
            x=x,
            ctr=ctr,
            cvr=cvr,
            pctr=pctr,
            pcvr=pcvr,
            price=price,
            value_per_conv=camp.value_per_conv,
            n_slots=T,
            forecast=self.base_curve.copy(),
            meta={"campaign": camp.name, "seed": seed},
        )

    # ------------------------------------------------------------------
    def calibrate_budget(self, camp: CampaignConfig, seed: int) -> float:
        """Daily budget = budget_frac x spend of unconstrained ROI-bidding
        (bid = v_hat / R, no budget) on a *reference* (historical) day."""
        day = self.generate_day(camp, seed)
        bid = day.v_hat / camp.target_roi
        win = bid >= day.price
        cost = day.price if self.m.auction == "second" else bid
        return float(camp.budget_frac * cost[win].sum())


def default_campaigns(n_requests: int = 30_000) -> list[CampaignConfig]:
    return [
        CampaignConfig("C1_ecom", 100.0, 1.5, 0.50, n_requests, 0.020, 0.050),
        CampaignConfig("C2_game", 60.0, 1.2, 0.40, n_requests, 0.025, 0.060),
        CampaignConfig("C3_finance", 180.0, 2.0, 1.20, n_requests, 0.015, 0.040),
        CampaignConfig("C4_travel", 120.0, 1.6, 4.00, n_requests, 0.018, 0.045),
    ]


def run_auction(bids: np.ndarray, price: np.ndarray, auction: str = "second"):
    """Return (win mask, cost per request). Ties are won by the advertiser."""
    win = (bids >= price) & (bids > 0)
    cost = np.where(win, price if auction == "second" else bids, 0.0)
    return win, cost
