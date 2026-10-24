"""Day-level auction replay engine: runs one strategy on one campaign-day."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pacing import SlotContext, Strategy
from .simulator import DayData, run_auction


class StaticValues:
    """Default value provider: the CTR/CVR model's predicted value v_hat."""

    def __init__(self, day: DayData):
        self.day = day

    def values(self, t, lo, hi):
        v = self.day.v_hat[lo:hi]
        # (value used in the bid, value used for ROI accounting, true conv prob)
        return v, v, self.day.p_conv[lo:hi]

    def feedback(self, t, lo, hi, win, rng):
        pass


@dataclass
class DayResult:
    strategy: str
    B: float
    R: float
    value_per_conv: float
    spend: np.ndarray  # per slot
    value_hat: np.ndarray  # per slot, predicted value won
    conv: np.ndarray  # per slot, expected true conversions won
    wins: np.ndarray
    requests: np.ndarray
    plan: np.ndarray  # static plan per slot (B * forecast)
    trace: dict = field(default_factory=dict)  # name -> per-slot array

    # ---- summary metrics --------------------------------------------------
    @property
    def total_spend(self):
        return float(self.spend.sum())

    @property
    def completion(self):
        return self.total_spend / self.B

    @property
    def conversions(self):
        return float(self.conv.sum())

    @property
    def roi_true(self):
        return self.conversions * self.value_per_conv / max(self.total_spend, 1e-12)

    @property
    def roi_hat(self):
        return float(self.value_hat.sum()) / max(self.total_spend, 1e-12)

    @property
    def roi_violation(self):
        """Relative shortfall of the (true) ROI below the target, >= 0."""
        return max(0.0, (self.R - self.roi_true) / self.R)

    @property
    def exhaustion_hour(self):
        T = len(self.spend)
        cum = np.cumsum(self.spend)
        hit = np.nonzero(cum >= 0.99 * self.B)[0]
        return 24.0 if len(hit) == 0 else float((hit[0] + 1) * 24.0 / T)

    @property
    def plan_deviation(self):
        """Mean |cumulative spend - cumulative plan| / B over the day."""
        return float(np.mean(np.abs(np.cumsum(self.spend) - np.cumsum(self.plan))) / self.B)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "spend": self.total_spend,
            "completion": self.completion,
            "exhaust_h": self.exhaustion_hour,
            "conversions": self.conversions,
            "roi_true": self.roi_true,
            "roi_hat": self.roi_hat,
            "roi_target": self.R,
            "roi_violation": self.roi_violation,
            "plan_dev": self.plan_deviation,
            "value_hat": float(self.value_hat.sum()),
        }


def run_day(
    day: DayData,
    strategy: Strategy,
    B: float,
    R: float,
    provider=None,
    auction: str = "second",
    seed: int = 0,
) -> DayResult:
    rng = np.random.default_rng(seed)
    provider = provider or StaticValues(day)
    T = day.n_slots
    bounds = day.slot_bounds()
    strategy.reset(B, R, day.forecast)
    remaining = B
    spend = np.zeros(T)
    vhat = np.zeros(T)
    conv = np.zeros(T)
    wins = np.zeros(T, int)
    reqs = np.diff(bounds)
    traces: dict[str, list] = {}
    for t in range(T):
        lo, hi = bounds[t], bounds[t + 1]
        ctx = SlotContext(t, T, B, remaining, day.forecast, hi - lo)
        for k, val in strategy.trace().items():
            traces.setdefault(k, []).append(val)
        if hi == lo:
            strategy.update(ctx, 0.0, 0.0)
            continue
        if remaining <= 1e-9:
            provider.feedback(t, lo, hi, np.zeros(hi - lo, bool), rng)
            continue
        v_bid, v_acct, p_true = provider.values(t, lo, hi)
        bids, part = strategy.bid(ctx, v_bid, rng)
        bids = np.where(part, bids, 0.0)
        win, cost = run_auction(bids, day.price[lo:hi], auction)
        # hard budget cap: requests arrive sequentially, stop once budget is out
        afford = np.cumsum(cost) <= remaining + 1e-12
        win &= afford
        cost = np.where(win, cost, 0.0)
        s = float(cost.sum())
        remaining -= s
        spend[t] = s
        vhat[t] = float(v_acct[win].sum())
        conv[t] = float(p_true[win].sum())
        wins[t] = int(win.sum())
        provider.feedback(t, lo, hi, win, rng)
        strategy.update(ctx, s, vhat[t])
    return DayResult(
        strategy=strategy.name,
        B=B,
        R=R,
        value_per_conv=day.value_per_conv,
        spend=spend,
        value_hat=vhat,
        conv=conv,
        wins=wins,
        requests=reqs,
        plan=B * day.forecast,
        trace={k: np.asarray(v) for k, v in traces.items()},
    )
