"""Unit tests for LP dual pacing & contextual bandits."""
import numpy as np
import pytest

from adbid.bandits import CreativeArm, LinUCBPolicy, LinearTSPolicy, EpsilonGreedyPolicy
from adbid.lp_offline import solve_hindsight_lp, dual_objective
from adbid.pacing import DualPacing, NoPacing, Throttling, SlotContext
from adbid.simulator import AuctionSimulator, default_campaigns
from adbid.engine import run_day


def test_hindsight_lp_strong_duality():
    sim = AuctionSimulator()
    camp = default_campaigns(n_requests=2000)[0]
    day = sim.generate_day(camp, seed=42)
    B = sim.calibrate_budget(camp, seed=42)
    R = camp.target_roi

    lp_res = solve_hindsight_lp(day, B, R)
    assert "Optimal" in lp_res.status or lp_res.status == "optimal"
    assert lp_res.duality_gap < 1e-4
    assert lp_res.lam >= 0.0
    assert lp_res.mu >= 0.0


def test_dual_pacing_runs_and_respects_budget():
    sim = AuctionSimulator()
    camp = default_campaigns(n_requests=3000)[0]
    day = sim.generate_day(camp, seed=42)
    B = sim.calibrate_budget(camp, seed=42)
    R = camp.target_roi

    strat = DualPacing(eta0=0.05)
    res = run_day(day, strat, B, R, seed=42)

    # Budget constraint must never be violated
    assert res.total_spend <= B + 1e-6
    # Budget completion should be reasonable (> 70%)
    assert res.completion > 0.70
    assert res.conversions > 0


def test_linucb_regret_sublinear():
    """Verify LinUCB learns better arm over random on a simple bandit task."""
    d = 4
    K = 3
    rng = np.random.default_rng(42)
    arms = [
        CreativeArm(id=0, name="arm0", weight=np.array([0.0, 0.1, 0.1, 0.1]), intercept=-1.0),
        CreativeArm(id=1, name="arm1", weight=np.array([0.0, 0.5, 0.5, 0.5]), intercept=0.5),  # best
        CreativeArm(id=2, name="arm2", weight=np.array([0.0, -0.2, -0.2, -0.2]), intercept=-2.0),
    ]
    pol = LinUCBPolicy(alpha=1.0)
    pol.reset(K, d, rng)

    n_rounds = 500
    chosen_counts = np.zeros(K)
    for _ in range(n_rounds):
        x = np.column_stack([np.ones(1), rng.normal(0, 1, (1, d - 1))])
        arm, _ = pol.select(x)
        chosen_counts[arm[0]] += 1
        true_p = arms[arm[0]].true_cvr(x)[0]
        y = np.array([float(rng.random() < true_p)])
        pol.update(x, arm, y)

    # Arm 1 should be selected most frequently after learning
    assert chosen_counts[1] > chosen_counts[0]
    assert chosen_counts[1] > chosen_counts[2]
