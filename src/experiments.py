"""Comprehensive benchmark runner for Ads LP Dual Pacing & Contextual Bandits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from adbid.bandits import (
    BanditValueProvider,
    CreativeArm,
    EpsilonGreedyPolicy,
    LinearTSPolicy,
    LinUCBPolicy,
    OraclePolicy,
    PureGreedyPolicy,
    RandomPolicy,
)
from adbid.engine import StaticValues, run_day
from adbid.lp_offline import solve_hindsight_lp
from adbid.pacing import DualPacing, NoPacing, PIDPacing, Throttling
from adbid.simulator import AuctionSimulator, CampaignConfig, default_campaigns


def run_pacing_benchmark(
    sim: AuctionSimulator,
    camp: CampaignConfig,
    n_seeds: int = 5,
    output_dir: Path | None = None,
) -> dict:
    """Benchmark pacing strategies against Hindsight LP upper bound across multiple seeds."""
    strategies_fn = {
        "NoPacing": lambda: NoPacing(),
        "Throttling": lambda: Throttling(),
        "PIDPacing": lambda: PIDPacing(kp=0.4, ki=0.08, kd=0.01),
        "DualPacing": lambda: DualPacing(eta0=0.04, adaptive=True),
    }

    all_results: dict[str, list[dict]] = {name: [] for name in strategies_fn}
    all_results["HindsightLP"] = []

    last_runs = {}

    for s in range(n_seeds):
        seed = 100 + s
        day = sim.generate_day(camp, seed=seed)
        B = sim.calibrate_budget(camp, seed=seed)
        R = camp.target_roi

        # Solve Hindsight LP
        lp_res = solve_hindsight_lp(day, B, R)
        won_lp = lp_res.x > 0.5
        all_results["HindsightLP"].append({
            "completion": float(day.price[won_lp].sum() / B),
            "conversions": float(day.p_conv[won_lp].sum()),
            "roi_true": float(day.p_conv[won_lp].sum() * day.value_per_conv / max(1e-6, day.price[won_lp].sum())),
            "roi_violation": 0.0,
            "exhaust_h": 24.0,
            "plan_dev": 0.0,
        })

        for name, fn in strategies_fn.items():
            strat = fn()
            res = run_day(day, strat, B, R, seed=seed)
            all_results[name].append(res.summary())
            if s == 0:
                last_runs[name] = res

    # Aggregate metrics
    summary = {}
    for name, res_list in all_results.items():
        summary[name] = {
            "completion_mean": float(np.mean([r["completion"] for r in res_list])),
            "completion_std": float(np.std([r["completion"] for r in res_list])),
            "conversions_mean": float(np.mean([r["conversions"] for r in res_list])),
            "conversions_std": float(np.std([r["conversions"] for r in res_list])),
            "roi_true_mean": float(np.mean([r["roi_true"] for r in res_list])),
            "roi_violation_mean": float(np.mean([r["roi_violation"] for r in res_list])),
            "exhaust_h_mean": float(np.mean([r["exhaust_h"] for r in res_list])),
            "plan_dev_mean": float(np.mean([r["plan_dev"] for r in res_list])),
        }

    # Generate plots
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # 1. Spend curve vs plan
        plt.figure(figsize=(10, 5))
        hours = np.linspace(0, 24, day.n_slots)
        plt.plot(hours, last_runs["DualPacing"].plan, "k--", label="Target Traffic Plan", alpha=0.8, linewidth=2)
        for name in ["NoPacing", "Throttling", "PIDPacing", "DualPacing"]:
            plt.plot(hours, last_runs[name].spend, label=name, alpha=0.85)
        plt.xlabel("Hour of Day")
        plt.ylabel("Spend per Slot ($)")
        plt.title("Budget Pacing: Real-Time Hourly Spend Trajectories")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "spend_trajectories.png", dpi=200)
        plt.close()

        # 2. Dual variable convergence
        if "lam" in last_runs["DualPacing"].trace:
            plt.figure(figsize=(10, 4))
            plt.plot(hours, last_runs["DualPacing"].trace["lam"], label=r"Dual multiplier $\lambda$ (Budget)", color="darkblue")
            plt.plot(hours, last_runs["DualPacing"].trace["mu"], label=r"Dual multiplier $\mu$ (ROI)", color="crimson")
            plt.plot(hours, last_runs["DualPacing"].trace["alpha"], label=r"Bid scalar $\alpha_t$", color="darkgreen", linestyle=":")
            plt.xlabel("Hour of Day")
            plt.ylabel("Value")
            plt.title("Online Lagrangian Dual Multiplier Trajectories")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "dual_multipliers.png", dpi=200)
            plt.close()

    return summary


def run_bandit_benchmark(
    sim: AuctionSimulator,
    camp: CampaignConfig,
    n_arms: int = 5,
    n_seeds: int = 3,
    output_dir: Path | None = None,
) -> dict:
    """Benchmark Contextual Bandits in cold start advertising."""
    d = sim.m.n_features
    rng_init = np.random.default_rng(999)

    # Create synthetic creative arms with varying effectiveness
    arms = []
    for a in range(n_arms):
        w = rng_init.normal(0, 0.4, size=d)
        # One high-potential arm, some average, one weak
        shift = 0.5 * (a - n_arms // 2)
        arms.append(CreativeArm(id=a, name=f"Creative_{a}", weight=w, intercept=-2.8 + shift))

    policies_fn = {
        "Random": lambda: RandomPolicy(),
        "PureGreedy": lambda: PureGreedyPolicy(),
        "EpsGreedy": lambda: EpsilonGreedyPolicy(epsilon=0.1),
        "LinearTS": lambda: LinearTSPolicy(v=0.3),
        "LinUCB": lambda: LinUCBPolicy(alpha=0.8),
        "Oracle": lambda: OraclePolicy(arms),
    }

    results: dict[str, list[dict]] = {name: [] for name in policies_fn}
    regrets: dict[str, list[np.ndarray]] = {name: [] for name in policies_fn}

    for s in range(n_seeds):
        seed = 300 + s
        day = sim.generate_day(camp, seed=seed)
        B = sim.calibrate_budget(camp, seed=seed)
        R = camp.target_roi

        for name, fn in policies_fn.items():
            pol = fn()
            provider = BanditValueProvider(day, arms, pol, seed=seed)
            # Use Dual Pacing for all to isolate bandit exploration effect
            pacer = DualPacing(eta0=0.04)
            res = run_day(day, pacer, B, R, provider=provider, seed=seed)
            res_dict = res.summary()
            res_dict["cum_regret"] = provider.cum_regret
            results[name].append(res_dict)
            regrets[name].append(np.array(provider.regret_history))

    summary = {}
    for name, r_list in results.items():
        summary[name] = {
            "conversions_mean": float(np.mean([r["conversions"] for r in r_list])),
            "conversions_std": float(np.std([r["conversions"] for r in r_list])),
            "cum_regret_mean": float(np.mean([r["cum_regret"] for r in r_list])),
            "cum_regret_std": float(np.std([r["cum_regret"] for r in r_list])),
            "completion_mean": float(np.mean([r["completion"] for r in r_list])),
            "roi_true_mean": float(np.mean([r["roi_true"] for r in r_list])),
        }

    # Plot regret curves
    if output_dir:
        fig_dir = Path(output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 5))
        for name in ["Random", "PureGreedy", "EpsGreedy", "LinearTS", "LinUCB"]:
            mean_reg = np.mean(regrets[name], axis=0)
            plt.plot(mean_reg, label=name, linewidth=1.8)
        plt.xlabel("Slot Index (Time)")
        plt.ylabel("Cumulative Regret vs Oracle")
        plt.title("Cold-Start Ad Creative Exploration: Cumulative Regret")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "bandit_regret.png", dpi=200)
        plt.close()

    return summary


def main():
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    sim = AuctionSimulator()
    camp = default_campaigns(n_requests=30_000)[0]  # C1_ecom

    print(f"--- Running Budget Pacing Benchmark for {camp.name} ---")
    pacing_metrics = run_pacing_benchmark(sim, camp, n_seeds=5, output_dir=output_dir)
    print("Pacing Results:")
    for k, v in pacing_metrics.items():
        print(f"  {k:12s} | Completion: {v['completion_mean']*100:.1f}% | Conv: {v['conversions_mean']:.1f} | ROI: {v['roi_true_mean']:.2f} | PlanDev: {v['plan_dev_mean']:.3f}")

    print("\n--- Running Contextual Bandits Benchmark ---")
    bandit_metrics = run_bandit_benchmark(sim, camp, n_arms=5, n_seeds=3, output_dir=output_dir)
    print("Bandit Results:")
    for k, v in bandit_metrics.items():
        print(f"  {k:12s} | Conv: {v['conversions_mean']:.1f} | Regret: {v['cum_regret_mean']:.2f} | Completion: {v['completion_mean']*100:.1f}%")

    # Save metrics json
    final_out = {
        "pacing": pacing_metrics,
        "bandits": bandit_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nAll results saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
