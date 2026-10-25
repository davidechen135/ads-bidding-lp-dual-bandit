"""Contextual Bandits for ad creative / campaign cold-start exploration.

Algorithms:
* Disjoint LinUCB (Li et al., WWW 2010)
* Contextual Linear Thompson Sampling (Agrawal & Goyal, ICML 2013)
* Contextual Epsilon-Greedy
* Pure Greedy (LinUCB with alpha=0)
* Random
* Oracle (knows true conversion probability per arm)

Integrated into RTB via :class:`BanditValueProvider`:
Per impression opportunity, the policy picks an ad creative (arm) based on
the user context x. The estimated value v_hat feeds the bidding strategy
(e.g. Dual Pacing). Feedback is received only for impressions actually *won*
in the second-price auction (censored feedback / auction mechanics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class CreativeArm:
    """An ad creative / sub-campaign candidate with latent response weights."""

    id: int
    name: str
    weight: np.ndarray  # (d,)
    intercept: float  # baseline logit shift

    def true_cvr(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ self.weight + self.intercept)


class BanditPolicy:
    name: str = "base_bandit"

    def reset(self, n_arms: int, n_features: int, rng: np.random.Generator) -> None:
        self.n_arms = n_arms
        self.d = n_features
        self.rng = rng

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Given context x of shape (N, d), return:
        (chosen_arms: (N,), bid_cvr: (N,)) or if return_point is True:
        (chosen_arms: (N,), bid_cvr: (N,), point_cvr: (N,))
        """
        raise NotImplementedError

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        """Update model on impressions that were actually won."""
        raise NotImplementedError


class LinUCBPolicy(BanditPolicy):
    """Disjoint Linear Upper Confidence Bound (LinUCB).

    For each arm a:
        A_a = I_d * lambda + sum x x^T
        b_a = sum x y + A_a * theta_prior
        theta_a = A_a^{-1} b_a
        score_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)
    """

    name = "LinUCB"

    def __init__(self, alpha: float = 0.5, ridge_lambda: float = 25.0, prior_cvr: float = 0.05):
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.prior_cvr = float(prior_cvr)

    def reset(self, n_arms: int, n_features: int, rng: np.random.Generator) -> None:
        super().reset(n_arms, n_features, rng)
        self.A = [np.eye(self.d) * self.ridge_lambda for _ in range(n_arms)]
        # Prior theta initialized to give expected prior_cvr on intercept (x[:, 0] == 1.0)
        self.prior_theta = np.zeros(self.d)
        self.prior_theta[0] = self.prior_cvr
        self.b = [self.A[a] @ self.prior_theta for a in range(n_arms)]
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(n_arms)]

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        scores = np.zeros((N, self.n_arms))
        pred_cvr_ucb = np.zeros((N, self.n_arms))
        pred_cvr_point = np.zeros((N, self.n_arms))
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            mean = x @ theta
            var = np.sum((x @ self.A_inv[a]) * x, axis=1)
            std = np.sqrt(np.maximum(var, 0.0))
            score = mean + self.alpha * std
            scores[:, a] = score
            pred_cvr_ucb[:, a] = np.clip(score, 0.01, 0.50)
            pred_cvr_point[:, a] = np.clip(mean, 0.01, 0.50)

        chosen = np.argmax(scores, axis=1)
        chosen_ucb = pred_cvr_ucb[np.arange(N), chosen]
        if return_point:
            chosen_point = pred_cvr_point[np.arange(N), chosen]
            return chosen, chosen_ucb, chosen_point
        return chosen, chosen_ucb

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        if len(x_won) == 0:
            return
        for a in range(self.n_arms):
            mask = arm_won == a
            if not np.any(mask):
                continue
            x_a = x_won[mask]
            y_a = y_won[mask]
            delta_A = x_a.T @ x_a
            self.A[a] += delta_A
            self.b[a] += x_a.T @ y_a
            self.A_inv[a] = np.linalg.inv(self.A[a])


class LinearTSPolicy(BanditPolicy):
    """Contextual Linear Thompson Sampling."""

    name = "LinearTS"

    def __init__(self, v: float = 0.3, ridge_lambda: float = 25.0, prior_cvr: float = 0.05):
        self.v = float(v)
        self.ridge_lambda = float(ridge_lambda)
        self.prior_cvr = float(prior_cvr)

    def reset(self, n_arms: int, n_features: int, rng: np.random.Generator) -> None:
        super().reset(n_arms, n_features, rng)
        self.A = [np.eye(self.d) * self.ridge_lambda for _ in range(n_arms)]
        self.prior_theta = np.zeros(self.d)
        self.prior_theta[0] = self.prior_cvr
        self.b = [self.A[a] @ self.prior_theta for a in range(n_arms)]
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(n_arms)]

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        scores = np.zeros((N, self.n_arms))
        pred_cvr_sample = np.zeros((N, self.n_arms))
        pred_cvr_point = np.zeros((N, self.n_arms))
        for a in range(self.n_arms):
            theta_hat = self.A_inv[a] @ self.b[a]
            cov = (self.v ** 2) * self.A_inv[a]
            cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(self.d)
            theta_tilde = self.rng.multivariate_normal(theta_hat, cov)
            score = x @ theta_tilde
            mean = x @ theta_hat
            scores[:, a] = score
            pred_cvr_sample[:, a] = np.clip(score, 0.01, 0.50)
            pred_cvr_point[:, a] = np.clip(mean, 0.01, 0.50)

        chosen = np.argmax(scores, axis=1)
        chosen_sample = pred_cvr_sample[np.arange(N), chosen]
        if return_point:
            chosen_point = pred_cvr_point[np.arange(N), chosen]
            return chosen, chosen_sample, chosen_point
        return chosen, chosen_sample

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        if len(x_won) == 0:
            return
        for a in range(self.n_arms):
            mask = arm_won == a
            if not np.any(mask):
                continue
            x_a = x_won[mask]
            y_a = y_won[mask]
            self.A[a] += x_a.T @ x_a
            self.b[a] += x_a.T @ y_a
            self.A_inv[a] = np.linalg.inv(self.A[a])


class EpsilonGreedyPolicy(BanditPolicy):
    """Contextual Epsilon-Greedy with linear reward model."""

    name = "EpsGreedy"

    def __init__(self, epsilon: float = 0.1, ridge_lambda: float = 25.0, prior_cvr: float = 0.05):
        self.epsilon = float(epsilon)
        self.ridge_lambda = float(ridge_lambda)
        self.prior_cvr = float(prior_cvr)

    def reset(self, n_arms: int, n_features: int, rng: np.random.Generator) -> None:
        super().reset(n_arms, n_features, rng)
        self.A = [np.eye(self.d) * self.ridge_lambda for _ in range(n_arms)]
        self.prior_theta = np.zeros(self.d)
        self.prior_theta[0] = self.prior_cvr
        self.b = [self.A[a] @ self.prior_theta for a in range(n_arms)]
        self.A_inv = [np.linalg.inv(self.A[a]) for a in range(n_arms)]

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        means = np.zeros((N, self.n_arms))
        for a in range(self.n_arms):
            theta = self.A_inv[a] @ self.b[a]
            means[:, a] = x @ theta
        greedy = np.argmax(means, axis=1)
        random_arms = self.rng.integers(0, self.n_arms, size=N)
        explore_mask = self.rng.random(N) < self.epsilon
        chosen = np.where(explore_mask, random_arms, greedy)
        point_cvr = np.clip(means[np.arange(N), chosen], 0.01, 0.50)
        # When exploring, bid at least the prior_cvr so exploratory pulls are not starved in RTB auctions
        bid_cvr = np.where(explore_mask, np.maximum(point_cvr, self.prior_cvr), point_cvr)
        if return_point:
            return chosen, bid_cvr, point_cvr
        return chosen, bid_cvr

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        if len(x_won) == 0:
            return
        for a in range(self.n_arms):
            mask = arm_won == a
            if not np.any(mask):
                continue
            x_a = x_won[mask]
            y_a = y_won[mask]
            self.A[a] += x_a.T @ x_a
            self.b[a] += x_a.T @ y_a
            self.A_inv[a] = np.linalg.inv(self.A[a])


class PureGreedyPolicy(LinUCBPolicy):
    """Pure greedy (LinUCB with alpha=0, no exploration bonus)."""

    name = "PureGreedy"

    def __init__(self, ridge_lambda: float = 25.0, prior_cvr: float = 0.05):
        super().__init__(alpha=0.0, ridge_lambda=ridge_lambda, prior_cvr=prior_cvr)


class RandomPolicy(BanditPolicy):
    name = "Random"

    def __init__(self, default_cvr: float = 0.05):
        self.default_cvr = default_cvr

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        chosen = self.rng.integers(0, self.n_arms, size=N)
        cvr = np.full(N, self.default_cvr)
        if return_point:
            return chosen, cvr, cvr
        return chosen, cvr

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        pass


class OraclePolicy(BanditPolicy):
    """Oracle with full knowledge of all arms' true CVR functions."""

    name = "Oracle"

    def __init__(self, arms: Sequence[CreativeArm]):
        self.arms = list(arms)

    def select(
        self, x: np.ndarray, return_point: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        true_cvr_matrix = np.column_stack([arm.true_cvr(x) for arm in self.arms])
        chosen = np.argmax(true_cvr_matrix, axis=1)
        best_cvr = true_cvr_matrix[np.arange(N), chosen]
        if return_point:
            return chosen, best_cvr, best_cvr
        return chosen, best_cvr

    def update(self, x_won: np.ndarray, arm_won: np.ndarray, y_won: np.ndarray) -> None:
        pass


class BanditValueProvider:
    """Connects a Bandit policy with DayData auction replay.

    Calculates:
    - chosen creative arm per impression
    - predicted value v_bid = pctr * pred_cvr * value_per_conv
    - actual conversion generation on won impressions
    - online policy learning from won conversions
    - cumulative regret vs Oracle
    """

    def __init__(
        self,
        day,
        arms: Sequence[CreativeArm],
        policy: BanditPolicy,
        seed: int = 42,
    ):
        self.day = day
        self.arms = list(arms)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.n_arms = len(self.arms)
        self.d = day.x.shape[1]
        self.policy.reset(self.n_arms, self.d, self.rng)

        # Precompute oracle true CVR for all impressions to measure regret
        self.all_true_cvr = np.column_stack([arm.true_cvr(day.x) for arm in self.arms])
        self.oracle_best_cvr = np.max(self.all_true_cvr, axis=1)

        # State tracking per slot
        self.current_arms = None
        self.cum_regret = 0.0
        self.regret_history = []
        self.arm_counts = np.zeros(self.n_arms, int)

    def values(self, t: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_slot = self.day.x[lo:hi]
        chosen_arms, bid_cvr, point_cvr = self.policy.select(x_slot, return_point=True)
        self.current_arms = chosen_arms

        # True conversion probability for the chosen arm
        chosen_true_cvr = np.array([
            self.arms[arm_id].true_cvr(x_slot[i : i + 1])[0]
            for i, arm_id in enumerate(chosen_arms)
        ])

        # Overall true conversion prob: p_click * p_cvr
        pctr = self.day.pctr[lo:hi]
        ctr_true = self.day.ctr[lo:hi]
        p_true_conv = ctr_true * chosen_true_cvr

        # Predicted value for bidding (using exploration/UCB score)
        v_bid = pctr * bid_cvr * self.day.value_per_conv
        # Predicted value for accounting & ROI pacing (using point estimate of value)
        v_acct = pctr * point_cvr * self.day.value_per_conv

        return v_bid, v_acct, p_true_conv

    def feedback(self, t: int, lo: int, hi: int, win: np.ndarray, rng: np.random.Generator) -> None:
        if self.current_arms is None or len(win) == 0:
            return

        x_slot = self.day.x[lo:hi]
        chosen_arms = self.current_arms
        for a in chosen_arms:
            self.arm_counts[a] += 1

        # Calculate slot regret vs Oracle on all opportunities
        n_req = hi - lo
        oracle_cvr_slot = self.oracle_best_cvr[lo:hi]
        chosen_true_cvr_slot = self.all_true_cvr[lo:hi, :][np.arange(n_req), chosen_arms]
        slot_regret = float(np.sum(oracle_cvr_slot - chosen_true_cvr_slot))
        self.cum_regret += slot_regret
        self.regret_history.append(self.cum_regret)

        # Update bandit on won impressions only (auction censoring!)
        if np.any(win):
            x_won = x_slot[win]
            arm_won = chosen_arms[win]
            ctr_true_won = self.day.ctr[lo:hi][win]
            cvr_true_won = np.array([
                self.arms[arm_id].true_cvr(x_won[i : i + 1])[0]
                for i, arm_id in enumerate(arm_won)
            ])
            # Realized conversion Bernoulli outcome
            p_won = ctr_true_won * cvr_true_won
            y_won = (rng.random(len(p_won)) < p_won).astype(float)
            self.policy.update(x_won, arm_won, y_won)
