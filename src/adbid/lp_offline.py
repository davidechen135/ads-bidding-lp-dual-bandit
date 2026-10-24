"""Hindsight (offline) LP for ROI- and budget-constrained auto-bidding.

Primal (P), given the whole day of requests in hindsight:

    max_x   sum_i v_i x_i
    s.t.    sum_i c_i x_i                <= B      (dual lambda >= 0)
            sum_i (R c_i - v_i) x_i      <= 0      (dual mu     >= 0)
            0 <= x_i <= 1

v_i : (predicted) value of impression i, c_i : its market price (cost when
won in a second-price auction), B : daily budget, R : target ROI.

Lagrangian relaxation gives, for fixed (lambda, mu),

    x_i* = 1  iff  (1 + mu) v_i >= (lambda + mu R) c_i
              iff  c_i <= b_i := (1 + mu) v_i / (lambda + mu R)

i.e. bidding b_i in a second-price auction implements the optimal primal
solution (up to the <=2 fractional variables of a basic LP solution).

The dual (D) is

    min_{lambda, mu >= 0}  lambda B + sum_i [ (1+mu) v_i - (lambda + mu R) c_i ]^+

and strong duality P* = D* holds (both are feasible & bounded LPs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


@dataclass
class LPResult:
    x: np.ndarray
    primal_obj: float
    lam: float
    mu: float
    dual_obj: float
    status: str

    @property
    def duality_gap(self) -> float:
        return abs(self.primal_obj - self.dual_obj) / max(1e-12, abs(self.primal_obj))


def dual_objective(lam: float, mu: float, v: np.ndarray, c: np.ndarray, B: float, R: float) -> float:
    return float(lam * B + np.maximum(0.0, (1 + mu) * v - (lam + mu * R) * c).sum())


def optimal_bid(v: np.ndarray, lam: float, mu: float, R: float, alpha_max: float = np.inf) -> np.ndarray:
    """b = (1 + mu) v / (lambda + mu R), with bid multiplier capped at alpha_max."""
    denom = lam + mu * R
    alpha = np.inf if denom <= 0 else (1.0 + mu) / denom
    alpha = min(alpha, alpha_max)
    return alpha * v


def solve_hindsight_lp(
    v: np.ndarray | object,
    c: np.ndarray | float | None = None,
    B: float | None = None,
    R: float | None = None,
) -> LPResult:
    """Solve (P) with HiGHS and read the dual multipliers (lambda, mu).

    Accepts either `(day, B, R)` or `(v, c, B, R)`.
    """
    if hasattr(v, "v_hat") and hasattr(v, "price"):
        day = v
        # Called as solve_hindsight_lp(day, B, R)
        actual_B = float(c)
        actual_R = float(B)
        v = day.v_hat
        c = day.price
        B = actual_B
        R = actual_R
    v = np.asarray(v, float)
    c = np.asarray(c, float)
    B = float(B)
    R = float(R)
    n = len(v)
    rows = np.r_[np.zeros(n, int), np.ones(n, int)]
    cols = np.r_[np.arange(n), np.arange(n)]
    data = np.r_[c, R * c - v]
    A = csr_matrix((data, (rows, cols)), shape=(2, n))
    res = linprog(
        -v,
        A_ub=A,
        b_ub=np.array([B, 0.0]),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    # HiGHS marginals are d(min obj)/d(b_ub) <= 0 ; we minimised -v.x
    lam = float(max(0.0, -res.ineqlin.marginals[0]))
    mu = float(max(0.0, -res.ineqlin.marginals[1]))
    primal = float(v @ res.x)
    return LPResult(
        x=res.x,
        primal_obj=primal,
        lam=lam,
        mu=mu,
        dual_obj=dual_objective(lam, mu, v, c, B, R),
        status=res.message,
    )
