"""稳健版权重优化（研报 3.2 节）：在高弹性版选出的 n 个指数上求解

    min_w Σ_j ((wᵀF_j − f_bench_j) / σ_j)²
    s.t.  Σ w = 1, w ≥ 0, w ≤ ETF_W_MAX, |wᵀF_j − f_bench_j| ≤ δ_j（beta δ=ETF_BETA_TOL，其他 ETF_STYLE_TOL）

F：n × J 指数风格暴露（成份等权），f_bench：基准暴露，σ_j：样本内 pooled 标准差。scipy SLSQP；不可行或求解失败时
退化为等权并记录。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config


def solve(F: np.ndarray, f_bench: np.ndarray, sigma: np.ndarray, factor_names: list[str], w_max: float | None = None,
          beta_tol: float | None = None, style_tol: float | None = None) -> tuple[np.ndarray, str]:
    """参数缺省取 config；style_tol / beta_tol 为 inf 表示不设该偏离约束。"""
    w_max = config.ETF_W_MAX if w_max is None else w_max
    beta_tol = config.ETF_BETA_TOL if beta_tol is None else beta_tol
    style_tol = config.ETF_STYLE_TOL if style_tol is None else style_tol
    n, J = F.shape
    tol = np.array([beta_tol if f == "BETA" else style_tol for f in factor_names])
    w0 = np.full(n, 1.0 / n)

    def obj(w):
        d = (w @ F - f_bench) / sigma
        return float(d @ d)

    def grad(w):
        d = (w @ F - f_bench) / sigma
        return 2.0 * (F / sigma) @ d

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    for j in range(J):
        if not np.isfinite(tol[j]):
            continue
        cons.append({"type": "ineq", "fun": (lambda w, j=j: tol[j] - (w @ F[:, j] - f_bench[j])), "jac": (lambda w, j=j: -F[:, j])})
        cons.append({"type": "ineq", "fun": (lambda w, j=j: tol[j] + (w @ F[:, j] - f_bench[j])), "jac": (lambda w, j=j: F[:, j])})
    bounds = [(0.0, w_max)] * n
    if n * w_max < 1.0:
        return w0, "infeasible_wmax"
    res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    if not res.success or w.sum() <= 0:
        # 放松风格偏离约束再试（只保留权重约束）
        res2 = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons[:1], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(res2.x, 0.0, None)
        if not res2.success or w.sum() <= 0:
            return w0, "fallback_equal"
        return w / w.sum(), "relaxed"
    return w / w.sum(), "ok"


def pooled_sigma(barra_index: pd.DataFrame, factor_names: list[str]) -> np.ndarray:
    return barra_index[factor_names].std(ddof=1).values


def solve_penalty(F: np.ndarray, f_bench: np.ndarray, sigma: np.ndarray, factor_names: list[str], w_max: float,
                  beta_tol: float, lam: float) -> tuple[np.ndarray, str]:
    """惩罚项版：min ||w − w_eq||² + λ·Σ_j ((wᵀF_j − f_bench_j)/σ_j)²，s.t. Σw=1, 0 ≤ w ≤ w_max，仅 beta 为硬约束（|偏离| ≤ beta_tol）。
    λ=0 退化为等权，λ→∞ 逼近无风格硬约束的最小偏离解；用于诊断风格约束的作用方式。"""
    n, J = F.shape
    w0 = np.full(n, 1.0 / n)
    Fs = F / sigma
    fb = f_bench / sigma
    jb = [j for j, f in enumerate(factor_names) if f == "BETA"]

    def obj(w):
        d = w @ Fs - fb
        return float((w - w0) @ (w - w0) + lam * (d @ d))

    def grad(w):
        d = w @ Fs - fb
        return 2.0 * (w - w0) + 2.0 * lam * Fs @ d

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    for j in jb:
        if np.isfinite(beta_tol):
            cons.append({"type": "ineq", "fun": (lambda w, j=j: beta_tol - (w @ F[:, j] - f_bench[j])), "jac": (lambda w, j=j: -F[:, j])})
            cons.append({"type": "ineq", "fun": (lambda w, j=j: beta_tol + (w @ F[:, j] - f_bench[j])), "jac": (lambda w, j=j: F[:, j])})
    if n * w_max < 1.0:
        return w0, "infeasible_wmax"
    res = minimize(obj, w0, jac=grad, bounds=[(0.0, w_max)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    if not res.success or w.sum() <= 0:
        res2 = minimize(obj, w0, jac=grad, bounds=[(0.0, w_max)] * n, constraints=cons[:1], method="SLSQP",
                        options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(res2.x, 0.0, None)
        if not res2.success or w.sum() <= 0:
            return w0, "fallback_equal"
        return w / w.sum(), "relaxed"
    return w / w.sum(), "ok"
