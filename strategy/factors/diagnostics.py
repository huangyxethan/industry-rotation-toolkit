"""因子快速诊断：覆盖率、月度 Rank IC（因子 vs 下月行业超额收益）、因子间相关性。

用于构建时的 sanity check，不替代回测层的分组测试。
IC 定义：每个月末 t，factor[t] 与 行业 t→t+1 月收益减去行业等权月收益 的 Spearman 相关系数。
"""

import numpy as np
import pandas as pd

from .. import base, config


def forward_excess_monthly_returns() -> pd.DataFrame:
    """index = 月末 t，值为 t 到下一月末的行业超额收益（相对 31 行业等权月收益）。"""
    m = base.monthly_returns(base.industry_ret())
    excess = m.sub(m.mean(axis=1), axis=0)
    return excess.shift(-1)


def rank_ic(factor: pd.DataFrame, sign: int = 1, start=None, end=None) -> pd.Series:
    fwd = forward_excess_monthly_returns()
    f = factor.reindex(fwd.index) * sign
    start = pd.Timestamp(start or config.BACKTEST_START)
    end = pd.Timestamp(end or config.BACKTEST_END)
    ics = {}
    for t in f.index:
        if t < start or t > end:
            continue
        a, b = f.loc[t], fwd.loc[t]
        ok = a.notna() & b.notna()
        if ok.sum() >= 10:
            ics[t] = a[ok].rank().corr(b[ok].rank())
    return pd.Series(ics, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    if ic.empty:
        return {"n": 0}
    return {"n": len(ic), "ic_mean": round(ic.mean(), 4), "ic_std": round(ic.std(), 4),
            "icir": round(ic.mean() / ic.std(), 3), "t_stat": round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2),
            "pos_rate": round((ic > 0).mean(), 3)}


def coverage(factor: pd.DataFrame) -> pd.Series:
    return factor.notna().sum(axis=1)


def factor_corr(factors: dict[str, pd.DataFrame], start=None) -> pd.DataFrame:
    """因子间截面 Spearman 相关系数的时间平均。"""
    names = list(factors)
    start = pd.Timestamp(start or config.BACKTEST_START)
    idx = factors[names[0]].index
    idx = idx[idx >= start]
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            vals = []
            for t in idx:
                if t not in factors[a].index or t not in factors[b].index:
                    continue
                x, y = factors[a].loc[t], factors[b].loc[t]
                ok = x.notna() & y.notna()
                if ok.sum() >= 10:
                    vals.append(x[ok].rank().corr(y[ok].rank()))
            out.loc[a, b] = out.loc[b, a] = np.mean(vals) if vals else np.nan
    return out
