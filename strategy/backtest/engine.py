"""月频行业组合回测引擎：由月末权重表生成日度收益，并计算绩效指标。"""

import numpy as np
import pandas as pd

from .. import base, config


def portfolio_daily_returns(weights: pd.DataFrame, ind_ret: pd.DataFrame | None = None,
                            cost_bps: float | None = None, end=None) -> pd.DataFrame:
    """weights: index = 月末调仓日 t（信号日），columns = 行业，行内权重之和为 1（可含 0）。
    返回日度表：ret（组合日收益，含成本）, turnover（调仓日的单边换手率，其余日为 0）。
    组合自 t 的下一交易日起按 weights.loc[t] 持有至下一个调仓日（含）。"""
    ind_ret = base.industry_ret() if ind_ret is None else ind_ret
    cost = (config.BACKTEST_COST_BPS if cost_bps is None else cost_bps) / 1e4
    w = weights.reindex(columns=ind_ret.columns).fillna(0.0)
    rebal_days = w.index
    last_day = pd.Timestamp(end) if end is not None else ind_ret.index[-1]   # 最后一个持有期的截止日
    out_ret, out_turn = [], []
    prev_end_w = pd.Series(0.0, index=ind_ret.columns)  # 上一持有期末的漂移权重
    for k, t in enumerate(rebal_days):
        period_end = rebal_days[k + 1] if k + 1 < len(rebal_days) else last_day
        days = ind_ret.index[(ind_ret.index > t) & (ind_ret.index <= period_end)]
        if len(days) == 0:
            continue
        w0 = w.loc[t]
        turnover = 0.5 * (w0 - prev_end_w).abs().sum()
        r = ind_ret.loc[days].fillna(0.0)
        growth = (1 + r).cumprod()
        pv = (growth * w0).sum(axis=1)
        pv_prev = pv.shift(1).fillna(1.0)
        daily = pv / pv_prev - 1
        daily.iloc[0] -= turnover * cost * 2  # 双边成本在调仓后首日计提
        out_ret.append(daily)
        turn = pd.Series(0.0, index=days)
        turn.iloc[0] = turnover
        out_turn.append(turn)
        end_w = growth.iloc[-1] * w0
        prev_end_w = end_w / end_w.sum() if end_w.sum() > 0 else prev_end_w
    ret = pd.concat(out_ret)
    return pd.DataFrame({"ret": ret, "turnover": pd.concat(out_turn)})


def nav(daily_ret: pd.Series) -> pd.Series:
    return (1 + daily_ret.fillna(0)).cumprod()


def max_drawdown(series_nav: pd.Series) -> float:
    return float((series_nav / series_nav.cummax() - 1).min())


def perf_metrics(port: pd.Series, bench: pd.Series) -> dict:
    """port/bench 为对齐的日收益。超额以净值比（组合净值 / 基准净值）衡量。"""
    df = pd.DataFrame({"p": port, "b": bench}).dropna()
    n = len(df)
    if n < 20:
        return {}
    years = n / 252
    nav_p, nav_b = nav(df.p), nav(df.b)
    ratio = nav_p / nav_b
    m_p = base.period_returns(df[["p"]], _month_ends_within(df.index)).p
    m_b = base.period_returns(df[["b"]], _month_ends_within(df.index)).b
    m_ex = m_p - m_b
    return {
        "start": df.index[0].date(), "end": df.index[-1].date(), "n_days": n,
        "ann_return": nav_p.iloc[-1] ** (1 / years) - 1,
        "bench_ann_return": nav_b.iloc[-1] ** (1 / years) - 1,
        "ann_excess": ratio.iloc[-1] ** (1 / years) - 1,
        "ann_vol": df.p.std() * np.sqrt(252),
        "sharpe": (nav_p.iloc[-1] ** (1 / years) - 1) / (df.p.std() * np.sqrt(252)),
        "info_ratio": m_ex.mean() / m_ex.std() * np.sqrt(12) if m_ex.std() > 0 else np.nan,
        "monthly_win_rate": float((m_ex > 0).mean()),
        "max_drawdown": max_drawdown(nav_p),
        "excess_max_drawdown": max_drawdown(ratio),
        "n_months": int(len(m_ex)),
    }


def yearly_table(port: pd.Series, bench: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"port": port, "bench": bench}).dropna()
    g = (1 + df).groupby(df.index.year).prod() - 1
    g["excess"] = (1 + g.port) / (1 + g.bench) - 1
    return g


def _month_ends_within(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    me = base.month_ends()
    inner = me[(me >= index[0]) & (me <= index[-1])]
    anchor = pd.DatetimeIndex([index[0]]) if index[0] not in inner else pd.DatetimeIndex([])
    return anchor.append(inner)
