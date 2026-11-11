"""单因子分组测试：多头（得分最高 N 个行业）、空头（最低 N 个）、多空对冲、N 组分组，均等权、月频。

得分 = sign × 因子值；当期有效行业数 < BACKTEST_MIN_INDUSTRIES 时该月持有基准（行业等权）。
"""

import numpy as np
import pandas as pd

from .. import base, config
from ..factors import REGISTRY, load
from .engine import perf_metrics, portfolio_daily_returns, yearly_table


def _score(name: str) -> pd.DataFrame:
    spec = REGISTRY[name]
    return (load(name) * spec.sign).drop(columns=list(config.BACKTEST_EXCLUDE_INDUSTRIES), errors="ignore")


def signal_dates(start, end) -> pd.DatetimeIndex:
    """信号日序列：start 之前最近的月末（使首个持有期覆盖 start 所在月）至 end 之前的月末。"""
    me = base.month_ends(None, end)
    before = me[me < pd.Timestamp(start)]
    first = before[-1:] if len(before) else me[:1]
    return first.append(me[(me >= pd.Timestamp(start)) & (me < pd.Timestamp(end))])


def selection_weights(score: pd.DataFrame, start, end, top_n: int, side: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """side='long' 取最高 N，'short' 取最低 N；返回 (权重表, 持仓清单表)。"""
    me = signal_dates(start, end)
    s = score.reindex(me)
    names = list(score.columns)
    w = pd.DataFrame(0.0, index=me, columns=names)
    holds = []
    for t in me:
        row = s.loc[t].dropna()
        if len(row) < config.BACKTEST_MIN_INDUSTRIES:
            w.loc[t] = 1.0 / len(names)
            holds.append({"date": t, "n_valid": len(row), "industries": "<benchmark>"})
            continue
        picks = row.nlargest(top_n).index if side == "long" else row.nsmallest(top_n).index
        w.loc[t, picks] = 1.0 / len(picks)
        holds.append({"date": t, "n_valid": len(row), "industries": "、".join(picks)})
    return w, pd.DataFrame(holds)


def group_weights(score: pd.DataFrame, start, end, n_groups: int) -> dict[int, pd.DataFrame]:
    """按得分从高到低分 n_groups 组（组 1 最高），各组内等权。"""
    me = signal_dates(start, end)
    s = score.reindex(me)
    names = list(score.columns)
    ws = {g: pd.DataFrame(0.0, index=me, columns=names) for g in range(1, n_groups + 1)}
    for t in me:
        row = s.loc[t].dropna().sort_values(ascending=False)
        if len(row) < config.BACKTEST_MIN_INDUSTRIES:
            for g in ws:
                ws[g].loc[t] = 1.0 / len(names)
            continue
        bins = np.array_split(row.index.values, n_groups)
        for g, members in enumerate(bins, 1):
            ws[g].loc[t, members] = 1.0 / len(members)
    return ws


def run(name: str, start=None, end=None, top_n: int | None = None, n_groups: int | None = None) -> dict:
    start = pd.Timestamp(start or config.BACKTEST_START)
    end = pd.Timestamp(end or config.BACKTEST_END)
    top_n = top_n or config.BACKTEST_TOP_N
    n_groups = n_groups or config.BACKTEST_N_GROUPS
    score = _score(name)
    bench = base.benchmark()[config.BACKTEST_BENCHMARK]

    w_long, holds_long = selection_weights(score, start, end, top_n, "long")
    w_short, holds_short = selection_weights(score, start, end, top_n, "short")
    r_long = portfolio_daily_returns(w_long, end=end)
    r_short = portfolio_daily_returns(w_short, end=end)
    idx = r_long.index
    b = bench.reindex(idx).fillna(0.0)
    daily = pd.DataFrame({"long": r_long.ret, "short": r_short.ret, "bench": b,
                          "long_short": r_long.ret - r_short.ret, "turnover_long": r_long.turnover}, index=idx)
    daily["nav_long"] = (1 + daily.long).cumprod()
    daily["nav_short"] = (1 + daily.short).cumprod()
    daily["nav_bench"] = (1 + daily.bench).cumprod()
    daily["excess_nav"] = daily.nav_long / daily.nav_bench
    daily["nav_long_short"] = (1 + daily.long_short).cumprod()

    metrics = {"long": perf_metrics(daily.long, daily.bench), "short": perf_metrics(daily.short, daily.bench)}
    ls = perf_metrics(daily.long_short, pd.Series(0.0, index=idx))
    metrics["long_short"] = {k: ls[k] for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")}
    metrics["long"]["avg_turnover"] = float(r_long.turnover[r_long.turnover > 0].mean())

    groups = {}
    for g, w in group_weights(score, start, end, n_groups).items():
        rg = portfolio_daily_returns(w, end=end).ret
        m = perf_metrics(rg, b)
        groups[g] = {"ann_return": m["ann_return"], "ann_excess": m["ann_excess"], "info_ratio": m["info_ratio"]}
    groups_df = pd.DataFrame(groups).T
    groups_df.index.name = "group"
    holdings = holds_long.merge(holds_short, on=["date", "n_valid"], suffixes=("_long", "_short"))
    return {"name": name, "daily": daily, "metrics": metrics, "yearly": yearly_table(daily.long, daily.bench),
            "groups": groups_df, "holdings": holdings}
