"""基础层读取接口（因子层/回测层从这里取数，不直接读原始表）。通过 strategy.base 导出。"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from .calendar import next_trade_day, on_or_after, on_or_before, prev_trade_day, trade_days  # noqa: F401  re-export
from .pit import analyst_avail_date, financial_avail_dates, load_holdings_avail, load_pit_financials  # noqa: F401


@lru_cache(maxsize=1)
def calendar_table() -> pd.DataFrame:
    return pd.read_parquet(config.CALENDAR_PATH)


def month_ends(start=None, end=None) -> pd.DatetimeIndex:
    c = calendar_table()
    d = c.loc[c.is_month_end, "trade_date"]
    if start is not None:
        d = d[d >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    return pd.DatetimeIndex(d)


def week_ends(start=None, end=None) -> pd.DatetimeIndex:
    c = calendar_table()
    d = c.loc[c.is_week_end, "trade_date"]
    if start is not None:
        d = d[d >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    return pd.DatetimeIndex(d)


@lru_cache(maxsize=1)
def industry_list() -> pd.DataFrame:
    return pd.read_parquet(config.INDUSTRY_LIST_PATH)


def industry_names() -> list[str]:
    return industry_list().l1_name.tolist()


@lru_cache(maxsize=8)
def stock_panel(field: str) -> pd.DataFrame:
    """field ∈ {ret, turnover, amount, mcap, fmcap, traded}；交易日 × 股票，float32（traded 为 bool）。"""
    return pd.read_parquet(config.STOCK_PANEL_PATHS[field])


@lru_cache(maxsize=1)
def stock_industry() -> pd.DataFrame:
    """交易日 × 股票 → 行业序号 int8（-1 无归属）。"""
    return pd.read_parquet(config.STOCK_INDUSTRY_PATH)


def industry_of(stock_codes: pd.Series, dates: pd.Series) -> pd.Series:
    """按 (股票, 日期) 查行业名称；日期需为面板内交易日，无归属返回 NaN。"""
    m = stock_industry()
    names = np.array(industry_names() + [None], dtype=object)
    di = m.index.get_indexer(pd.to_datetime(dates).values.astype("datetime64[ns]"))
    ci = m.columns.get_indexer(stock_codes.astype(str).values)
    ok = (di >= 0) & (ci >= 0)
    idx = np.full(len(di), -1, dtype=int)
    idx[ok] = m.values[di[ok], ci[ok]]
    return pd.Series(names[np.where(idx < 0, len(names) - 1, idx)], index=stock_codes.index)


@lru_cache(maxsize=1)
def industry_ret() -> pd.DataFrame:
    return pd.read_parquet(config.INDUSTRY_RET_PATH)


@lru_cache(maxsize=1)
def industry_close() -> pd.DataFrame:
    return pd.read_parquet(config.INDUSTRY_CLOSE_PATH)


@lru_cache(maxsize=1)
def benchmark() -> pd.DataFrame:
    return pd.read_parquet(config.BENCHMARK_PATH)


def period_returns(daily_ret: pd.DataFrame, period_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """把日收益按期末日序列复利聚合为期收益。period_ends[0] 为起始锚点（不出现在结果中），
    返回 period_ends[1:] 各期收益：level(e_k) / level(e_{k-1}) - 1。"""
    growth = (1 + daily_ret.fillna(0)).cumprod()
    level = growth.reindex(period_ends)
    return (level / level.shift(1) - 1).iloc[1:]


def _ends_with_anchor(ends_all: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    ends = ends_all[ends_all <= pd.Timestamp(end)] if end is not None else ends_all
    if start is not None:
        start = pd.Timestamp(start)
        before = ends[ends < start]
        anchor = before[-1:] if len(before) else ends[:1]
        ends = anchor.append(ends[ends >= start])
    return ends


def monthly_returns(daily_ret: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """月收益，index 为月末交易日；起始锚点取 start 之前最近的月末。"""
    return period_returns(daily_ret, _ends_with_anchor(month_ends(), start, end))


def weekly_returns(daily_ret: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """周收益（周内最后一个交易日收盘对上周最后交易日收盘），index 为周末交易日。"""
    return period_returns(daily_ret, _ends_with_anchor(week_ends(), start, end))
