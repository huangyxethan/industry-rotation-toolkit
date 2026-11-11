"""交易日历与调仓日序列。

calendar.pq 字段：
- trade_date            交易日
- year, month           自然年月
- is_month_end          该月最后一个交易日（月频调仓日）
- is_week_end           该周（周一至周日）最后一个交易日（趋势因子周度收益的取样日）
- is_quarter_end        该季度最后一个交易日
- month_id, week_id     单调递增的月/周编号，便于 groupby
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config


@lru_cache(maxsize=1)
def trade_days() -> pd.DatetimeIndex:
    """全部交易日（含面板区间之外），升序。"""
    cal = pd.read_parquet(config.TRADE_CAL_PATH).trade_date.astype("datetime64[ns]")
    return pd.DatetimeIndex(sorted(cal.unique()))


def panel_days() -> pd.DatetimeIndex:
    cal = trade_days()
    return cal[(cal >= config.PANEL_START) & (cal <= config.PANEL_END)]


def build_calendar_table() -> pd.DataFrame:
    days = panel_days()
    df = pd.DataFrame({"trade_date": days})
    df["year"] = days.year
    df["month"] = days.month
    period_m = days.to_period("M")
    period_w = days.to_period("W")  # 周一至周日
    period_q = days.to_period("Q")
    df["is_month_end"] = period_m != np.append(period_m[1:], None)
    df["is_week_end"] = period_w != np.append(period_w[1:], None)
    df["is_quarter_end"] = period_q != np.append(period_q[1:], None)
    df["month_id"] = pd.factorize(period_m)[0]
    df["week_id"] = pd.factorize(period_w)[0]
    return df


def next_trade_day(dates: pd.Series | pd.DatetimeIndex, n: int = 1) -> pd.Series:
    """严格晚于 dates 的第 n 个交易日（n>=1）；超出日历范围返回 NaT。"""
    cal = trade_days().values
    d = pd.to_datetime(pd.Series(np.asarray(dates))).values.astype("datetime64[ns]")
    pos = np.searchsorted(cal, d, side="right") + (n - 1)
    out = np.where(pos < len(cal), cal[np.minimum(pos, len(cal) - 1)], np.datetime64("NaT"))
    return pd.Series(out, index=getattr(dates, "index", None))


def on_or_after(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """不早于 dates 的第一个交易日。"""
    cal = trade_days().values
    d = pd.to_datetime(pd.Series(np.asarray(dates))).values.astype("datetime64[ns]")
    pos = np.searchsorted(cal, d, side="left")
    out = np.where(pos < len(cal), cal[np.minimum(pos, len(cal) - 1)], np.datetime64("NaT"))
    return pd.Series(out, index=getattr(dates, "index", None))


def prev_trade_day(dates: pd.Series | pd.DatetimeIndex, n: int = 1) -> pd.Series:
    """严格早于 dates 的第 n 个交易日。"""
    cal = trade_days().values
    d = pd.to_datetime(pd.Series(np.asarray(dates))).values.astype("datetime64[ns]")
    pos = np.searchsorted(cal, d, side="left") - n
    out = np.where(pos >= 0, cal[np.maximum(pos, 0)], np.datetime64("NaT"))
    return pd.Series(out, index=getattr(dates, "index", None))


def on_or_before(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """不晚于 dates 的最后一个交易日。"""
    cal = trade_days().values
    d = pd.to_datetime(pd.Series(np.asarray(dates))).values.astype("datetime64[ns]")
    pos = np.searchsorted(cal, d, side="right") - 1
    out = np.where(pos >= 0, cal[np.maximum(pos, 0)], np.datetime64("NaT"))
    return pd.Series(out, index=getattr(dates, "index", None))
