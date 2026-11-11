"""PIT（point-in-time）可用日对齐。

约定：可用日为交易日，表示"在该交易日收盘计算信号时可以使用该数据"。

1. 财报（业绩报表 / 资产负债表等同一报告期的所有报表项）：
   可用日 = 首次公告日（资产负债表 notice_date）的下一交易日；缺公告日时用法定披露截止日
   （Q1 4/30、Q2 8/31、Q3 10/31、Q4 次年 4/30）的下一交易日。
   若公告日晚于截止日超过 ANOMALY_DAYS 天，视为数据源把更正/补充公告当成了首次公告，改用截止日。
   产出 pit_financials.pq：stock_code, report_date, notice_date, deadline, avail_date, avail_source。
2. 基金季报持仓：沿用数据层给出的 disclosure_deadlines（前十大：季末后第 15 个交易日；全持仓：8/31、3/31 后首个交易日）。
3. 分析师报告：报告日 create_date 的下一交易日（函数 analyst_avail_date）。
"""

import pandas as pd

from etl.utils import get_logger

from .. import config
from .calendar import next_trade_day

logger = get_logger("strategy.pit")

ANOMALY_DAYS = 45


def statutory_deadline(report_date: pd.Series) -> pd.Series:
    rd = pd.to_datetime(report_date)
    q = rd.dt.quarter
    year = rd.dt.year
    out = pd.Series(pd.NaT, index=rd.index, dtype="datetime64[ns]")
    out[q == 1] = pd.to_datetime(year[q == 1].astype(str) + "-04-30")
    out[q == 2] = pd.to_datetime(year[q == 2].astype(str) + "-08-31")
    out[q == 3] = pd.to_datetime(year[q == 3].astype(str) + "-10-31")
    out[q == 4] = pd.to_datetime((year[q == 4] + 1).astype(str) + "-04-30")
    return out


def build_pit_financials() -> pd.DataFrame:
    bs = pd.read_parquet(config.BALANCE_SHEET_PATH, columns=["stock_code", "report_date", "notice_date"])
    bs["report_date"] = bs["report_date"].astype("datetime64[ns]")
    bs["notice_date"] = bs["notice_date"].astype("datetime64[ns]")
    bs["deadline"] = statutory_deadline(bs.report_date)
    late_days = (bs.notice_date - bs.deadline).dt.days
    anomaly = bs.notice_date.isna() | (late_days > ANOMALY_DAYS)
    base_date = bs.notice_date.where(~anomaly, bs.deadline)
    bs["avail_date"] = next_trade_day(base_date).values
    bs["avail_source"] = "notice"
    bs.loc[anomaly, "avail_source"] = "deadline"
    logger.info("PIT 财报: %d 行; 公告日缺失 %d, 晚于截止日>%d天(改用截止日) %d; 公告日-截止日 分位数(天) %s",
                len(bs), bs.notice_date.isna().sum(), ANOMALY_DAYS, (late_days > ANOMALY_DAYS).sum(),
                late_days.quantile([0.5, 0.9, 0.99]).round(0).to_dict())
    return bs[["stock_code", "report_date", "notice_date", "deadline", "avail_date", "avail_source"]]


def load_pit_financials() -> pd.DataFrame:
    return pd.read_parquet(config.PIT_FINANCIALS_PATH)


def financial_avail_dates(stock_codes: pd.Series, report_dates: pd.Series,
                          pit: pd.DataFrame | None = None) -> pd.Series:
    """按 (股票, 报告期) 查询财报可用日；无记录的 (股票, 报告期) 退化为法定截止日的下一交易日。"""
    pit = load_pit_financials() if pit is None else pit
    q = pd.DataFrame({"stock_code": stock_codes.values,
                      "report_date": pd.to_datetime(report_dates).values.astype("datetime64[ns]")})
    m = q.merge(pit[["stock_code", "report_date", "avail_date"]], on=["stock_code", "report_date"], how="left")
    fallback = next_trade_day(statutory_deadline(m.report_date))
    return pd.Series(m.avail_date.fillna(fallback).values, index=stock_codes.index)


def load_holdings_avail() -> pd.DataFrame:
    d = pd.read_parquet(config.DISCLOSURE_PATH)
    for c in ("report_date", "top10_avail", "full_avail"):
        d[c] = d[c].astype("datetime64[ns]")
    return d[["report_date", "top10_avail", "full_avail"]]


def analyst_avail_date(create_dates: pd.Series) -> pd.Series:
    return next_trade_day(create_dates)
