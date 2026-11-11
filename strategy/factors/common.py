"""因子层公共工具：因子规格与存取、标准化、按日期采样、个股统计量到行业的聚合。"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from .. import base


@dataclass
class FactorSpec:
    name: str
    dimension: str            # market / enterprise / analyst / manager
    description: str
    sign: int                 # +1 越高越好；-1 越低越好（如拥挤度）
    build: Callable[[], dict[str, pd.DataFrame]]  # 返回 {因子名: 月频因子表}，可含辅助/分项因子
    outputs: list[str] = field(default_factory=list)  # build 返回的全部表名（首个为主因子）


def factor_path(name: str):
    return config.FACTOR_DIR / f"{name}.pq"


def save(name: str, df: pd.DataFrame) -> None:
    config.FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).astype("datetime64[ns]")
    out.index.name = "date"
    out.columns = [str(c) for c in out.columns]
    out.astype("float64").to_parquet(factor_path(name))


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(factor_path(name))


# ---------- 标准化 ----------

def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面（按行）Z-score。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)


def ts_zscore(df: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """时序（按列）滚动 Z-score，窗口含当期。"""
    r = df.rolling(window, min_periods=min_periods)
    return (df - r.mean()) / r.std(ddof=1)


def asof_sample(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """取每个 date 之前（含当日）最近一行。"""
    return df.reindex(df.index.union(dates)).ffill().reindex(dates)


# ---------- 个股 → 行业 ----------

def industry_mean(stat: pd.DataFrame, valid: pd.DataFrame | None = None) -> pd.DataFrame:
    """把个股月频统计量 stat（index=月末日, columns=股票）按当日行业归属等权平均为行业表。

    valid 为同形状 bool 表（可选），False 的样本不参与。行业归属取 base.stock_industry() 当日值。
    """
    member = base.stock_industry()
    n_ind = len(base.industry_names())
    m = member.reindex(index=stat.index, columns=stat.columns).fillna(-1).astype(int).values
    x = stat.values.astype("float64")
    ok = ~np.isnan(x) & (m >= 0)
    if valid is not None:
        ok &= valid.reindex(index=stat.index, columns=stat.columns).fillna(False).values
    out = np.full((len(stat), n_ind), np.nan)
    for i in range(len(stat)):
        sel = ok[i]
        if not sel.any():
            continue
        s = np.bincount(m[i, sel], weights=x[i, sel], minlength=n_ind)
        c = np.bincount(m[i, sel], minlength=n_ind)
        out[i] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return pd.DataFrame(out, index=stat.index, columns=base.industry_names())
