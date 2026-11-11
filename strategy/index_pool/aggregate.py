"""成员矩阵聚合：把个股层统计量按指数成份聚合为指数表（多对多成员关系）。

所有函数以 base.members_at(t)（t 时点生效的成份长表）为成员来源；每个日期取最近一次半年末快照。
"""

import numpy as np
import pandas as pd

from .. import config
from . import base as ipb


def _member_matrix(t: pd.Timestamp, stock_index: pd.Index) -> tuple[np.ndarray, list[str]]:
    """t 时点的 0/1 成员矩阵（股票 × 指数），股票按 stock_index 排列；不在 stock_index 的成份忽略。"""
    m = ipb.members_at(t)
    codes = ipb.index_codes()
    M = np.zeros((len(stock_index), len(codes)), dtype="float64")
    if m.empty:
        return M, codes
    si = stock_index.get_indexer(m.stock_code.values)
    ci = pd.Index(codes).get_indexer(m.index_code.values)
    ok = (si >= 0) & (ci >= 0)
    M[si[ok], ci[ok]] = 1.0
    return M, codes


def member_mean(stat: pd.DataFrame, valid: pd.DataFrame | None = None, min_members: int | None = None) -> pd.DataFrame:
    """stat（日期 × 股票）按各日期生效的成份等权平均 → 日期 × 指数；有效成份数 < min_members 记 NaN。"""
    min_members = config.INDEX_POOL_MIN_MEMBERS if min_members is None else min_members
    codes = ipb.index_codes()
    x = stat.values.astype("float64")
    ok = ~np.isnan(x)
    if valid is not None:
        ok &= valid.reindex(index=stat.index, columns=stat.columns).fillna(False).values
    out = np.full((len(stat), len(codes)), np.nan)
    snap = ipb.snapshot_for(pd.DatetimeIndex(stat.index))
    cache: dict = {}
    for i, t in enumerate(stat.index):
        s = snap.iloc[i]
        if pd.isna(s):
            continue
        if s not in cache:
            cache[s] = _member_matrix(t, stat.columns)[0]
        M = cache[s]
        xi = np.where(ok[i], x[i], 0.0)
        num = xi @ M
        cnt = ok[i].astype("float64") @ M
        out[i] = np.where(cnt >= min_members, num / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, index=stat.index, columns=codes)


def member_mad_mean(stat: pd.DataFrame, k: float, min_members: int) -> pd.DataFrame:
    """指数内 median ± k·MAD 截尾后的等权平均（逐日、逐指数）。"""
    codes = ipb.index_codes()
    out = np.full((len(stat), len(codes)), np.nan)
    snap = ipb.snapshot_for(pd.DatetimeIndex(stat.index))
    cache: dict = {}
    for i, t in enumerate(stat.index):
        s = snap.iloc[i]
        if pd.isna(s):
            continue
        if s not in cache:
            m = ipb.members_at(t)
            cache[s] = m[m.stock_code.isin(stat.columns)]
        m = cache[s]
        row = pd.Series(stat.iloc[i].values, index=stat.columns)
        v = row.reindex(m.stock_code.values).values
        d = pd.DataFrame({"v": v, "g": m.index_code.values}).dropna(subset=["v"])
        if d.empty:
            continue
        grp = d.groupby("g").v
        med = grp.transform("median")
        mad = (d.v - med).abs().groupby(d.g).transform("median")
        clipped = d.v.clip(lower=med - k * mad, upper=med + k * mad)
        agg = clipped.groupby(d.g).agg(["mean", "count"])
        agg = agg[agg["count"] >= min_members]
        ci = pd.Index(codes).get_indexer(agg.index)
        out[i, ci] = agg["mean"].values
    return pd.DataFrame(out, index=stat.index, columns=codes)


def explode_members(df: pd.DataFrame, t: pd.Timestamp, stock_col: str = "stock_code", out_col: str = "index_code") -> pd.DataFrame:
    """把股票键的表按 t 时点成份展开为 (原行, 指数) 行：一只股票属于几个指数就复制几行；不属于任何指数的行丢弃。"""
    m = ipb.members_at(t)[["index_code", "stock_code"]].rename(columns={"index_code": out_col, "stock_code": stock_col})
    return df.merge(m, on=stock_col, how="inner")


def member_count_share(stocks: pd.Series, t: pd.Timestamp) -> pd.Series:
    """给定股票集合（Series 或 array），各指数中含有的数量 / 集合大小。"""
    s = pd.Series(np.asarray(stocks)).drop_duplicates()
    m = ipb.members_at(t)
    cnt = m[m.stock_code.isin(s)].groupby("index_code").size()
    return (cnt / max(len(s), 1)).reindex(ipb.index_codes()).fillna(0.0)
