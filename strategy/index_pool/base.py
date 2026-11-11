"""指数级基础层读取接口：指数池、成员矩阵、指数收益。

文件（.data/base/index_pool/）：
  index_list.pq          index_code, index_name, is_broad, n_etf, in_pool, exclude_reason, history_start, cons_start, cons_end
  membership_long.pq     长表：snapshot_date（成份快照日）, index_code, stock_code（6 位）, weight
  index_ret_daily.pq     交易日 × 指数 日收益（跨断档不计）
  index_close_daily.pq   交易日 × 指数 收盘价
读取接口：index_list / index_codes / pool_codes / membership_long / snapshot_for / members_at / index_ret / index_close

本公开版本不含从数据源构建这些文件的流程；scripts/make_synthetic_data.py 生成同格式的合成文件。
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config

D = config.INDEX_POOL_DIR
INDEX_LIST_PATH = D / "index_list.pq"
MEMBERSHIP_PATH = D / "membership_long.pq"
INDEX_RET_PATH = D / "index_ret_daily.pq"
INDEX_CLOSE_PATH = D / "index_close_daily.pq"


@lru_cache(maxsize=1)
def index_list() -> pd.DataFrame:
    return pd.read_parquet(INDEX_LIST_PATH)


def index_codes() -> list[str]:
    """全部计算对象的指数代码（含 in_pool=False 者）。"""
    return index_list().index_code.tolist()


def pool_codes() -> list[str]:
    """in_pool=True 的指数代码。"""
    il = index_list()
    return il[il.in_pool].index_code.tolist()


@lru_cache(maxsize=1)
def membership_long() -> pd.DataFrame:
    return pd.read_parquet(MEMBERSHIP_PATH)


def snapshot_for(dates: pd.DatetimeIndex) -> pd.Series:
    """每个日期对应的最近一次成份快照日（≤ 日期），无则 NaT。"""
    snaps = np.sort(membership_long().snapshot_date.unique())
    k = np.searchsorted(snaps, dates.values.astype("datetime64[ns]"), side="right") - 1
    out = np.where(k >= 0, snaps[np.maximum(k, 0)], np.datetime64("NaT"))
    return pd.Series(out, index=dates)


def members_at(t: pd.Timestamp) -> pd.DataFrame:
    """t 时点生效的成份长表（index_code, stock_code, weight）。"""
    snap = snapshot_for(pd.DatetimeIndex([t])).iloc[0]
    if pd.isna(snap):
        return pd.DataFrame(columns=["index_code", "stock_code", "weight"])
    m = membership_long()
    return m[m.snapshot_date == snap][["index_code", "stock_code", "weight"]]


@lru_cache(maxsize=1)
def index_ret() -> pd.DataFrame:
    return pd.read_parquet(INDEX_RET_PATH)


@lru_cache(maxsize=1)
def index_close() -> pd.DataFrame:
    return pd.read_parquet(INDEX_CLOSE_PATH)
