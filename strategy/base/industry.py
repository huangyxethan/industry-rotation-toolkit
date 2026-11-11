"""行业层：行业列表、个股行业归属面板、行业日收益（指数 + 成份股重建回补）、行业等权基准。

行业收益的来源优先级：
1. 申万一级指数日收益（sw_index_daily 收盘价，仅在前一交易日也有数据时计算，跨断档不计）；
2. 缺失处用成份股重建：ret_k,t = Σ_i w_i,t-1 r_i,t / Σ_i w_i,t-1，i 遍历当日归属行业 k 且有收益的股票。
   加权方式在 equal / mcap（总市值）/ fmcap（流通市值）中按校准选择：对有指数历史的区间比较
   重建收益与指数收益的相关系数与年化跟踪误差，取跟踪误差中位数最小者。
   个股收益为后复权全收益、申万指数为价格指数，重建收益相对指数有约 +3%/年的系统偏差（分红），
   回补时按校准的中位年化偏差逐日扣除（config.RECON_BIAS_ADJUST）。
   面板首日（无前一日）所有行业收益为 NaN，净值序列从 1000 起算。

行业等权基准（策略的比较基准）：31 个行业等权，月末调仓、月内权重随行业收益漂移。
另存日度再平衡版本（每日行业简单平均）供参考。
"""

import numpy as np
import pandas as pd

from etl.utils import get_logger

from .. import config
from .calendar import panel_days

logger = get_logger("strategy.industry")


def build_industry_list() -> pd.DataFrame:
    pm = pd.read_parquet(config.SW_PREFIX_MAP_PATH)[["l1_prefix", "l1_code", "l1_name"]]
    pm = pm.sort_values("l1_code").reset_index(drop=True)
    pm.insert(0, "idx", np.arange(len(pm), dtype="int8"))
    assert len(pm) == 31
    return pm


def build_membership_panel(industry_list: pd.DataFrame, stock_codes: list[str],
                           first_trade_days: pd.Series | None = None) -> pd.DataFrame:
    """交易日 × 股票 → 行业序号(int8)，-1 表示无归属。

    first_trade_days（股票 → 行情首日）给出时，把每只股票的首条分类记录向前回填到行情首日：
    覆盖新股上市至申万计入日之间的几天，以及换代码股票在新代码启用前挂在该代码下的历史行情。
    """
    days = panel_days()
    hist = pd.read_parquet(config.SW_STOCK_HIST_PATH)
    hist = hist[hist.is_a_share & hist.stock_code.isin(stock_codes)].copy()
    hist["start_date"] = hist.start_date.astype("datetime64[ns]")
    if first_trade_days is not None:
        first_rec = hist.start_date == hist.groupby("stock_code").start_date.transform("min")
        fd = hist.stock_code.map(first_trade_days).astype("datetime64[ns]")
        backfill = first_rec & fd.notna() & (fd < hist.start_date)
        hist.loc[backfill, "start_date"] = fd[backfill]
        logger.info("首条分类记录回填到行情首日: %d 只股票", int(backfill.sum()))
    code_to_idx = dict(zip(industry_list.l1_code, industry_list.idx))
    hist = hist.assign(idx=hist.l1_code.map(code_to_idx).astype("int8"))
    start = hist.start_date.astype("datetime64[ns]").values
    end = hist.end_date.astype("datetime64[ns]").fillna(pd.Timestamp("2262-01-01")).values
    lo = np.searchsorted(days.values, start, side="left")
    hi = np.searchsorted(days.values, end, side="left")
    n = np.maximum(hi - lo, 0)
    rows = np.repeat(np.arange(len(hist)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    long = pd.DataFrame({
        "date": days.values[lo[rows] + offs],
        "stock_code": hist.stock_code.values[rows],
        "idx": hist.idx.values[rows],
    })
    wide = long.pivot(index="date", columns="stock_code", values="idx")
    wide = wide.reindex(index=days, columns=stock_codes).fillna(-1).astype("int8")
    wide.columns.name = None
    logger.info("行业归属面板 %s（含上市前/退市后空白格）", wide.shape)
    return wide


def index_returns(industry_list: pd.DataFrame) -> pd.DataFrame:
    days = panel_days()
    d = pd.read_parquet(config.SW_INDEX_DAILY_PATH, columns=["industry_code", "日期", "收盘"])
    d["日期"] = d["日期"].astype("datetime64[ns]")
    close = d.pivot(index="日期", columns="industry_code", values="收盘")
    close = close.reindex(columns=industry_list.l1_code).reindex(days)
    prev = close.shift(1)
    ret = close / prev - 1
    ret = ret.where(close.notna() & prev.notna())
    ret.columns = industry_list.l1_name.values
    return ret


def reconstruct_returns(ret_panel: pd.DataFrame, member: pd.DataFrame, weight_panel: pd.DataFrame | None,
                        industry_list: pd.DataFrame) -> pd.DataFrame:
    """成份股加权重建行业日收益；weight_panel=None 为等权，否则用其前一日值加权。"""
    R = ret_panel.values.astype("float64")
    M = member.reindex(columns=ret_panel.columns).values
    valid = ~np.isnan(R) & (M >= 0)
    if weight_panel is None:
        W = np.ones_like(R)
    else:
        W = weight_panel.reindex(columns=ret_panel.columns).shift(1).values.astype("float64")
        W = np.where(np.isnan(W) | (W <= 0), 0.0, W)
    out = np.full((R.shape[0], len(industry_list)), np.nan)
    for k in industry_list.idx:
        mask = valid & (M == k) & (W > 0)
        wsum = (W * mask).sum(axis=1)
        num = (np.nan_to_num(R) * W * mask).sum(axis=1)
        out[:, k] = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)
    return pd.DataFrame(out, index=ret_panel.index, columns=industry_list.l1_name.values)


def calibrate_recon(idx_ret: pd.DataFrame, recon: dict[str, pd.DataFrame]) -> tuple[str, float, pd.DataFrame]:
    rows = []
    for name, rc in recon.items():
        for ind in idx_ret.columns:
            both = pd.concat([idx_ret[ind], rc[ind]], axis=1, keys=["idx", "rc"]).dropna()
            if len(both) < 250:
                continue
            diff = both.rc - both.idx
            rows.append({"weighting": name, "industry": ind, "n_days": len(both),
                         "corr": both.idx.corr(both.rc), "te_ann": diff.std() * np.sqrt(252),
                         "mean_diff_ann": diff.mean() * 252})
    calib = pd.DataFrame(rows)
    summary = calib.groupby("weighting").agg(corr_median=("corr", "median"), te_median=("te_ann", "median"),
                                             te_max=("te_ann", "max"), bias_median=("mean_diff_ann", "median"))
    logger.info("行业收益重建校准（重建 vs 申万指数）:\n%s", summary.to_string())
    best = summary.te_median.idxmin()
    bias = float(summary.loc[best, "bias_median"])
    logger.info("采用加权方式: %s, 回补时扣除年化偏差 %.4f", best, bias if config.RECON_BIAS_ADJUST else 0.0)
    return best, bias, calib


def build_industry_returns(ret_panel: pd.DataFrame, member: pd.DataFrame, panels: dict[str, pd.DataFrame],
                           industry_list: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx_ret = index_returns(industry_list)
    recon = {
        "equal": reconstruct_returns(ret_panel, member, None, industry_list),
        "mcap": reconstruct_returns(ret_panel, member, panels["mcap"], industry_list),
        "fmcap": reconstruct_returns(ret_panel, member, panels["fmcap"], industry_list),
    }
    best, bias, calib = calibrate_recon(idx_ret, recon)
    calib.to_csv(config.INDUSTRY_RECON_CALIB_PATH, index=False)
    recon_best = recon[best] - (bias / 252 if config.RECON_BIAS_ADJUST else 0.0)

    final = idx_ret.copy()
    source = pd.DataFrame(np.where(idx_ret.notna(), 1, 0), index=idx_ret.index, columns=idx_ret.columns)
    fill = idx_ret.isna() & recon_best.notna()
    final = final.where(~fill, recon_best)
    source = source.where(~fill, 2).astype("int8")
    n_missing = final.iloc[1:].isna().sum()  # 首日无前一日，预期为 NaN
    if n_missing.sum():
        logger.warning("首日之外仍有缺失的行业收益（日）: %s", n_missing[n_missing > 0].to_dict())
    summary = pd.DataFrame({
        "index_first_day": idx_ret.apply(lambda s: s.first_valid_index()),
        "days_from_index": (source == 1).sum(),
        "days_reconstructed": (source == 2).sum(),
    })
    logger.info("行业收益来源汇总:\n%s", summary[summary.days_reconstructed > 0].to_string())
    close = 1000 * (1 + final.fillna(0)).cumprod()
    return final, close, source


def build_benchmark(ind_ret: pd.DataFrame) -> pd.DataFrame:
    r = ind_ret.fillna(0.0)
    month = r.index.to_period("M")
    growth = (1 + r).groupby(month).cumprod()
    port_val = growth.mean(axis=1)
    prev_val = port_val.groupby(month).shift(1).fillna(1.0)
    bench = pd.DataFrame({"ind_ew_monthly": port_val / prev_val - 1, "ind_ew_daily": r.mean(axis=1)}, index=r.index)

    idx = pd.read_parquet(config.INDEX_DAILY_PATH, columns=["index_code", "date", "close"])
    idx["date"] = idx["date"].astype("datetime64[ns]")
    for code, name in (("000906", "csi800"), ("000985", "csi_all"), ("801001", "sw_a"), ("000300", "csi300")):
        s = idx[idx.index_code == code].set_index("date").close.sort_index()
        s = s[~s.index.duplicated()].reindex(r.index)
        bench[name] = s.pct_change().where(s.notna() & s.shift(1).notna())
    return bench
