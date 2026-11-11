"""Synthetic inputs in the exact formats the base layer reads, so every example runs without market data.

Writes, under `.data/` (git-ignored):

  raw/trade_calendar.pq, raw/index_daily.pq, raw/stock_balance_sheet.pq      raw tables consumed by the builders
  base/calendar.pq                 built with strategy.base.calendar.build_calendar_table
  base/industry_list.pq            31 industries (Shenwan level-1 names are used as labels only)
  base/stock_industry.pq           trade day x stock -> industry index (int8, -1 = not listed)
  base/stock_ret.pq, stock_mcap.pq daily return and market cap panels (float32)
  base/industry_ret_daily.pq       equal-weight industry returns, industry_close_daily.pq the 1000-based level
  base/benchmark_daily.pq          built with strategy.base.industry.build_benchmark
  base/pit_financials.pq           built with strategy.base.pit.build_pit_financials
  base/index_pool/*.pq             12 overlapping synthetic indices with semi-annual membership snapshots

The return model is a one-factor market model plus an industry component whose drift follows a slow AR(1),
so a momentum-type factor has something to find. Nothing here resembles any market; the data exist to
exercise the code. Runs in a few seconds; `--seed` changes the draw.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy import config  # noqa: E402
from strategy.base import calendar as cal  # noqa: E402
from strategy.base import industry, pit  # noqa: E402
from strategy.index_pool import base as ipb  # noqa: E402

N_STOCKS = 600
N_THEME_INDICES = 9

# Shenwan 2021 level-1 classification: public labels, used here only to give the 31 columns realistic names.
SW_L1 = [
    ("801010", "农林牧渔"), ("801030", "基础化工"), ("801040", "钢铁"), ("801050", "有色金属"), ("801080", "电子"),
    ("801110", "家用电器"), ("801120", "食品饮料"), ("801130", "纺织服饰"), ("801140", "轻工制造"), ("801150", "医药生物"),
    ("801160", "公用事业"), ("801170", "交通运输"), ("801180", "房地产"), ("801200", "商贸零售"), ("801210", "社会服务"),
    ("801230", "综合"), ("801710", "建筑材料"), ("801720", "建筑装饰"), ("801730", "电力设备"), ("801740", "国防军工"),
    ("801750", "计算机"), ("801760", "传媒"), ("801770", "通信"), ("801780", "银行"), ("801790", "非银金融"),
    ("801880", "汽车"), ("801890", "机械设备"), ("801950", "煤炭"), ("801960", "石油石化"), ("801970", "环保"),
    ("801980", "美容护理"),
]


def trading_calendar() -> pd.DatetimeIndex:
    """Weekdays from 2009-12 to 2025-12 minus a few fixed holiday blocks, so the calendar has gaps like a real one."""
    days = pd.bdate_range("2009-12-01", "2025-12-31")
    drop = []
    for y in range(2009, 2026):
        drop += list(pd.date_range(f"{y}-01-01", f"{y}-01-01"))
        drop += list(pd.date_range(f"{y}-05-01", f"{y}-05-03"))
        drop += list(pd.date_range(f"{y}-10-01", f"{y}-10-07"))
    return days.difference(pd.DatetimeIndex(drop))


def assign_industries(rng: np.random.Generator, n: int, k: int, min_size: int = 8) -> np.ndarray:
    probs = rng.dirichlet(np.full(k, 3.0))
    ind = rng.choice(k, n, p=probs)
    counts = np.bincount(ind, minlength=k)
    while counts.min() < min_size:
        small, big = counts.argmin(), counts.argmax()
        movers = np.flatnonzero(ind == big)[: min_size - counts[small]]
        ind[movers] = small
        counts = np.bincount(ind, minlength=k)
    return ind


def listing_windows(rng: np.random.Generator, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(first, last) row index of each stock's listed period."""
    first = np.zeros(n, dtype=int)
    late = rng.random(n) < 0.15
    first[late] = rng.integers(1, int(T * 0.7), late.sum())
    last = np.full(n, T - 1)
    gone = rng.random(n) < 0.03
    last[gone] = np.minimum(first[gone] + rng.integers(500, T, gone.sum()), T - 1)
    return first, last


def simulate_returns(rng: np.random.Generator, T: int, ind_of: np.ndarray, first: np.ndarray, last: np.ndarray,
                     k: int) -> np.ndarray:
    n = len(ind_of)
    market = rng.normal(0.0002, 0.011, T)
    drift = np.zeros((T, k))
    eps = rng.normal(0.0, 7e-5, (T, k))
    for t in range(1, T):
        drift[t] = 0.99 * drift[t - 1] + eps[t]
    ind_component = drift + rng.normal(0.0, 0.006, (T, k))
    beta = np.clip(rng.normal(1.0, 0.25, n), 0.3, 2.0)
    R = market[:, None] * beta[None, :] + ind_component[:, ind_of] + rng.normal(0.0, 0.018, (T, n))
    rows = np.arange(T)[:, None]
    listed = (rows >= first[None, :]) & (rows <= last[None, :])
    R[~listed] = np.nan
    R[first, np.arange(n)] = 0.0     # no return on the listing day
    return R


def write_index_pool(rng: np.random.Generator, days: pd.DatetimeIndex, codes: list[str], R: np.ndarray,
                     mcap: np.ndarray, ind_of: np.ndarray, k: int) -> None:
    ipb.D.mkdir(parents=True, exist_ok=True)
    me = pd.DatetimeIndex(pd.Series(days).groupby(days.to_period("M")).last().values)
    snaps = me[(me.month.isin([6, 12])) & (me >= pd.Timestamp("2013-12-01")) & (me <= pd.Timestamp("2025-06-30"))]
    index_codes = [f"9000{i + 1:02d}" for i in range(3 + N_THEME_INDICES)]
    names = ["SYN Broad 300", "SYN Broad 500", "SYN Small 300"] + [f"SYN Theme {i + 1:02d}" for i in range(N_THEME_INDICES)]
    is_broad = [True, True, True] + [False] * N_THEME_INDICES
    themes = [rng.choice(k, size=rng.integers(2, 4), replace=False) for _ in range(N_THEME_INDICES)]
    rows = []
    for s in snaps:
        i = days.get_loc(s)
        listed = ~np.isnan(R[i])
        size = np.where(listed, mcap[i], -np.inf)
        order = np.argsort(-size)
        broad = {0: order[:300], 1: order[300:800], 2: order[300:600]}
        for j, code in enumerate(index_codes):
            if j < 3:
                members = broad[j]
                members = members[listed[members]]
            else:
                pool = np.flatnonzero(listed & np.isin(ind_of, themes[j - 3]))
                members = pool[rng.random(len(pool)) < 0.75]
            w = mcap[i, members]
            rows.append(pd.DataFrame({"snapshot_date": s, "index_code": code, "stock_code": np.array(codes)[members],
                                      "weight": w / w.sum()}))
    membership = pd.concat(rows, ignore_index=True).sort_values(["snapshot_date", "index_code", "stock_code"]).reset_index(drop=True)
    membership["snapshot_date"] = membership.snapshot_date.astype("datetime64[ns]")
    membership.to_parquet(ipb.MEMBERSHIP_PATH, index=False)

    index_list = pd.DataFrame({
        "index_code": index_codes, "index_name": names, "is_broad": is_broad,
        "n_etf": rng.integers(1, 6, len(index_codes)), "in_pool": True, "exclude_reason": pd.Series([None] * len(index_codes), dtype="object"),
        "history_start": snaps[0], "cons_start": snaps[0], "cons_end": snaps[-1],
    })
    index_list.to_parquet(ipb.INDEX_LIST_PATH, index=False)

    # equal-weight member returns between snapshots, NaN before the first snapshot
    ret = np.full((len(days), len(index_codes)), np.nan)
    bounds = list(snaps) + [days[-1] + pd.Timedelta(days=1)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = (days > a) & (days < b) if b != bounds[-1] else (days > a)
        rows_idx = np.flatnonzero(sel)
        m = membership[membership.snapshot_date == a]
        for j, code in enumerate(index_codes):
            cols = pd.Index(codes).get_indexer(m[m.index_code == code].stock_code)
            block = R[rows_idx][:, cols]
            ret[rows_idx, j] = np.nanmean(block, axis=1)
    index_ret = pd.DataFrame(ret, index=days, columns=index_codes)
    index_ret.index.name = "date"
    index_ret.to_parquet(ipb.INDEX_RET_PATH)
    close = 1000 * (1 + index_ret.fillna(0)).cumprod()
    close[index_ret.isna().cummin()] = np.nan
    close.to_parquet(ipb.INDEX_CLOSE_PATH)
    print(f"  index_pool: {len(index_codes)} indices, {len(snaps)} snapshots, {len(membership):,} membership rows")


def write_balance_sheet(rng: np.random.Generator, codes: list[str], first: np.ndarray, last: np.ndarray,
                        days: pd.DatetimeIndex) -> None:
    """Quarterly report dates with announcement dates inside the statutory window, a few missing and a few anomalous."""
    qends = pd.date_range("2010-03-31", "2025-09-30", freq="QE")
    lag_lo = {1: 10, 2: 25, 3: 10, 4: 50}
    lag_hi = {1: 30, 2: 62, 3: 31, 4: 120}
    rows = []
    for i, code in enumerate(codes):
        rd = qends[(qends >= days[first[i]]) & (qends <= days[last[i]])]
        q = rd.quarter.values
        lag = rng.integers([lag_lo[x] for x in q], [lag_hi[x] + 1 for x in q])
        notice = rd + pd.to_timedelta(lag, unit="D")
        notice = pd.Series(notice)
        notice[rng.random(len(rd)) < 0.02] = pd.NaT                               # missing announcement date
        late = rng.random(len(rd)) < 0.01
        notice[late] = notice[late] + pd.Timedelta(days=200)                     # looks like a restatement
        rows.append(pd.DataFrame({"stock_code": code, "report_date": rd, "notice_date": notice.values,
                                  "share_capital": float(rng.lognormal(20.5, 1.0))}))
    bs = pd.concat(rows, ignore_index=True)
    bs.to_parquet(config.BALANCE_SHEET_PATH, index=False)
    print(f"  balance sheet: {len(bs):,} report rows for {len(codes)} stocks")


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)

    # calendar: raw table -> calendar.pq through the real builder
    pd.DataFrame({"trade_date": trading_calendar()}).to_parquet(config.TRADE_CAL_PATH, index=False)
    cal.trade_days.cache_clear()
    caltab = cal.build_calendar_table()
    caltab.to_parquet(config.CALENDAR_PATH, index=False)
    days = pd.DatetimeIndex(caltab.trade_date)
    T, k = len(days), len(SW_L1)
    print(f"  calendar: {T} trading days {days[0].date()} ~ {days[-1].date()}, {int(caltab.is_month_end.sum())} month ends")

    il = pd.DataFrame({"idx": np.arange(k, dtype="int8"), "l1_code": [c for c, _ in SW_L1], "l1_name": [n for _, n in SW_L1]})
    il.to_parquet(config.INDUSTRY_LIST_PATH, index=False)

    codes = [f"{600000 + i:06d}" if i % 2 else f"{i + 1:06d}" for i in range(N_STOCKS)]
    ind_of = assign_industries(rng, N_STOCKS, k)
    first, last = listing_windows(rng, N_STOCKS, T)
    R = simulate_returns(rng, T, ind_of, first, last, k)

    member = np.where(np.isnan(R), -1, ind_of[None, :]).astype("int8")
    pd.DataFrame(member, index=days, columns=codes).rename_axis("date").to_parquet(config.STOCK_INDUSTRY_PATH)
    ret_panel = pd.DataFrame(R.astype("float32"), index=days, columns=codes).rename_axis("date")
    ret_panel.to_parquet(config.STOCK_PANEL_PATHS["ret"])
    size = rng.lognormal(np.log(1.5e10), 1.1, N_STOCKS)
    mcap = size[None, :] * np.cumprod(1 + np.nan_to_num(R), axis=0)
    mcap[np.isnan(R)] = np.nan
    pd.DataFrame(mcap.astype("float32"), index=days, columns=codes).rename_axis("date").to_parquet(config.STOCK_PANEL_PATHS["mcap"])
    print(f"  stock panels: {N_STOCKS} stocks, industries sized {np.bincount(ind_of).min()}~{np.bincount(ind_of).max()}")

    # industry returns = equal-weight mean of listed members; first panel day NaN by convention
    onehot = np.eye(k)[ind_of]
    valid = ~np.isnan(R)
    num = np.nan_to_num(R) @ onehot
    cnt = valid.astype(float) @ onehot
    ind_ret = pd.DataFrame(np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan), index=days, columns=il.l1_name.values).rename_axis("date")
    ind_ret.iloc[0] = np.nan
    ind_ret.to_parquet(config.INDUSTRY_RET_PATH)
    (1000 * (1 + ind_ret.fillna(0)).cumprod()).to_parquet(config.INDUSTRY_CLOSE_PATH)
    pd.DataFrame(1, index=days, columns=il.l1_name.values, dtype="int8").rename_axis("date").to_parquet(config.INDUSTRY_RET_SOURCE_PATH)

    # market indices as raw closes, then the real benchmark builder
    order = np.argsort(-size)
    subsets = {"000906": order[:int(0.6 * N_STOCKS)], "000985": order, "801001": order, "000300": order[:int(0.3 * N_STOCKS)]}
    parts = []
    for code, sel in subsets.items():
        r = np.nanmean(R[:, sel], axis=1)
        r[0] = 0.0
        close = 1000 * np.cumprod(1 + r)
        parts.append(pd.DataFrame({"index_code": code, "date": days, "close": close}))
    pd.concat(parts, ignore_index=True).to_parquet(config.INDEX_DAILY_PATH, index=False)
    bench = industry.build_benchmark(ind_ret)
    bench.rename_axis("date").to_parquet(config.BENCHMARK_PATH)
    print(f"  benchmark columns: {list(bench.columns)}")

    write_balance_sheet(rng, codes, first, last, days)
    pit.build_pit_financials().to_parquet(config.PIT_FINANCIALS_PATH, index=False)

    write_index_pool(rng, days, codes, R, mcap, ind_of, k)
    print(f"done -> {config.DATA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20260101)
    main(ap.parse_args().seed)
