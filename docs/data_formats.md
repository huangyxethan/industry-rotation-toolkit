# Data formats

The toolkit reads everything through `strategy.base` (and `strategy.index_pool.base` for the
index pool). The files below are what those readers expect. `scripts/make_synthetic_data.py`
writes every one of them, so it doubles as an executable specification; a real data pipeline
only has to produce the same files under `.data/`.

Conventions shared by all wide tables: the index is the trading day as `datetime64[ns]`,
stock columns are six-character code strings, industry columns are the names in
`industry_list.pq` in that order, values are `float32` for stock panels and `float64`
elsewhere. "Available date" always means a trading day on which the datum can be used at the
close.

## Raw inputs (`.data/raw/`)

Consumed by the builder functions in `strategy/base/*.py`.

| File | Columns | Used by |
|---|---|---|
| `trade_calendar.pq` | `trade_date` — every exchange trading day, may extend beyond the panel window | `calendar.trade_days`, `build_calendar_table` |
| `index_daily.pq` | `index_code`, `date`, `close` — daily closes of the market indices mapped to benchmark columns (`000906`→`csi800`, `000985`→`csi_all`, `801001`→`sw_a`, `000300`→`csi300`) | `industry.build_benchmark` |
| `stock_balance_sheet.pq` | `stock_code`, `report_date`, `notice_date` (first announcement date, may be missing), `share_capital` | `pit.build_pit_financials` |
| `sw_l1_prefix_map.pq` | `l1_prefix`, `l1_code`, `l1_name` — the 31 level-1 industries | `industry.build_industry_list` |
| `sw_stock_industry_hist.pq` | `stock_code`, `l1_code`, `start_date`, `end_date` (NaT = open), `is_a_share` | `industry.build_membership_panel` |
| `sw_index_daily.pq` | `industry_code`, `日期`, `收盘` — industry index closes where available | `industry.index_returns` |
| `disclosure_deadlines.pq` | `report_date`, `top10_avail`, `full_avail` — availability dates of fund holdings | `pit.load_holdings_avail` |

## Base layer (`.data/base/`)

| File | Shape / columns | Reader |
|---|---|---|
| `calendar.pq` | `trade_date`, `year`, `month`, `is_month_end`, `is_week_end`, `is_quarter_end`, `month_id`, `week_id` | `base.calendar_table`, `month_ends`, `week_ends` |
| `industry_list.pq` | `idx` (int8 position), `l1_code`, `l1_name` | `base.industry_list`, `industry_names` |
| `stock_industry.pq` | trading day × stock → `int8` industry position, `-1` = not listed / no classification | `base.stock_industry`, `industry_of` |
| `stock_ret.pq`, `stock_turnover.pq`, `stock_amount.pq`, `stock_mcap.pq`, `stock_fmcap.pq`, `stock_traded.pq` | trading day × stock, `float32` (`traded` is bool); NaN outside the listing window | `base.stock_panel(field)` |
| `industry_ret_daily.pq` | trading day × industry daily return; the first panel day is NaN | `base.industry_ret` |
| `industry_close_daily.pq` | trading day × industry level, 1000 on the first day | `base.industry_close` |
| `industry_ret_source.pq` | trading day × industry `int8`: 1 = index, 2 = reconstructed from members | diagnostics only |
| `benchmark_daily.pq` | `ind_ew_monthly` (equal-weight industries, monthly rebalanced), `ind_ew_daily`, `csi800`, `csi_all`, `sw_a`, `csi300` | `base.benchmark` |
| `pit_financials.pq` | `stock_code`, `report_date`, `notice_date`, `deadline`, `avail_date`, `avail_source` (`notice` / `deadline`) | `pit.load_pit_financials`, `financial_avail_dates` |

## Factor tables (`.data/factors/{name}.pq`)

Index `date` = month-end trading days (`datetime64[ns]`), one column per industry, `float64`
raw factor values (no cross-sectional standardisation). The value at `t` uses only data
available by the close of `t`. Written by `strategy.factors.save`, read by `load`.

## Index pool (`.data/base/index_pool/`)

| File | Columns |
|---|---|
| `index_list.pq` | `index_code`, `index_name`, `is_broad`, `n_etf`, `in_pool`, `exclude_reason`, `history_start`, `cons_start`, `cons_end` |
| `membership_long.pq` | `snapshot_date`, `index_code`, `stock_code`, `weight` — one row per member per snapshot; a snapshot is in force until the next one |
| `index_ret_daily.pq` | trading day × `index_code` daily return, NaN before the index history starts |
| `index_close_daily.pq` | trading day × `index_code` close |

## Run outputs

- `.data/backtest/` — daily series of each backtest (parquet).
- `.data/reports/<section>/` — CSV tables and JSON summaries written through `strategy.reports`;
  the chartbook reads only these.
- `.data/logs/` — one log per run when `etl.utils.setup_run_logging` is used.
- `output/` — chartbook HTML, PDF and PNG figures.
