"""Register two generic industry factors, build them, run the diagnostics and the single-factor group test.

The factors are deliberately textbook ones (12-1 month momentum, 3-month realised volatility): the point is the
pipeline — FactorSpec -> REGISTRY -> build/save -> rank IC and coverage -> long/short/group backtest -> saved reports —
not the signals. Requires `python scripts/make_synthetic_data.py` first.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy import base, config, factors, reports  # noqa: E402
from strategy.backtest import single_factor  # noqa: E402
from strategy.factors import FactorSpec, diagnostics  # noqa: E402

SECTION = "examples"
OUT_DIR = config.BACKTEST_DIR / "single"


def build_momentum() -> dict[str, pd.DataFrame]:
    """12-1 momentum: industry level at the previous month end over the level 12 months earlier."""
    level = base.industry_close().reindex(base.month_ends())
    return {"mom_12_1": level.shift(1) / level.shift(12) - 1}


def build_volatility() -> dict[str, pd.DataFrame]:
    """Annualised 63-day realised volatility of industry returns, sampled at month ends."""
    vol = base.industry_ret().rolling(63, min_periods=40).std() * np.sqrt(252)
    return {"vol_3m": vol.reindex(base.month_ends())}


factors.register(
    FactorSpec("mom_12_1", "market", "12-1 month industry momentum", +1, build_momentum, ["mom_12_1"]),
    FactorSpec("vol_3m", "market", "3-month realised volatility (low is good)", -1, build_volatility, ["vol_3m"]),
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in factors.names():
        factors.build(name)
    tables = {n: factors.load(n) for n in factors.names()}

    print("== factor diagnostics (monthly rank IC vs next-month industry excess return) ==")
    for n, df in tables.items():
        ic = diagnostics.rank_ic(df, factors.REGISTRY[n].sign)
        s = diagnostics.ic_summary(ic)
        cov = diagnostics.coverage(df).loc[config.BACKTEST_START:]
        print(f"  {n:10s} IC mean {s['ic_mean']:+.3f}  ICIR {s['icir']:+.2f}  t {s['t_stat']:+.1f}  "
              f"positive {s['pos_rate']:.0%}  coverage {int(cov.min())}~{int(cov.max())} industries")
        reports.save_json(SECTION, f"{n}_ic.json", s)
    corr = diagnostics.factor_corr(tables)
    print("  cross-sectional rank correlation between factors:")
    print(corr.round(3).to_string())
    reports.save_df(SECTION, "factor_corr.csv", corr)

    print("\n== single-factor test: long top 5 / short bottom 5 / 5 groups, monthly, equal weight ==")
    for n in factors.names():
        res = single_factor.run(n)
        res["daily"].to_parquet(OUT_DIR / f"{n}_daily.pq")
        reports.save_df(SECTION, f"{n}_yearly.csv", res["yearly"])
        reports.save_df(SECTION, f"{n}_groups.csv", res["groups"])
        reports.save_df(SECTION, f"{n}_holdings.csv", res["holdings"], index=False)
        reports.save_json(SECTION, f"{n}_metrics.json", res["metrics"])
        m, ls = res["metrics"]["long"], res["metrics"]["long_short"]
        print(f"  {n:10s} {m['start']} ~ {m['end']}  long: ann {m['ann_return']:+.1%} (bench {m['bench_ann_return']:+.1%}), "
              f"excess {m['ann_excess']:+.1%}, IR {m['info_ratio']:.2f}, excess max DD {m['excess_max_drawdown']:.1%}, "
              f"turnover {m['avg_turnover']:.0%}/month | long-short: ann {ls['ann_return']:+.1%}, Sharpe {ls['sharpe']:.2f}")
        print("    groups (1 = highest score):", " ".join(f"G{g}:{v:+.1%}" for g, v in res["groups"].ann_excess.items()))
    print(f"\nsaved: {OUT_DIR} and {config.REPORT_DIR / SECTION}")


if __name__ == "__main__":
    main()
