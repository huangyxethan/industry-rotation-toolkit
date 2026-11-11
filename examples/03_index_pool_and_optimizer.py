"""Aggregate a stock-level statistic two ways (industry partition vs overlapping index memberships) and solve
style-constrained weights for a basket of indices.

Steps:
  1. 12-1 month momentum per stock from the daily return panel, sampled at month ends;
  2. industry aggregation with factors.common.industry_mean (one label per stock, bincount),
     index aggregation with index_pool.aggregate.member_mean / member_mad_mean (many-to-many membership matrix);
  3. a quick rank-IC check of the index-level signal against next-month index excess return over the broad benchmark;
  4. pick the top 8 indices at the last month end, aggregate synthetic stock-level style exposures to those indices,
     and solve etf_rotation.optimize.solve for weights whose exposures stay close to the market's,
     with a per-index cap and tolerance bands; compare with equal weight and with the penalty formulation.
Requires `python scripts/make_synthetic_data.py` first.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy import base, config  # noqa: E402
from strategy.etf_rotation import optimize  # noqa: E402
from strategy.factors.common import industry_mean  # noqa: E402
from strategy.index_pool import aggregate  # noqa: E402
from strategy.index_pool import base as ipb  # noqa: E402


def stock_momentum() -> pd.DataFrame:
    monthly = base.monthly_returns(base.stock_panel("ret"))
    mom = np.expm1(np.log1p(monthly).rolling(11, min_periods=11).sum().shift(1))
    return mom.loc[config.BACKTEST_START:config.BACKTEST_END]


def main() -> None:
    stat = stock_momentum()
    print(f"stock-level 12-1 momentum: {stat.shape[0]} month ends x {stat.shape[1]} stocks")

    by_industry = industry_mean(stat)
    by_index = aggregate.member_mean(stat)
    by_index_mad = aggregate.member_mad_mean(stat, k=3.0, min_members=config.INDEX_POOL_MIN_MEMBERS)
    print(f"industry aggregation: {by_industry.shape}, index aggregation: {by_index.shape}")
    t = stat.index[-1]
    print(f"\nlatest month end {t.date()} — top industries by member momentum:")
    print(by_industry.loc[t].nlargest(5).map("{:+.1%}".format).to_string())
    print("\nindex-level momentum, plain vs MAD-trimmed mean:")
    print(pd.DataFrame({"mean": by_index.loc[t], "mad_mean": by_index_mad.loc[t]}).map("{:+.1%}".format).to_string())

    # rank IC of the index signal against next-month index excess return over the broad benchmark
    idx_m = base.monthly_returns(ipb.index_ret()).reindex(stat.index)
    bench_m = base.monthly_returns(base.benchmark()[[config.INDEX_POOL_BENCHMARK]]).reindex(stat.index)[config.INDEX_POOL_BENCHMARK]
    fwd = idx_m.sub(bench_m, axis=0).shift(-1)
    ics = []
    for d in stat.index:
        a, b = by_index.loc[d], fwd.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 8:
            ics.append(a[ok].rank().corr(b[ok].rank()))
    ics = pd.Series(ics)
    print(f"\nindex momentum rank IC vs next-month excess over {config.INDEX_POOL_BENCHMARK}: "
          f"mean {ics.mean():+.3f}, ICIR {ics.mean() / ics.std():+.2f}, n {len(ics)}")

    # style-constrained weights on the top-8 basket at the last month end
    rng = np.random.default_rng(7)
    styles = list(config.STYLE_FACTORS)
    stocks = stat.columns
    E = pd.DataFrame(rng.normal(size=(len(stocks), len(styles))), index=stocks, columns=styles)
    E["SZ"] = np.log(base.stock_panel("mcap").loc[t].reindex(stocks)).pipe(lambda s: (s - s.mean()) / s.std())
    E["BETA"] = 1.0 + 0.3 * E["BETA"]
    listed = stat.loc[t].notna()
    F_all = pd.concat({s: aggregate.member_mean(pd.DataFrame([E[s].where(listed)], index=[t])).loc[t] for s in styles}, axis=1)
    top = by_index.loc[t].nlargest(8).index
    F = F_all.loc[top].values
    f_bench = E[listed].mean().values            # market-wide exposure as the benchmark
    sigma = E[listed].std().values

    w_eq = np.full(len(top), 1 / len(top))
    w_opt, status = optimize.solve(F, f_bench, sigma, styles)
    dev = pd.DataFrame({"equal_weight": (w_eq @ F - f_bench) / sigma, "optimised": (w_opt @ F - f_bench) / sigma}, index=styles)
    print(f"\nSLSQP status: {status}; weight cap {config.ETF_W_MAX}, beta tol {config.ETF_BETA_TOL}, style tol {config.ETF_STYLE_TOL}")
    print(pd.DataFrame({"index": top, "name": ipb.index_list().set_index('index_code').loc[top, 'index_name'].values,
                        "w_equal": w_eq, "w_opt": w_opt.round(3)}).to_string(index=False))
    print("\nstandardised exposure deviation from the market (sum of squares: "
          f"{(dev.equal_weight ** 2).sum():.2f} -> {(dev.optimised ** 2).sum():.2f}):")
    print(dev.round(2).T.to_string())

    print("\npenalty formulation, lambda sweep (distance to equal weight vs exposure fit):")
    for lam in (0.0, 1.0, 10.0, 100.0):
        w, st = optimize.solve_penalty(F, f_bench, sigma, styles, w_max=config.ETF_W_MAX, beta_tol=config.ETF_BETA_TOL, lam=lam)
        d = (w @ F - f_bench) / sigma
        print(f"  lambda {lam:6.1f}  status {st:8s}  |w - w_eq| {np.abs(w - w_eq).sum():.3f}  "
              f"exposure SS {(d @ d):.2f}  max weight {w.max():.3f}")


if __name__ == "__main__":
    main()
