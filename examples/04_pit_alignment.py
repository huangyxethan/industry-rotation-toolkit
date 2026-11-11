"""Point-in-time alignment: when does a quarterly report become usable for a signal?

Shows the base layer's convention — an "available date" is the first trading day on which a datum could be used
at the close — on the synthetic balance sheet: announcement-date lag distribution, the fallback to the statutory
deadline when the announcement date is missing or anomalous, and look-ups for arbitrary (stock, report period) pairs.
Requires `python scripts/make_synthetic_data.py` first.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy import base  # noqa: E402
from strategy.base import pit  # noqa: E402
from strategy.factors.common import asof_sample  # noqa: E402


def main() -> None:
    pf = pit.load_pit_financials()
    print(f"PIT table: {len(pf):,} (stock, report period) rows, {pf.stock_code.nunique()} stocks")
    print("available-date source:", pf.avail_source.value_counts().to_dict())
    lag = (pf.avail_date - pf.report_date).dt.days
    print("days from report period end to availability, by quarter:")
    print(lag.groupby(pf.report_date.dt.quarter).describe()[["min", "25%", "50%", "75%", "max"]].astype(int).to_string())

    sample = pf.sample(3, random_state=1)
    q = pd.DataFrame({"stock_code": list(sample.stock_code) + ["999999"],
                      "report_date": list(sample.report_date) + [pd.Timestamp("2021-06-30")]})
    q["avail_date"] = pit.financial_avail_dates(q.stock_code, q.report_date).values
    q["weekday"] = pd.to_datetime(q.avail_date).dt.day_name()
    print("\nlook-ups (the last row has no record and falls back to the statutory deadline's next trading day):")
    print(q.to_string(index=False))

    # a report-day event becomes usable at the next close; sampling 'as of' month ends carries the last known value
    me = base.month_ends("2021-01-01", "2021-12-31")
    events = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(["2021-04-29", "2021-08-30", "2021-10-29"]))
    events.index = pd.DatetimeIndex(base.next_trade_day(events.index).values)
    print("\nevent values available at the next trading day, sampled as of each month end:")
    print(asof_sample(events, me).value.to_string())


if __name__ == "__main__":
    main()
