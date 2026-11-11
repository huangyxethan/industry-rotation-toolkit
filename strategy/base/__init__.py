"""基础层：对齐面板与查询接口，落地 .data/base/。

读取：from strategy import base; base.industry_ret() / base.stock_panel("ret") / base.month_ends(...) 等，
      全部接口定义在 loaders.py。
构建函数：calendar.build_calendar_table / industry.build_* / pit.build_pit_financials，
      其输入格式见 docs/data_formats.md；scripts/make_synthetic_data.py 演示了完整的构建顺序。
"""

from .loaders import *  # noqa: F401,F403
from .loaders import (analyst_avail_date, benchmark, calendar_table, financial_avail_dates,  # noqa: F401
                      industry_close, industry_list, industry_names, industry_of, industry_ret,
                      load_holdings_avail, load_pit_financials, month_ends, monthly_returns,
                      next_trade_day, on_or_after, on_or_before, period_returns, prev_trade_day, stock_industry,
                      stock_panel, trade_days, week_ends, weekly_returns)
