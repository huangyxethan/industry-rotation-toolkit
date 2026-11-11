"""Render a one-chapter chartbook (self-contained HTML + PDF) from the reports saved by example 01.

The chartbook layer never recomputes anything: it reads the saved daily series and report tables, draws the
figures, and writes `output/example_chartbook.html` and `.pdf`. Chart labels are in Chinese, the working
language of the project.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chartbook.common import (CHARTBOOK_DIR, Chapter, close_all, drawdown_events, fig_month_heatmap,  # noqa: E402
                              fig_nav_with_excess, fig_yearly_bars, num, pct, setup_style)
from chartbook.render import render_html, render_pdf  # noqa: E402
from strategy import base, config, reports  # noqa: E402

NAME = "mom_12_1"
SECTION = "examples"


def main() -> None:
    setup_style()
    daily = pd.read_parquet(config.BACKTEST_DIR / "single" / f"{NAME}_daily.pq")
    yearly = pd.read_csv(reports.path(SECTION, f"{NAME}_yearly.csv"), index_col=0)
    groups = pd.read_csv(reports.path(SECTION, f"{NAME}_groups.csv"), index_col=0)
    metrics = reports.load_json(SECTION, f"{NAME}_metrics.json")
    m = metrics["long"]

    ch = Chapter("ex1", "动量因子多头组合（合成数据）", "在合成的行业收益上，12-1 月动量多头相对行业等权基准表现如何？")
    ch.add_cards([
        ("年化收益", pct(m["ann_return"]), f"基准 {pct(m['bench_ann_return'])}"),
        ("年化超额", pct(m["ann_excess"]), "净值比口径"),
        ("信息比率", num(m["info_ratio"]), f"月度胜率 {pct(m['monthly_win_rate'], 0)}"),
        ("超额最大回撤", pct(m["excess_max_drawdown"]), f"组合最大回撤 {pct(m['max_drawdown'])}"),
    ], "关键指标")
    ch.add_fig(fig_nav_with_excess(daily, "long", "bench", ("动量多头", "行业等权基准"), "净值与超额净值"),
               "nav", "净值与超额净值", caption="合成数据，仅用于演示代码；曲线不代表任何市场结果。")
    ch.add_fig(fig_yearly_bars(yearly, ("动量多头", "行业等权基准")), "yearly", "分年度收益")
    monthly = base.monthly_returns(daily[["long", "bench"]])
    ch.add_fig(fig_month_heatmap(monthly.long - monthly.bench), "heatmap", "月度超额热力图")
    ch.add_table(groups, "分组测试（组 1 得分最高）",
                 fmt={"ann_return": "pct", "ann_excess": "pct", "info_ratio": "num"},
                 caption="各组等权持有，月末调仓；ann_excess 为相对行业等权基准的年化超额（净值比口径）。")
    ex_nav = daily.excess_nav
    dd = drawdown_events(ex_nav, top=3)
    if not dd.empty:
        dd["peak"] = dd.peak.dt.date
        dd["trough"] = dd.trough.dt.date
        dd["recover"] = dd.recover.dt.date
        ch.add_table(dd, "超额净值最大回撤事件", fmt={"depth": "pct", "to_trough_days": "int", "to_recover_days": "int"}, index=False)
    ch.conclusions = [
        f"样本 {m['start']} ~ {m['end']}：年化超额 {pct(m['ann_excess'])}，信息比率 {num(m['info_ratio'])}，超额最大回撤 {pct(m['excess_max_drawdown'])}。",
        "数据为合成的随机样本，结论不具任何市场含义。",
    ]

    CHARTBOOK_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html([ch], "示例图集", "合成数据上的单因子测试结果", CHARTBOOK_DIR / "example_chartbook.html")
    pdf = render_pdf([ch], "示例图集", "合成数据上的单因子测试结果", CHARTBOOK_DIR / "example_chartbook.pdf")
    close_all([ch])
    print(f"written: {html}\n         {pdf}\n         figures under {CHARTBOOK_DIR / 'figs'}")


if __name__ == "__main__":
    main()
