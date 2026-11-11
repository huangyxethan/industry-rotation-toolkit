"""结果展示层：读取 .data/reports/ 与 .data/backtest/ 的运行产出，生成 chartbook（PNG 图集 + 自包含 HTML + PDF）。

章节按"结论 → 证据 → 可信度"组织：每章一个 Chapter，回答一个业务问题，由图、表、指标卡与文字区块组成；
所有图表与结论句均由运行结果生成。独立于 strategy 包（只消费其产出，不参与计算）。
用法见 examples/02_chartbook.py。
"""
