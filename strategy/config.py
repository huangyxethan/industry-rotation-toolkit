"""策略层全局配置：路径、样本区间与通用回测设置。

这里只放基础设施层面的设置。具体因子的定义与参数属于各因子模块，由其通过
`strategy.factors.register` 登记（见 examples/）。
"""

import pandas as pd

from etl import config as etl_config

DATA_DIR = etl_config.DATA_DIR                    # .data/ 项目生成数据
EXTERNAL_DATA_DIR = etl_config.EXTERNAL_DATA_DIR  # data/  外部提供数据
LOG_DIR = etl_config.LOG_DIR
REPORT_DIR = etl_config.REPORT_DIR
OUTPUT_DIR = etl_config.PROJECT_ROOT / "output"   # 结果汇总文件（chartbook），项目根目录直接可见
RAW_DIR = DATA_DIR / "raw"                        # 基础层构建函数读取的原始表
BASE_DIR = DATA_DIR / "base"

# 面板区间
PANEL_START = pd.Timestamp("2010-01-04")
PANEL_END = pd.Timestamp("2025-12-31")

# 回测区间（因子/回测层默认值）
BACKTEST_START = pd.Timestamp("2016-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")

# ---------- 原始表路径（strategy/base/*.py 的构建函数读取；格式见 docs/data_formats.md）----------
TRADE_CAL_PATH = RAW_DIR / "trade_calendar.pq"              # trade_date
DISCLOSURE_PATH = RAW_DIR / "disclosure_deadlines.pq"       # report_date, top10_avail, full_avail
BALANCE_SHEET_PATH = RAW_DIR / "stock_balance_sheet.pq"     # stock_code, report_date, notice_date, share_capital
SW_INDEX_DAILY_PATH = RAW_DIR / "sw_index_daily.pq"         # industry_code, 日期, 收盘
SW_PREFIX_MAP_PATH = RAW_DIR / "sw_l1_prefix_map.pq"        # l1_prefix, l1_code, l1_name
SW_STOCK_HIST_PATH = RAW_DIR / "sw_stock_industry_hist.pq"  # stock_code, l1_code, start_date, end_date, is_a_share
INDEX_DAILY_PATH = RAW_DIR / "index_daily.pq"               # index_code, date, close

# ---------- 基础层产出 ----------
CALENDAR_PATH = BASE_DIR / "calendar.pq"
INDUSTRY_LIST_PATH = BASE_DIR / "industry_list.pq"
STOCK_INDUSTRY_PATH = BASE_DIR / "stock_industry.pq"        # 宽表 int8：行业序号，-1 = 无归属
STOCK_PANEL_PATHS = {                                       # 宽表 float32
    "ret": BASE_DIR / "stock_ret.pq",                       # 日收益（后复权，含分红）
    "turnover": BASE_DIR / "stock_turnover.pq",             # 换手率（小数）
    "amount": BASE_DIR / "stock_amount.pq",                 # 成交额（元）
    "mcap": BASE_DIR / "stock_mcap.pq",                     # 总市值（元）
    "fmcap": BASE_DIR / "stock_fmcap.pq",                   # 流通市值（元）
    "traded": BASE_DIR / "stock_traded.pq",                 # bool：当日是否有成交
}
INDUSTRY_RET_PATH = BASE_DIR / "industry_ret_daily.pq"      # 行业 × 日收益
INDUSTRY_CLOSE_PATH = BASE_DIR / "industry_close_daily.pq"  # 行业 × 净值（首日 1000）
INDUSTRY_RET_SOURCE_PATH = BASE_DIR / "industry_ret_source.pq"  # int8：1 指数 / 2 重建
INDUSTRY_RECON_CALIB_PATH = BASE_DIR / "industry_recon_calibration.csv"
BENCHMARK_PATH = BASE_DIR / "benchmark_daily.pq"
PIT_FINANCIALS_PATH = BASE_DIR / "pit_financials.pq"

# 行业收益重建：候选加权方式按校准自动选择；重建收益为全收益口径而指数为价格口径，
# 回补时扣除校准得到的中位年化偏差（≈分红收益率）
RECON_WEIGHTINGS = ("equal", "mcap", "fmcap")
RECON_BIAS_ADJUST = True

# ---------- 因子层 ----------
FACTOR_DIR = DATA_DIR / "factors"

# ---------- 回测层 ----------
BACKTEST_DIR = DATA_DIR / "backtest"
BACKTEST_TOP_N = 5              # 单因子测试：多头/空头行业数
BACKTEST_N_GROUPS = 5           # 分组测试的组数
BACKTEST_MIN_INDUSTRIES = 20    # 当期有效因子值的行业数下限，不足则持有基准（行业等权）
BACKTEST_COST_BPS = 0.0         # 单边交易成本（基点），按换手率计提；默认不计成本
BACKTEST_BENCHMARK = "ind_ew_monthly"  # 基准列名（base.benchmark()）
BACKTEST_EXCLUDE_INDUSTRIES: tuple[str, ...] = ()  # 不参与选取的行业（仍在基准与因子表中）

# ---------- 指数池层（多对多成员关系的聚合）----------
INDEX_POOL_DIR = BASE_DIR / "index_pool"
INDEX_POOL_MIN_MEMBERS = 5          # 指数当期在个股面板内的有效成份数下限，不足则该期各聚合量记 NaN
INDEX_POOL_BENCHMARK = "csi800"     # 指数级超额基准（benchmark_daily 列名）

# ---------- 风格约束的权重优化（strategy/etf_rotation/optimize.py）----------
STYLE_FACTORS = ("SZ", "NSZ", "BETA", "MOM", "BP", "LIQD", "ResVOL", "EY", "GRO", "LVG")   # 10 维风格暴露的列名
ETF_W_MAX = 0.35                    # 单只指数权重上限
ETF_BETA_TOL = 0.5                  # beta 暴露相对基准的偏离容忍度
ETF_STYLE_TOL = 2.0                 # 其他风格的偏离容忍度
