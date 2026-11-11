"""数据层全局配置：路径与请求控制。

本公开版本不含数据拉取流水线：`.data/` 由 `scripts/make_synthetic_data.py` 生成，
或由使用者自己的加载器按 `docs/data_formats.md` 描述的格式写入。
"""

from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# data/  外部提供、不可再生的数据（不入库）
# .data/ 项目自行加工的数据，可重新生成
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = PROJECT_ROOT / ".data"
LOG_DIR = DATA_DIR / "logs"        # 每次运行的完整日志（带时间戳 + *_latest.log）
REPORT_DIR = DATA_DIR / "reports"  # 运行产生的结构化诊断/结果表，供结果展示模块读取

# ---------- 请求控制（etl.utils.RateLimiter / with_retry）----------
REQUEST_INTERVAL_SEC = 0.5   # 相邻请求最小间隔
MAX_RETRIES = 3              # 单请求最大重试次数
RETRY_BACKOFF_SEC = 5.0      # 重试退避基数（指数退避：5s, 10s, 20s）
