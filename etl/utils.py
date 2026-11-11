"""ETL 公共工具：日志、限速、带重试的请求包装。"""

import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from . import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class RateLimiter:
    """保证相邻调用之间的最小间隔。"""

    def __init__(self, interval_sec: float = config.REQUEST_INTERVAL_SEC):
        self.interval = interval_sec
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.monotonic()


def with_retry(func):
    """指数退避重试。重试耗尽后抛出最后一次异常，由调用方决定跳过或中止。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 网络层异常类型不可枚举
                last_exc = exc
                wait = config.RETRY_BACKOFF_SEC * (2 ** attempt)
                time.sleep(wait)
        raise last_exc

    return wrapper


def setup_run_logging(run_name: str) -> Path:
    """把本次运行的全部日志同时写入 .data/logs/{run_name}_{时间戳}.log，并维护 {run_name}_latest.log 软链。

    通过在 root logger 上挂 FileHandler 实现，各模块 get_logger 的控制台输出不受影响。
    返回日志文件路径（编排器用它追加子进程输出）。
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    latest = config.LOG_DIR / f"{run_name}_latest.log"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)
    return path
