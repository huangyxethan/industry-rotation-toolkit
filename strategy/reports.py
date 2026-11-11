"""运行结果存档：把各层构建时产生的诊断表/统计量写入 .data/reports/<section>/，供结果展示模块读取。

约定：表格用 CSV（index 保留），标量与嵌套摘要用 JSON；同名文件每次运行覆盖，历史版本以日志为准。
"""

import json
from pathlib import Path

import pandas as pd

from . import config


def path(section: str, name: str) -> Path:
    d = config.REPORT_DIR / section
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def save_df(section: str, name: str, df: pd.DataFrame, index: bool = True) -> Path:
    p = path(section, name)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def save_json(section: str, name: str, obj) -> Path:
    p = path(section, name)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    return p


def load_json(section: str, name: str) -> dict:
    p = config.REPORT_DIR / section / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_json(section: str, name: str, key: str, value) -> Path:
    obj = load_json(section, name)
    obj[key] = value
    return save_json(section, name, obj)


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    return str(o)
