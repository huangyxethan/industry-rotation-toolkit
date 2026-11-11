"""因子层：行业月频因子，落地 .data/factors/{name}.pq。

因子表约定：index = 月末调仓日（交易日，datetime64[ns]），columns = 行业（顺序同 base.industry_names()），
值为原始因子值（未做截面标准化；合成与排序由回测层处理）。因子在 t 的值只使用 t 收盘前可得的数据。

注册表 REGISTRY 记录每个因子的维度、方向（sign=+1 越高越好，-1 越低越好）与构建函数。
本公开版本不附带具体因子：因子模块定义自己的 FactorSpec 并调用 register() 登记（见 examples/01_single_factor_test.py）。

读取：from strategy import factors; factors.load("mom_12_1")
"""

import pandas as pd

from .common import FactorSpec, load, save  # noqa: F401

REGISTRY: dict[str, FactorSpec] = {}


def register(*specs: FactorSpec) -> None:
    """登记因子规格；同名覆盖。"""
    for s in specs:
        REGISTRY[s.name] = s


def names(dimension: str | None = None) -> list[str]:
    return [n for n, s in REGISTRY.items() if dimension is None or s.dimension == dimension]


def build(name: str) -> dict[str, pd.DataFrame]:
    """运行已登记因子的构建函数，并把它返回的全部表落盘到 FACTOR_DIR。"""
    spec = REGISTRY[name]
    tables = spec.build()
    for key, df in tables.items():
        save(key, df)
    return tables
