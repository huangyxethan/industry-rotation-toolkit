"""chartbook 公共设施：绘图样式、图表落盘、章节/区块数据结构与通用绘图函数。"""

import dataclasses
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from strategy import config

matplotlib.use("Agg")

CHARTBOOK_DIR = config.OUTPUT_DIR
FIG_DIR = CHARTBOOK_DIR / "figs"

# 配色：主策略 / 基准 / 超额 / 正 / 负 / 中性
C_PORT, C_BENCH, C_EXCESS, C_POS, C_NEG, C_GREY = "#1f4e79", "#9aa5b1", "#c0392b", "#c0392b", "#1e8449", "#7f8c8d"
C_HELD, C_DROP_CROWD, C_DROP_PROSP, C_NONE = "#1f4e79", "#e67e22", "#8e44ad", "#eeeeee"
STATUS_LABELS = {0: "未持有", 1: "持有", 2: "拥挤剔除", 3: "低景气剔除"}


def setup_style() -> None:
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "AR PL UMing CN", "WenQuanYi Zen Hei",
                 "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 130, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "figure.max_open_warning": 0,
    })


@dataclasses.dataclass
class Block:
    kind: str                       # fig | table | text | cards
    title: str = ""
    path: Path | None = None        # fig：PNG 路径
    caption: str = ""               # fig：图下说明（含自动结论）
    html: str = ""                  # table：表格 HTML
    text: str = ""                  # text：段落
    items: list = dataclasses.field(default_factory=list)  # cards：[(label, value, sub)]
    fig: object = None              # 供 PDF 输出复用的 figure 对象（渲染后关闭）


@dataclasses.dataclass
class Chapter:
    key: str
    title: str
    question: str                   # 本章回答的业务问题
    blocks: list[Block] = dataclasses.field(default_factory=list)
    conclusions: list[str] = dataclasses.field(default_factory=list)

    def add_fig(self, fig, name: str, title: str, caption: str = "") -> Block:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / f"{self.key}_{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        b = Block("fig", title=title, path=path, caption=caption, fig=fig)
        self.blocks.append(b)
        return b

    def add_table(self, df: pd.DataFrame, title: str, fmt: dict | None = None, index: bool = True,
                  caption: str = "") -> Block:
        b = Block("table", title=title, html=df_to_html(df, fmt, index), caption=caption)
        self.blocks.append(b)
        return b

    def add_text(self, text: str, title: str = "") -> Block:
        b = Block("text", title=title, text=text)
        self.blocks.append(b)
        return b

    def add_cards(self, items: list[tuple[str, str, str]], title: str = "") -> Block:
        b = Block("cards", title=title, items=items)
        self.blocks.append(b)
        return b


# ---------- 格式化 ----------

def pct(x, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def df_to_html(df: pd.DataFrame, fmt: dict | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        f = fmt.get(c)
        if f == "pct":
            d[c] = d[c].map(lambda v: pct(v, 1))
        elif f == "pct2":
            d[c] = d[c].map(lambda v: pct(v, 2))
        elif f == "num":
            d[c] = d[c].map(lambda v: num(v, 2))
        elif f == "num3":
            d[c] = d[c].map(lambda v: num(v, 3))
        elif f == "int":
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        elif callable(f):
            d[c] = d[c].map(f)
    return d.to_html(index=index, border=0, classes="tbl", escape=False, na_rep="—")


# ---------- 通用绘图 ----------

def fig_nav_with_excess(daily: pd.DataFrame, port_col: str, bench_col: str, labels=("策略", "基准"),
                        title: str = "") -> plt.Figure:
    """上：策略与基准净值；下：超额净值（比值）及其回撤阴影。"""
    nav_p = (1 + daily[port_col].fillna(0)).cumprod()
    nav_b = (1 + daily[bench_col].fillna(0)).cumprod()
    ex = nav_p / nav_b
    dd = ex / ex.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    a1.plot(nav_p, color=C_PORT, lw=1.6, label=labels[0])
    a1.plot(nav_b, color=C_BENCH, lw=1.3, label=labels[1])
    a1.set_ylabel("净值（起点 1）")
    a1.legend(loc="upper left")
    a1.set_title(title)
    a2.plot(ex, color=C_EXCESS, lw=1.4, label="超额净值（策略 / 基准）")
    a2b = a2.twinx()
    a2b.fill_between(dd.index, dd * 100, 0, color=C_EXCESS, alpha=0.15, label="超额回撤（右轴，%）")
    a2b.set_ylim(min(dd.min() * 100 * 2.2, -5), 0)
    a2b.set_ylabel("超额回撤 %")
    a2b.grid(False)
    a2.set_ylabel("超额净值")
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    return fig


def fig_yearly_bars(yearly: pd.DataFrame, labels=("策略", "基准"), title: str = "") -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(yearly))
    a1.bar(x - 0.2, yearly.port * 100, 0.4, color=C_PORT, label=labels[0])
    a1.bar(x + 0.2, yearly.bench * 100, 0.4, color=C_BENCH, label=labels[1])
    a1.set_xticks(x, yearly.index.astype(str))
    a1.axhline(0, color="black", lw=0.6)
    a1.set_ylabel("年度收益 %")
    a1.legend()
    a1.set_title("分年度收益")
    colors = [C_POS if v > 0 else C_NEG for v in yearly.excess]
    a2.bar(x, yearly.excess * 100, 0.6, color=colors)
    for i, v in enumerate(yearly.excess * 100):
        a2.text(i, v + (0.8 if v >= 0 else -2.5), f"{v:.1f}", ha="center", fontsize=8)
    a2.set_xticks(x, yearly.index.astype(str))
    a2.axhline(0, color="black", lw=0.6)
    a2.set_ylabel("年度超额 %")
    a2.set_title("分年度超额（净值比口径）")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_month_heatmap(monthly_excess: pd.Series, title: str = "月度超额热力图") -> plt.Figure:
    s = monthly_excess.copy()
    tab = pd.DataFrame({"y": s.index.year, "m": s.index.month, "v": s.values * 100}).pivot(index="y", columns="m", values="v")
    tab = tab.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(tab) + 1.4))
    vmax = np.nanpercentile(np.abs(tab.values), 95)
    im = ax.imshow(tab.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), [f"{m}月" for m in range(1, 13)])
    ax.set_yticks(range(len(tab)), tab.index.astype(str))
    ax.grid(False)
    for i in range(tab.shape[0]):
        for j in range(12):
            v = tab.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < vmax * 0.7 else "white")
    yr = tab.mean(axis=1, skipna=True)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="月超额 %")
    fig.tight_layout()
    return fig


def fig_heatmap(tab: pd.DataFrame, title: str, cmap: str = "RdYlGn", fmt: str = "{:.1f}", cbar_label: str = "",
                ax=None, center: float | None = None, vmin=None, vmax=None, xlabel="", ylabel=""):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(0.7 * tab.shape[1] + 2.5, 0.5 * tab.shape[0] + 1.5))
    vals = tab.values.astype(float)
    if center is not None:
        r = np.nanmax(np.abs(vals - center))
        vmin, vmax = center - r, center + r
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(tab.shape[1]), [str(c) for c in tab.columns], rotation=45 if tab.shape[1] > 8 else 0,
                  ha="right" if tab.shape[1] > 8 else "center")
    ax.set_yticks(range(tab.shape[0]), [str(i) for i in tab.index])
    ax.grid(False)
    norm = im.norm
    cm = im.cmap
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                r, g, b, _ = cm(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if lum < 0.45 else "black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if own:
        ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
        ax.figure.tight_layout()
        return ax.figure
    return im


def drawdown_events(nav_series: pd.Series, top: int = 3) -> pd.DataFrame:
    """回撤事件：峰值日、谷底日、深度、修复日（未修复记空）、持续与修复天数。"""
    s = nav_series.dropna()
    peak = s.cummax()
    dd = s / peak - 1
    events = []
    in_dd, start = False, None
    for i, (t, v) in enumerate(dd.items()):
        if not in_dd and v < 0:
            in_dd, start = True, s.index[i - 1] if i > 0 else t
        elif in_dd and v == 0:
            seg = dd.loc[start:t]
            events.append({"peak": start, "trough": seg.idxmin(), "depth": seg.min(), "recover": t})
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        events.append({"peak": start, "trough": seg.idxmin(), "depth": seg.min(), "recover": pd.NaT})
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    ev["to_trough_days"] = [(b - a).days for a, b in zip(ev.peak, ev.trough)]
    ev["to_recover_days"] = [(b - a).days if pd.notna(b) else np.nan for a, b in zip(ev.trough, ev.recover)]
    return ev.sort_values("depth").head(top).reset_index(drop=True)


def rolling_metrics(monthly: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """滚动 window 月：累计超额（净值比）、年化 IR。monthly 需含 port/bench/excess。"""
    ex = monthly.excess
    roll_ex = (1 + ex).rolling(window).apply(np.prod, raw=True) - 1
    roll_ir = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(12)
    return pd.DataFrame({"roll_excess": roll_ex, "roll_ir": roll_ir})


def close_all(chapters: list[Chapter]) -> None:
    for ch in chapters:
        for b in ch.blocks:
            if b.fig is not None:
                plt.close(b.fig)
                b.fig = None
