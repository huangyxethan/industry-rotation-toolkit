"""把章节渲染为自包含 HTML（左侧两级目录侧边栏 + 图片 base64 内嵌，JS 滚动定位与高亮）与 PDF（matplotlib PdfPages，每图一页）。"""

import base64
import datetime as dt
import html
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .common import CHARTBOOK_DIR, Chapter

CSS = """
*{box-sizing:border-box;}
body{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f6f7f9;color:#222;}
.layout{display:flex;min-height:100vh;}
aside.side{position:fixed;left:0;top:0;bottom:0;width:260px;background:#1f2f45;color:#d6dde6;overflow-y:auto;padding:18px 0 30px;}
aside .brand{padding:0 18px 12px;border-bottom:1px solid #34475f;margin-bottom:8px;}
aside .brand .t{font-size:16px;font-weight:700;color:#fff;} aside .brand .s{font-size:11px;color:#9fb0c3;margin-top:4px;line-height:1.5;}
aside a{display:block;color:#d6dde6;text-decoration:none;padding:7px 18px;font-size:13.5px;line-height:1.35;cursor:pointer;border-left:3px solid transparent;}
aside a:hover{background:#2b3f5a;color:#fff;}
aside a.ch{font-weight:600;margin-top:4px;}
aside a.ch.active{background:#2b3f5a;border-left-color:#e67e22;color:#fff;}
aside a.sub{padding:4px 18px 4px 34px;font-size:12px;color:#aab8c8;}
aside a.sub.active{color:#fff;border-left-color:#e67e22;}
aside .num{display:inline-block;width:20px;color:#9fb0c3;}
main.content{margin-left:260px;flex:1;padding:24px 34px 80px;max-width:1240px;}
header.top{border-bottom:3px solid #1f4e79;padding-bottom:10px;margin-bottom:18px;}
header.top h1{margin:0;font-size:26px;color:#1f4e79;} header.top .meta{color:#666;font-size:13px;margin-top:6px;}
section.ch{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:20px 26px;margin-bottom:28px;scroll-margin-top:12px;}
section.ch h2{margin:0 0 4px;font-size:21px;color:#1f4e79;} section.ch h2 .num{color:#9aa5b1;margin-right:8px;}
.question{color:#555;font-style:italic;margin-bottom:14px;}
.concl{background:#eef3f8;border-left:4px solid #1f4e79;padding:10px 14px;margin:0 0 18px;}
.concl ul{margin:6px 0 0;padding-left:20px;} .concl li{margin:3px 0;}
.block{margin:22px 0;scroll-margin-top:12px;} .block h3{font-size:16px;margin:0 0 8px;color:#333;}
.block img{max-width:100%;border:1px solid #eee;} .caption{color:#555;font-size:13px;margin-top:6px;line-height:1.55;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;}
.card{background:#f9fafb;border:1px solid #e3e6ea;border-radius:6px;padding:12px 14px;}
.card .v{font-size:22px;font-weight:700;color:#1f4e79;} .card .l{font-size:12px;color:#666;} .card .s{font-size:11px;color:#999;}
table.tbl{border-collapse:collapse;font-size:12.5px;width:100%;} table.tbl th,table.tbl td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;}
table.tbl th{background:#f1f3f5;text-align:center;} table.tbl td:first-child,table.tbl th:first-child{text-align:left;}
.tblwrap{overflow-x:auto;} .text{line-height:1.7;font-size:14px;}
footer{color:#888;font-size:12px;margin-top:30px;}
@media (max-width:900px){aside.side{position:static;width:auto;} .layout{flex-direction:column;} main.content{margin-left:0;padding:16px;}}
@media print{aside.side{display:none;} main.content{margin-left:0;} section.ch{break-before:page;border:none;}}
"""

JS = """
(function(){
  var links=document.querySelectorAll('aside a[data-target]');
  links.forEach(function(a){a.addEventListener('click',function(e){
    e.preventDefault();
    var el=document.getElementById(a.getAttribute('data-target'));
    if(el){el.scrollIntoView({behavior:'smooth',block:'start'});
      try{history.replaceState(null,'','#'+el.id);}catch(err){}}
  });});
  var chs=Array.prototype.slice.call(document.querySelectorAll('section.ch'));
  var blocks=Array.prototype.slice.call(document.querySelectorAll('.block[id]'));
  function setActive(){
    var y=window.scrollY+80, cur=null, curb=null;
    chs.forEach(function(s){if(s.offsetTop<=y)cur=s.id;});
    blocks.forEach(function(b){if(b.offsetTop<=y)curb=b.id;});
    links.forEach(function(a){
      var t=a.getAttribute('data-target');
      a.classList.toggle('active', t===cur || t===curb);
    });
    var act=document.querySelector('aside a.ch.active');
    if(act){var side=document.querySelector('aside.side');var r=act.getBoundingClientRect();
      if(r.top<0||r.bottom>window.innerHeight)act.scrollIntoView({block:'nearest'});}
  }
  window.addEventListener('scroll',setActive,{passive:true});
  window.addEventListener('resize',setActive);
  setActive();
  if(location.hash){var el=document.getElementById(location.hash.slice(1));if(el)setTimeout(function(){el.scrollIntoView();},50);}
})();
"""


def _img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _block_id(ch: Chapter, i: int) -> str:
    return f"{ch.key}-b{i}"


def render_html(chapters: list[Chapter], title: str, subtitle: str, out: Path) -> Path:
    side = [f"<div class='brand'><div class='t'>{html.escape(title)}</div><div class='s'>{html.escape(subtitle)}</div></div>"]
    for k, c in enumerate(chapters, 1):
        side.append(f"<a class='ch' href='#{c.key}' data-target='{c.key}'><span class='num'>{k}</span>{html.escape(c.title)}</a>")
        for i, b in enumerate(c.blocks):
            if b.kind in ("fig", "table") and b.title:
                side.append(f"<a class='sub' href='#{_block_id(c, i)}' data-target='{_block_id(c, i)}'>{html.escape(b.title)}</a>")
    parts = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class='layout'>",
             "<aside class='side'>" + "".join(side) + "</aside>", "<main class='content'>",
             f"<header class='top'><h1>{html.escape(title)}</h1><div class='meta'>{html.escape(subtitle)}</div></header>"]
    for k, c in enumerate(chapters, 1):
        parts.append(f"<section class='ch' id='{c.key}'><h2><span class='num'>{k}</span>{html.escape(c.title)}</h2>"
                     f"<div class='question'>{html.escape(c.question)}</div>")
        if c.conclusions:
            parts.append("<div class='concl'><b>要点</b><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in c.conclusions) + "</ul></div>")
        for i, b in enumerate(c.blocks):
            parts.append(f"<div class='block' id='{_block_id(c, i)}'>")
            if b.title and b.kind != "text":
                parts.append(f"<h3>{html.escape(b.title)}</h3>")
            if b.kind == "fig":
                parts.append(f"<img src='{_img_b64(b.path)}' alt='{html.escape(b.title)}'>")
                if b.caption:
                    parts.append(f"<div class='caption'>{html.escape(b.caption)}</div>")
            elif b.kind == "table":
                parts.append(f"<div class='tblwrap'>{b.html}</div>")
                if b.caption:
                    parts.append(f"<div class='caption'>{html.escape(b.caption)}</div>")
            elif b.kind == "cards":
                parts.append("<div class='cards'>" + "".join(
                    f"<div class='card'><div class='v'>{html.escape(v)}</div><div class='l'>{html.escape(l)}</div>"
                    f"<div class='s'>{html.escape(s)}</div></div>" for l, v, s in b.items) + "</div>")
            elif b.kind == "text":
                if b.title:
                    parts.append(f"<h3>{html.escape(b.title)}</h3>")
                if b.text:
                    parts.append(f"<div class='text'>{b.text}</div>")
            parts.append("</div>")
        parts.append("</section>")
    parts.append(f"<footer>生成时间 {dt.datetime.now():%Y-%m-%d %H:%M}；所有图表与结论均由本项目运行产出自动生成。</footer>"
                 f"</main></div><script>{JS}</script></body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _text_page(pdf: PdfPages, title: str, lines: list[str], size=(11, 8.5)) -> None:
    fig = plt.figure(figsize=size)
    fig.text(0.06, 0.9, title, fontsize=20, fontweight="bold", color="#1f4e79")
    y = 0.82
    for ln in lines:
        fig.text(0.06, y, ln, fontsize=11, wrap=True, va="top")
        y -= 0.06 + 0.03 * (len(ln) // 90)
    pdf.savefig(fig)
    plt.close(fig)


def render_pdf(chapters: list[Chapter], title: str, subtitle: str, out: Path) -> Path:
    with PdfPages(out) as pdf:
        _text_page(pdf, title, [subtitle, ""] + [f"{i + 1}. {c.title} —— {c.question}" for i, c in enumerate(chapters)])
        for c in chapters:
            _text_page(pdf, c.title, [c.question, ""] + [f"• {t}" for t in c.conclusions])
            for b in c.blocks:
                if b.kind == "fig" and b.fig is not None:
                    pdf.savefig(b.fig, bbox_inches="tight")
    return out
