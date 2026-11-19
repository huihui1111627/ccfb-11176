"""Self-contained HTML report: block map, erase histogram, valid-page
distribution, GC log and remaining-life estimate for both GC policies."""

from __future__ import annotations

import html
import json
import os
from typing import Any

CSS = """
:root{
 --bg:#0e1116; --panel:#161b22; --line:#26303b; --text:#d8e0e8; --dim:#8b98a5;
 --accent:#4fa3ff; --warn:#ffb454; --bad:#ff6b6b; --good:#3ddc97;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
 font:14px/1.55 -apple-system,"PingFang SC","Hiragino Sans GB",Helvetica,Arial,sans-serif}
header{padding:28px 32px 8px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:22px;font-weight:650;letter-spacing:.2px}
h2{margin:34px 0 12px;font-size:17px;font-weight:600}
h3{margin:20px 0 8px;font-size:14px;color:var(--dim);font-weight:600;
 text-transform:uppercase;letter-spacing:.08em}
.sub{color:var(--dim);font-size:13px}
main{padding:0 32px 64px;max-width:1280px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:24px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.card .n{color:var(--dim);font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;background:var(--panel);
 border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);
 font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{background:#1b222c;color:var(--dim);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.05em}
tr:last-child td{border-bottom:none}
.best{color:var(--good);font-weight:650}
.worse{color:var(--warn)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
 border:1px solid var(--line);color:var(--dim)}
.pill.ok{color:var(--good);border-color:#1f4d38;background:#10261c}
.pill.fail{color:var(--bad);border-color:#5a2222;background:#2a1212}
.map{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel);padding:10px}
.map svg{display:block;width:100%;height:auto}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--dim);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
 vertical-align:middle}
.bar{height:10px;background:#1b222c;border-radius:6px;overflow:hidden;display:flex}
.bar span{display:block;height:100%}
.note{background:#131922;border-left:3px solid var(--accent);padding:10px 14px;
 border-radius:0 8px 8px 0;color:var(--dim);margin:12px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
details{margin:10px 0}
summary{cursor:pointer;color:var(--accent)}
code{background:#1b222c;padding:1px 6px;border-radius:4px;font-size:12px}
"""


def esc(x: Any) -> str:
    return html.escape(str(x))


def fmt(n: Any) -> str:
    if isinstance(n, float):
        return f"{n:,.3f}"
    if isinstance(n, int):
        return f"{n:,}"
    return esc(n)


# -- block map ------------------------------------------------------------

def block_map_svg(blocks: list[dict], cols: int = 64) -> str:
    """Colour blocks by erase count; height shows valid-page fill.

    ``blocks`` items: {block, erases, valid, cap, kind}
    """
    if not blocks:
        return "<p class=sub>no block data</p>"
    n = len(blocks)
    rows = (n + cols - 1) // cols
    cell = 13
    gap = 2
    w = cols * (cell + gap)
    h = rows * (cell + gap)
    max_erase = max((b["erases"] for b in blocks), default=1) or 1
    out = [
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="block wear map, {n} blocks, {cols} per row">'
    ]
    for i, b in enumerate(blocks):
        r, c = divmod(i, cols)
        x = c * (cell + gap)
        y = r * (cell + gap)
        kind = b["kind"]
        if kind == "bad":
            fill = "#3a1f1f"
            title = f"block {b['block']} BAD erase={b['erases']}"
        elif kind == "retired":
            fill = "#2b2b2b"
            title = f"block {b['block']} RETIRED erase={b['erases']}"
        else:
            t = min(1.0, b["erases"] / max_erase)
            # cool -> hot: teal to amber to red
            if t < 0.5:
                r0, g0, b0 = 61, 220, 151
                r1, g1, b1 = 255, 180, 84
                u = t / 0.5
            else:
                r0, g0, b0 = 255, 180, 84
                r1, g1, b1 = 255, 107, 107
                u = (t - 0.5) / 0.5
            rr = int(r0 + (r1 - r0) * u)
            gg = int(g0 + (g1 - g0) * u)
            bb = int(b0 + (b1 - b0) * u)
            fill = f"rgb({rr},{gg},{bb})"
            valid_frac = b["valid"] / b["cap"] if b["cap"] else 0
            title = (f"block {b['block']} erase={b['erases']} "
                     f"valid={b['valid']}/{b['cap']} fill={valid_frac:.0%}")
        out.append(
            f'<g><title>{esc(title)}</title>'
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" '
            f'fill="{fill}" opacity="{0.35 + 0.65 * (b["valid"] / b["cap"] if b["cap"] else 0):.2f}"/>'
            f'</g>'
        )
    out.append("</svg>")
    return "".join(out)


# -- svg bar chart --------------------------------------------------------

def bar_chart(labels: list[str], series: list[tuple[str, list[float], str]],
              *, width: int = 900, height: int = 240, ylog: bool = False) -> str:
    n = len(labels)
    if not n:
        return "<p class=sub>no data</p>"
    pad_l, pad_b, pad_t, pad_r = 54, 34, 12, 12
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    vals = [v for _, vs, _ in series for v in vs]
    vmax = max(vals) if vals else 1
    import math

    def y(v: float) -> float:
        if ylog:
            lv = math.log10(max(v, 1))
            lmax = math.log10(max(vmax, 1)) or 1
            return pad_t + ph - (lv / lmax) * ph
        return pad_t + ph - (v / vmax) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    # grid
    for i in range(5):
        gy = pad_t + ph * i / 4
        val = (vmax * (1 - i / 4)) if not ylog else 10 ** (math.log10(max(vmax, 1)) * (1 - i / 4))
        out.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
                   f'stroke="#26303b"/>')
        out.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" fill="#8b98a5" font-size="10" '
                   f'text-anchor="end">{val:.0f}</text>')
    group_w = pw / n
    bw = group_w * 0.8 / max(1, len(series))
    for i, lab in enumerate(labels):
        for j, (name, vs, colour) in enumerate(series):
            v = vs[i]
            bx = pad_l + i * group_w + group_w * 0.1 + j * bw
            by = y(v)
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(1, bw - 1):.1f}" '
                       f'height="{pad_t + ph - by:.1f}" fill="{colour}" rx="1.5">'
                       f'<title>{esc(name)} {esc(lab)}: {v}</title></rect>')
        if n <= 40 or i % max(1, n // 20) == 0:
            out.append(f'<text x="{pad_l + i * group_w + group_w / 2:.1f}" '
                       f'y="{height - 12}" fill="#8b98a5" font-size="10" '
                       f'text-anchor="middle">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(xs: list[float], series: list[tuple[str, list[float], str]],
               xlabel: str = "", ylabel: str = "", *, width: int = 900,
               height: int = 260) -> str:
    if not xs:
        return "<p class=sub>no data</p>"
    pad_l, pad_b, pad_t, pad_r = 58, 40, 14, 16
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    ys = [v for _, vs, _ in series for v in vs]
    ymin, ymax = min(ys + [0]), max(ys) or 1
    xmin, xmax = min(xs), max(xs) or 1
    if xmax == xmin:
        xmax = xmin + 1

    def px(x): return pad_l + (x - xmin) / (xmax - xmin) * pw
    def py(y): return pad_t + ph - (y - ymin) / (ymax - ymin or 1) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        gy = pad_t + ph * i / 4
        val = ymax - (ymax - ymin) * i / 4
        out.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" stroke="#26303b"/>')
        out.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" fill="#8b98a5" font-size="10" text-anchor="end">{val:.0f}</text>')
    for name, vs, colour in series:
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, vs))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.8"/>')
    for i in range(5):
        gx = pad_l + pw * i / 4
        val = xmin + (xmax - xmin) * i / 4
        out.append(f'<text x="{gx:.1f}" y="{height - 14}" fill="#8b98a5" font-size="10" text-anchor="middle">{val:.0f}</text>')
    out.append(f'<text x="{pad_l + pw/2:.0f}" y="{height - 2}" fill="#8b98a5" font-size="10" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="12" y="{pad_t + ph/2:.0f}" fill="#8b98a5" font-size="10" text-anchor="middle" transform="rotate(-90 12 {pad_t + ph/2:.0f})">{esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)
