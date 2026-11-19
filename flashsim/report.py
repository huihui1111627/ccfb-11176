"""Build the self-contained HTML report from experiment results."""

from __future__ import annotations

import html
import json
import os

from .report_assets import CSS, bar_chart, block_map_svg, esc, fmt, line_chart


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def _card(k: str, v: str, note: str = "") -> str:
    return f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{v}</div>' \
           f'<div class="n">{note}</div></div>'


def build_report(data: dict, path: str) -> None:
    wl = data.get("workload", {})
    pol = data.get("policy", {})
    pl = data.get("powerloss", {})
    pol_cfg = data.get("policy_config", {})
    geom = data.get("geometry", {})
    mapdata = data.get("block_map", {})
    timeline = data.get("timeline", {})
    gclog = data.get("gc_log", {})

    order = [p for p in ("greedy", "cost") if p in pol]
    names = {"greedy": "只按有效页数（对照组）", "cost": "有效页数 + 擦除次数（默认）"}
    colours = {"greedy": "#ffb454", "cost": "#4fa3ff"}

    head = [
        "<!doctype html><html lang=zh-CN><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<title>植保无人机飞行日志 · NAND FTL 仿真报告</title>",
        f"<style>{CSS}</style></head><body>",
        "<header><h1>植保无人机飞行日志 · 裸闪存 FTL 仿真报告</h1>",
        f"<div class=sub>{esc(geom.get('describe',''))} · 逻辑容量 "
        f"{fmt(geom.get('logical_pages',0))} 页 / "
        f"{geom.get('logical_mib',0):.0f} MiB · 预留空间 "
        f"{_pct(100*geom.get('op_ratio',0))}</div></header><main>",
    ]

    # ---- top-line comparison ------------------------------------------
    head.append("<h2>两种回收策略的量化对比</h2>")
    head.append("<table><tr><th>指标</th>" +
                "".join(f"<th>{names[p]}</th>" for p in order) + "<th>差异</th></tr>")

    def row(label: str, key: str, f=lambda x: fmt(x), better="low") -> str:
        vals = [pol[p][key] for p in order]
        cells = "".join(f"<td>{f(v)}</td>" for v in vals)
        diff = ""
        if len(vals) == 2 and vals[0]:
            d = (vals[1] - vals[0]) / abs(vals[0]) * 100
            cls = "best" if (d < 0) == (better == "low") and abs(d) > 0.5 else (
                "worse" if abs(d) > 0.5 else "")
            if key in ("wa",):
                cls = "best" if d < -0.5 else ("worse" if d > 0.5 else "")
            diff = f'<td class="{cls}">{d:+.1f}%</td>'
        return f"<tr><td>{label}</td>{cells}{diff}</tr>"

    head.append(row("写放大 WA（物理页 / 主机页）", "wa", lambda x: f"{x:.3f}"))
    head.append(row("块擦除总次数", "erases"))
    head.append(row("GC 回收块数", "gc_erases"))
    head.append(row("GC 搬运的有效页", "gc_copies"))
    head.append(row("静态磨损均衡搬块数", "wl_moves"))
    head.append("<tr><td>擦除次数 最大 / 最小 / 平均</td>" + "".join(
        f"<td>{pol[p]['wear']['max']} / {pol[p]['wear']['min']} / {pol[p]['wear']['mean']:.1f}</td>"
        for p in order) + "</tr>")
    head.append("<tr><td>擦除次数 极差 / 标准差</td>" + "".join(
        f"<td>{pol[p]['wear']['spread']} / {pol[p]['wear']['stdev']:.2f}</td>"
        for p in order) + "</tr>")
    head.append("<tr><td>最热块已用寿命</td>" + "".join(
        f"<td>{pol[p]['lifetime']['used_pct_of_pe']:.1f}%</td>" for p in order) + "</tr>")
    head.append("<tr><td>按最热块推算还可写入</td>" + "".join(
        f"<td>{pol[p]['lifetime']['host_writes_until_wearout']/1e6:.1f} M 页</td>"
        for p in order) + "</tr>")
    head.append("<tr><td>运行中新增坏块 / 出厂坏块</td>" + "".join(
        f"<td>{pol[p]['runtime_bad']} / {pol[p]['factory_bad']}</td>"
        for p in order) + "</tr>")
    head.append("</table>")

    # ---- pressure sweep -------------------------------------------------
    sweep = data.get("sweep", [])
    if sweep:
        head.append("<h2>预留空间扫描：策略差异随压力放大</h2>")
        head.append(
            "<div class=note>预留空间宽松时，回收到的块几乎都是纯垃圾，任何策略表现都一样。"
            "预留空间收紧、每次回收都得带走一半以上有效页时，"
            "“挑哪个块”才真正开始影响写放大和擦除次数分布。</div>")
        ratios = sorted({r["op_ratio"] for r in sweep})
        head.append("<table><tr><th>预留空间</th>"
                    + "".join(f"<th>{names[p]}<br>WA / 擦除 / 极差 / σ</th>" for p in order)
                    + "<th>WA 改善</th></tr>")
        for op in ratios:
            cells = []
            was = {}
            for p in order:
                row = next((r for r in sweep if r["op_ratio"] == op and r["policy"] == p), None)
                if row is None:
                    cells.append("<td>—</td>")
                    continue
                was[p] = row["wa"]
                cells.append(f"<td>{row['wa']:.3f} / {row['erases']:,} / "
                             f"{row['spread']} / {row['stdev']:.1f}</td>")
            delta = ""
            if len(was) == 2 and was[order[0]]:
                d = (was[order[1]] - was[order[0]]) / was[order[0]] * 100
                cls = "best" if d < -0.5 else ("worse" if d > 0.5 else "")
                delta = f'<td class="{cls}">{d:+.1f}%</td>'
            head.append(f"<tr><td>{op:.0%}</td>" + "".join(cells) + delta + "</tr>")
        head.append("</table>")
        xs = [f"{op:.0%}" for op in ratios]
        for metric, label, ylab in (("wa", "写放大", "WA"), ("erases", "擦除次数", "块擦除"),
                                    ("spread", "擦除极差", "max-min")):
            series = []
            for p in order:
                vals = []
                for op in ratios:
                    row = next((r for r in sweep if r["op_ratio"] == op and r["policy"] == p), None)
                    vals.append(row[metric] if row else 0)
                series.append((names[p], vals, colours[p]))
            head.append(f"<h3>{label}</h3>")
            head.append(bar_chart(xs, series, width=900, height=220))
        head.append("<div class=legend>" +
                    "".join(f"<span><i style='background:{colours[p]}'></i>{names[p]}</span>"
                            for p in order) + "</div>")

    # ---- key cards ------------------------------------------------------
    head.append("<h2>关键结果</h2><div class=cards>")
    if "cost" in pol:
        c = pol["cost"]
        head.append(_card("主机写入", f"{c['host_writes']/1e6:.2f} M 页",
                          f"{c['host_writes']*4/1024:.1f} GiB 逻辑"))
        head.append(_card("写放大（默认策略）", f"{c['wa']:.2f}×",
                          "物理编程页 / 主机页"))
        head.append(_card("擦除次数", fmt(c["erases"]), "整块擦除"))
        head.append(_card("寿命余量", f"{100-c['lifetime']['used_pct_of_pe']:.0f}%",
                          f"最热块 {c['lifetime']['max_erase_count']}/{c['lifetime']['pe_cycles']} 次"))
        head.append(_card("磨损均衡度", f"{100*c['lifetime']['wear_evenness']:.0f}%",
                          "最冷块/最热块擦除次数"))
    if pl:
        total = sum(v.get("cuts", 0) for v in pl.values())
        fails = sum(v.get("failures", 0) for v in pl.values())
        head.append(_card("掉电注入", f"{total - fails}/{total} 通过",
                          f"撕裂 {sum(v.get('torn_total',0) for v in pl.values())} 次"))
    head.append("</div>")

    # ---- power loss -----------------------------------------------------
    head.append("<h2>掉电一致性验证</h2>")
    head.append("<div class=note>每次掉电发生在一次 NAND 操作的中途（0% 丢弃 / "
                "35% 只落一页 / 35% 页标签损坏 / 其余在操作完成后），重启后逐页比对："
                "每个逻辑页必须读到掉电前最后一次<b>已确认</b>写入的内容；"
                "掉电瞬间在途的那一页允许是旧值、新值或未映射，但不允许出现任何别的数据。</div>")
    head.append("<table><tr><th>策略</th><th>掉电次数</th><th>失败</th>"
                "<th>撕裂操作</th><th>中止操作</th><th>校验页数</th><th>结果</th></tr>")
    for p in order:
        v = pl.get(p, {})
        ok = v.get("failures", 1) == 0
        head.append(
            f"<tr><td>{names[p]}</td><td>{v.get('cuts',0)}</td>"
            f"<td>{v.get('failures',0)}</td><td>{v.get('torn_total',0)}</td>"
            f"<td>{v.get('abort_total',0)}</td><td>{fmt(v.get('checked_pages',0))}</td>"
            f"<td><span class='pill {'ok' if ok else 'fail'}'>"
            f"{'全部通过' if ok else '存在失败'}</span></td></tr>")
    head.append("</table>")
    fails = [(p, f) for p in order for f in pl.get(p, {}).get("first_failures", [])]
    if fails:
        head.append("<details><summary>失败样例</summary><pre>" +
                    esc("\n".join(str(f) for _, f in fails)) + "</pre></details>")

    # ---- block map ------------------------------------------------------
    head.append("<h2>块磨损地图</h2>")
    head.append("<div class=note>每格一个块；颜色=擦除次数（绿→黄→红），"
                "不透明度=有效页占比；暗红格为坏块。悬停可看具体数值。</div>")
    for p in order:
        if p not in mapdata:
            continue
        m = mapdata[p]
        cols = 128 if len(m) > 1024 else 64
        head.append(f"<h3>{names[p]} · {len(m)} 个块</h3><div class=map>")
        head.append(block_map_svg(m, cols=cols))
        head.append("</div>")

    # ---- erase histogram -----------------------------------------------
    head.append("<h2>擦除次数分布</h2>")
    for p in order:
        if p not in pol:
            continue
        hist = pol[p]["histogram"]
        if not hist:
            continue
        keys = sorted(int(k) for k in hist)
        lo, hi = keys[0], keys[-1]
        if hi - lo <= 30:
            labels = [str(k) for k in range(lo, hi + 1)]
            vals = [hist.get(str(k), hist.get(k, 0)) for k in range(lo, hi + 1)]
        else:
            nb = 20
            step = max(1, (hi - lo) // nb)
            labels, vals = [], []
            for start in range(lo, hi + 1, step):
                end = min(hi, start + step - 1)
                labels.append(f"{start}" if start == end else f"{start}-{end}")
                vals.append(sum(hist.get(str(k), hist.get(k, 0))
                                for k in range(start, end + 1)))
        head.append(f"<h3>{names[p]} · 共 {pol[p]['wear']['superblocks']} 个超级块</h3>")
        head.append(bar_chart(labels, [(p, vals, colours[p])], width=900, height=210))
    head.append("<div class=legend>" +
                "".join(f"<span><i style='background:{colours[p]}'></i>{names[p]}</span>"
                        for p in order) + "</div>")

    # ---- write amplification breakdown ----------------------------------
    if "cost" in pol:
        c = pol["cost"]
        head.append("<h2>写放大的构成</h2>")
        head.append("<div class=note>WA 不是单一数字：主机页之外，"
                    "GC 搬运、块擦除标记、映射快照和丢弃记录都算物理编程。"
                    "快照是全量写，所以它随“主机页 / 快照间隔”放大，"
                    "这里单独列出来，便于判断哪些开销该优化。</div>")
        hp = max(1, c["physical_pages"])
        parts = [
            ("主机写", c["host_writes"], "#4fa3ff"),
            ("GC 搬运", c["gc_copies"], "#ffb454"),
            ("擦除标记", c["marker_pages"], "#8b98a5"),
            ("映射快照", c["meta_pages"], "#3ddc97"),
            ("丢弃记录", c["discard_pages"], "#c792ea"),
            ("坏块日志", c.get("badlog_pages", 0), "#e06c9f"),
        ]
        residual = c["physical_pages"] - sum(v for _, v, _ in parts)
        if residual:
            parts.append(("其它", residual, "#6b7785"))
        bar = "".join(
            f'<span style="width:{100*v/hp:.2f}%;background:{col}" title="{esc(lbl)} {v:,}"></span>'
            for lbl, v, col in parts
        )
        head.append(f'<div class="bar" style="height:18px">{bar}</div>')
        head.append("<div class=legend>" + "".join(
            f"<span><i style='background:{col}'></i>{esc(lbl)} {100*v/hp:.1f}% ({v:,} 页)</span>"
            for lbl, v, col in parts) + "</div>")
        head.append("<table><tr><th>构成</th><th>物理页</th><th>占比</th><th>除以主机页</th></tr>")
        for lbl, v, _ in parts:
            head.append(f"<tr><td>{lbl}</td><td>{v:,}</td>"
                        f"<td>{100*v/hp:.2f}%</td>"
                        f"<td>{v/max(1,c['host_writes']):.4f}</td></tr>")
        head.append(f"<tr><td><b>合计</b></td><td><b>{c['physical_pages']:,}</b></td>"
                    f"<td><b>100%</b></td><td><b>{c['wa']:.4f}</b></td></tr></table>")
        head.append(f"<div class=note>本配置每 {pol_cfg.get('检查点间隔','')} 写一次全量快照，"
                    f"共 {c['checkpoints']} 次、{c['meta_pages']:,} 页。"
                    f"把快照间隔调大或改成增量快照即可压低这一项。</div>")

    # ---- valid-page distribution ---------------------------------------
    head.append("<h2>有效页分布</h2>")
    head.append("<div class=note>按有效页占比分桶统计超级块数。回收策略的差异主要在这里："
                "只看有效页数会反复挑中“几乎全空”的块而把冷块一直留在原地；"
                "综合擦除次数会把冷块也拉回回收队列，从而抹平磨损。</div>")
    head.append("<table><tr><th>有效页占比</th>" +
                "".join(f"<th>{names[p]}</th>" for p in order) + "</tr>")
    buckets = [(0, 0), (1, 25), (26, 50), (51, 75), (76, 99), (100, 100)]
    for lo, hi in buckets:
        cells = []
        for p in order:
            vd = data.get("valid_dist", {}).get(p, [])
            cells.append(fmt(vd.get(f"{lo}-{hi}", 0)))
        label = "0（全空）" if lo == hi == 0 else (
            "100%（写满且全有效）" if lo == hi == 100 else f"{lo}–{hi}%")
        head.append(f"<tr><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head.append("</table>")

    # ---- GC timeline ----------------------------------------------------
    if timeline:
        head.append("<h2>GC 与擦除随时间变化</h2>")
        head.append(line_chart(
            timeline.get("x", []),
            [(names[p], timeline[p], colours[p]) for p in order if p in timeline],
            xlabel="主机写入页数", ylabel="累计擦除次数", width=900, height=250))
        head.append("<div class=legend>" +
                    "".join(f"<span><i style='background:{colours[p]}'></i>{names[p]}</span>"
                            for p in order if p in timeline) + "</div>")
        head.append(line_chart(
            timeline.get("x", []),
            [(names[p] + " · 热块擦除", timeline[p + "_hot"], colours[p])
             for p in order if p + "_hot" in timeline],
            xlabel="主机写入页数", ylabel="最热块擦除次数", width=900, height=230))

    # ---- GC log ---------------------------------------------------------
    if gclog:
        head.append("<h2>回收记录</h2>")
        for p in order:
            rows = gclog.get(p, [])
            if not rows:
                continue
            head.append(f"<details><summary>{names[p]} · 最近 {len(rows)} 条</summary>")
            head.append("<table><tr><th>#</th><th>受害块</th><th>原因</th>"
                        "<th>搬运有效页</th><th>块擦除次数</th><th>候选块数</th></tr>")
            for r in rows:
                reason = {"space": "空间不足", "static_wl": "静态磨损均衡",
                          "bad_block": "坏块"}.get(r.get("reason"), r.get("reason"))
                head.append(f"<tr><td>{r.get('seq')}</td><td>{r.get('sb')}</td>"
                            f"<td style='text-align:left'>{reason}</td>"
                            f"<td>{r.get('valid_copied')}</td>"
                            f"<td>{r.get('erase_count')}</td>"
                            f"<td>{r.get('scanned')}</td></tr>")
            head.append("</table></details>")

    # ---- lifetime -------------------------------------------------------
    head.append("<h2>寿命估算</h2>")
    head.append("<div class=note>不以平均擦除次数估算，而以<b>最热的那一个块</b>为准——"
                "闪存寿命由最先写坏的那块决定。剩余寿命按当前最热块的擦除速率外推。"
                "本实验把 P/E 上限缩短到 "
                f"{pol[order[0]]['lifetime']['pe_cycles']:,} 次以便在有限写入量内观察到磨损运动；"
                "最后一行的 30 000 次列是按真实 SLC 标称值等比换算的结果，"
                "逻辑等价于把同样的工作负载跑在同一块盘上直到磨穿。</div>")
    page_mib = data.get("geometry", {}).get("page_size", 4096) / 2**20
    head.append("<table><tr><th>项</th>" + "".join(f"<th>{names[p]}</th>" for p in order) + "</tr>")
    for label, key, f in (
        ("P/E 标称次数", "pe_cycles", fmt),
        ("最热块擦除次数", "max_erase_count", fmt),
        ("已消耗寿命", "used_pct_of_pe", lambda x: f"{x:.1f}%"),
        ("剩余 P/E", "remaining_pe", fmt),
        ("按当前速率还可写入", "host_writes_until_wearout", lambda x: f"{x/1e6:.1f} M 页"),
        ("换算到 30 000 P/E 还可写入", "host_writes_until_wearout_30k",
         lambda x: f"{x/1e9:.2f} G 页 / {x*page_mib/1024/1024:.1f} TiB"),
        ("磨损均衡度（最冷/最热）", "wear_evenness", lambda x: f"{100*x:.0f}%"),
        ("已退役块", "retired_blocks", fmt),
    ):
        head.append(f"<tr><td>{label}</td>" + "".join(
            f"<td>{f(pol[p]['lifetime'][key])}</td>" for p in order) + "</tr>")
    head.append("</table>")

    # ---- methodology ----------------------------------------------------
    head.append("<h2>仿真与实现说明</h2>")
    head.append("<div class=grid2><div>")
    head.append("<h3>设备规则（违规即抛错）</h3><ul>")
    head.append("<li>擦除只能整块，擦除前块内必须无已编程页</li>")
    head.append("<li>每页每次擦除周期只能编程一次</li>")
    head.append("<li>块内页只能按递增偏移编程，空洞不可回填</li>")
    head.append("<li>坏块（出厂 / 使用中）拒绝一切编程与擦除</li>")
    head.append("<li>多平面编程必须落在同一超块对、同一页偏移</li>")
    head.append("</ul><h3>掉电模型</h3><ul>")
    head.append("<li>以 NAND 操作为单位；被选中的那一次操作可能整体丢弃、"
                "只落一半（多平面）、或页标签损坏</li>")
    head.append("<li>恢复只依赖扫描页标签与保留区记录，不依赖“正常关机”</li>")
    head.append("</ul></div><div>")
    head.append("<h3>FTL 结构</h3><ul>")
    head.append("<li>页级映射表常驻内存，另在保留区按双 Bank 轮转写快照</li>")
    head.append("<li>快照带 CRC 与提交页；重启先从最新有效快照加载，再回放更新的页</li>")
    head.append("<li>GC 先搬有效页再擦除，拷贝获得更高序号，掉电中途原件仍可用</li>")
    head.append("<li>块内偏移 0 存擦除标记（含擦除次数），重启后擦除计数不丢</li>")
    head.append("<li>丢弃通知先落盘再撤映射，回收时不搬运被丢弃的页</li>")
    head.append("<li>双平面：两条主机写可合并为一次多平面编程；GC 拷贝同样成对下发</li>")
    head.append("</ul></div></div>")

    if pol_cfg:
        head.append("<h3>运行参数</h3><table><tr><th>参数</th><th>值</th></tr>")
        for k, v in pol_cfg.items():
            head.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
        head.append("</table>")

    head.append("<h3>工作负载</h3><table><tr><th>参数</th><th>值</th></tr>")
    for k, v in wl.items():
        head.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
    head.append("</table>")

    head.append("</main></body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(head))
