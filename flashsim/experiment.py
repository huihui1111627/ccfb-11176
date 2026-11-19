#!/usr/bin/env python3
"""Full experiment: millions of flight-log writes, random power cuts on both
GC policies, block-level telemetry, and a self-contained HTML report.

    python3 -m flashsim.experiment --runs-dir runs --report report/index.html
    python3 -m flashsim.experiment --quick          # fast smoke version

Deterministic: same arguments produce the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

from .config import Config, Geometry
from .device import NandDevice, NoSpace, PowerLoss
from .ftl import FTL
from .powerloss import Shadow, check_recovery, reboot
from .workload import FlightLogWorkload, Op, WorkloadParams

# ~4 GiB raw die, 2 planes.  Reduced P/E so a single test run can show wear
# motion; the lifetime section scales the rate to the real 30 000 cycles.
MAIN_GEOMETRY = Geometry(
    page_size=4096,
    pages_per_block=256,
    blocks_per_plane=1024,
    planes=2,
    pe_cycles=3000,
)
QUICK_GEOMETRY = Geometry(
    page_size=4096,
    pages_per_block=128,
    blocks_per_plane=192,
    planes=2,
    pe_cycles=1500,
)


def make_cfg(policy: str, geometry: Geometry, seed: int = 1, **kw) -> Config:
    base = dict(
        geometry=geometry,
        meta_blocks=80,  # must hold >=2 full mapping banks + bad-block log
        op_ratio=0.10,
        gc_policy=policy,
        gc_wear_weight=3.0,
        gc_reserve_pct=3.0,
        static_wl=True,
        dynamic_wl=True,
        static_wl_min_age=20_000,
        factory_bad_pct=0.01,
        erase_fail_prob=1e-4,
        checkpoint_every=8192,
        seed=seed,
    )
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------------------
# single run
# ---------------------------------------------------------------------------

@dataclass
class Run:
    policy: str
    ftl: FTL
    elapsed: float
    block_map: list[dict] = field(default_factory=list)
    gc_recent: list[dict] = field(default_factory=list)
    timeline: dict = field(default_factory=dict)
    valid_dist: dict = field(default_factory=dict)


def drive(ftl: FTL, ops) -> None:
    for op in ops:
        if op.kind == Op.PAIR:
            ftl.write_pair(op.lpn, op.payload, op.lpn_b, op.payload_b)
        elif op.kind == Op.WRITE:
            ftl.write(op.lpn, op.payload)
        else:
            ftl.discard_range(op.lpn, op.lpn_b)


def sample_blocks(ftl: FTL) -> tuple[list[dict], dict]:
    g = ftl.g
    cap = g.pages_per_block - 1
    blocks = []
    for s, info in sorted(ftl.sb.items()):
        for blk in info.blocks:
            kind = info.kind
            blocks.append({
                "block": blk,
                "erases": info.erase_count,
                "valid": info.valid,
                "cap": cap,
                "kind": "bad" if kind == "bad" else ("retired" if kind == "retired" else "ok"),
            })
    dist: dict[str, int] = {}
    for b in blocks:
        if b["kind"] == "bad":
            continue
        pct = int(b["valid"] / cap * 100) if cap else 0
        if pct == 0:
            key = "0-0"
        elif pct == 100:
            key = "100-100"
        elif pct <= 25:
            key = "1-25"
        elif pct <= 50:
            key = "26-50"
        elif pct <= 75:
            key = "51-75"
        else:
            key = "76-99"
        dist[key] = dist.get(key, 0) + 1
    return blocks, dist


def run_policy(policy: str, geometry: Geometry, wl: WorkloadParams,
               *, sample_every: int = 0) -> Run:
    cfg = make_cfg(policy, geometry)
    ftl = FTL(cfg)
    t0 = time.time()
    xs: list[int] = []
    ys: list[int] = []
    hot: list[int] = []
    last_bucket = -1
    try:
        for i, op in enumerate(FlightLogWorkload(wl)):
            if op.kind == Op.PAIR:
                ftl.write_pair(op.lpn, op.payload, op.lpn_b, op.payload_b)
            elif op.kind == Op.WRITE:
                ftl.write(op.lpn, op.payload)
            else:
                ftl.discard_range(op.lpn, op.lpn_b)
            hw = ftl.stats["host_writes"]
            if sample_every and hw >= sample_every and hw // sample_every > last_bucket:
                last_bucket = hw // sample_every
                xs.append(hw)
                ys.append(ftl.stats["erases"])
                hot.append(max((s.erase_count for s in ftl.sb.values()
                                if s.kind != "bad"), default=0))
    except NoSpace:
        pass  # device genuinely full (bad blocks + discarded OP budget)
    ftl.flush()
    elapsed = time.time() - t0
    blocks, dist = sample_blocks(ftl)
    recent = [
        {
            "seq": r.seq, "sb": r.sb, "reason": r.reason,
            "valid_copied": r.valid_copied, "erase_count": r.erase_count,
            "scanned": r.scanned,
        }
        for r in ftl.gc_log[-60:]
    ]
    return Run(
        policy=policy, ftl=ftl, elapsed=elapsed, block_map=blocks,
        gc_recent=recent,
        timeline={"x": xs, policy: ys, policy + "_hot": hot},
        valid_dist=dist,
    )


# ---------------------------------------------------------------------------
# power loss
# ---------------------------------------------------------------------------

def _drive_shadowed(ftl: FTL, shadow: Shadow, wl: WorkloadParams) -> None:
    for op in FlightLogWorkload(wl):
        if op.kind == Op.PAIR:
            shadow.issue_write(op.lpn, op.payload)
            shadow.issue_write(op.lpn_b, op.payload_b)
            ftl.write_pair(op.lpn, op.payload, op.lpn_b, op.payload_b)
            shadow.ack_write(op.lpn, op.payload)
            shadow.ack_write(op.lpn_b, op.payload_b)
        elif op.kind == Op.WRITE:
            shadow.issue_write(op.lpn, op.payload)
            ftl.write(op.lpn, op.payload)
            shadow.ack_write(op.lpn, op.payload)
        else:
            for l in range(op.lpn, op.lpn + op.lpn_b):
                shadow.issue_discard(l)
            ftl.discard_range(op.lpn, op.lpn_b)
            for l in range(op.lpn, op.lpn + op.lpn_b):
                shadow.ack_discard(l)


def powerloss_sweep(geometry: Geometry, wl: WorkloadParams, cuts: int,
                    seed: int, policies=("greedy", "cost")) -> dict:
    base_cfg = make_cfg("cost", geometry)
    # learn the operation count from a clean run
    base = FTL(base_cfg)
    try:
        _drive_shadowed(base, Shadow(base_cfg.logical_pages), wl)
    except NoSpace:
        pass
    total_ops = max(1, base.dev.ops)
    out: dict = {}
    for policy in policies:
        cfg = make_cfg(policy, geometry)
        rnd = random.Random(seed * 131 + hash(policy) % 97)
        results = []
        for _ in range(cuts):
            point = rnd.randrange(1, total_ops)
            dev = NandDevice(geometry, seed=cfg.seed)
            ftl = FTL(cfg, dev=dev)
            shadow = Shadow(cfg.logical_pages)
            dev.set_power_loss_after(point, seed=rnd.randrange(1 << 30))
            try:
                _drive_shadowed(ftl, shadow, wl)
            except PowerLoss:
                pass
            except NoSpace:
                pass
            boot = FTL(cfg, dev=dev)
            boot.recover()
            res = check_recovery(shadow, boot)
            results.append({
                "point": point, "ok": res.ok, "wrong_n": len(res.wrong),
                "wrong": res.wrong[:3], "checked": res.checked,
                "torn": dev.tears, "aborts": dev.aborts,
                "mode": boot.stats["recovery_mode"],
            })
        fails = [r for r in results if not r["ok"]]
        modes: dict[str, int] = {}
        for r in results:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        out[policy] = {
            "cuts": len(results),
            "failures": len(fails),
            "torn_total": sum(r["torn"] for r in results),
            "abort_total": sum(r["aborts"] for r in results),
            "checked_pages": sum(r["checked"] for r in results),
            "recovery_modes": modes,
            "first_failures": fails[:5],
        }
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize(run: Run) -> dict:
    ftl = run.ftl
    st = ftl.stats
    wear = ftl.wear_stats()
    hist = {str(k): v for k, v in ftl.wear_histogram().items()}
    gc_space = [r for r in ftl.gc_log if r.reason == "space"]
    gc_wl = [r for r in ftl.gc_log if r.reason == "static_wl"]
    max_ec = wear["max"]
    min_ec = wear["min"]
    pe = ftl.g.pe_cycles
    host = st["host_writes"]
    writes_per_erase = host / max_ec if max_ec else 0

    def scale_to(pe_target: int) -> float:
        """Host pages until wear-out, holding the wear *rate* constant."""
        if not max_ec:
            return 0.0
        remaining_cycles = max(0, pe_target - max_ec)
        return remaining_cycles * writes_per_erase

    return {
        "policy": run.policy,
        "host_writes": host,
        "host_mib": host * ftl.g.page_size / 2**20,
        "logical_pages": ftl.n_lpn,
        "physical_pages": ftl.physical_pages(),
        "wa": ftl.wa(),
        "erases": st["erases"],
        "gc_erases": st["gc_erases"],
        "gc_copies": st["gc_copies"],
        "wl_moves": st["wl_moves"],
        "wl_copies": st["wl_copies"],
        "multipage_ops": st["multipage_ops"],
        "marker_pages": st["marker_pages"],
        "meta_pages": st["meta_pages"],
        "badlog_pages": st["badlog_pages"],
        "discard_pages": st["discard_pages"],
        "discard_ops": st["discard_ops"],
        "checkpoints": st["checkpoints"],
        "runtime_bad": len(ftl.dev.bad_blocks) - len(ftl.dev.factory_bad),
        "factory_bad": len(ftl.dev.factory_bad),
        "retired_wear": st["retired_wear"],
        "wear": wear,
        "histogram": hist,
        "gc": {
            "invocations": st["gc_invocations"],
            "space_erases": len(gc_space),
            "wl_moves": len(gc_wl),
            "copied_pages": sum(r.valid_copied for r in gc_space),
            "avg_copied": (sum(r.valid_copied for r in gc_space) / len(gc_space)) if gc_space else 0.0,
            "avg_wl_copied": (sum(r.valid_copied for r in gc_wl) / len(gc_wl)) if gc_wl else 0.0,
            "avg_scanned": st["gc_scanned"] / max(1, st["gc_invocations"]),
            "wl_on": ftl.cfg.static_wl,
        },
        "lifetime": {
            "pe_cycles": pe,
            "max_erase_count": max_ec,
            "min_erase_count": min_ec,
            "used_pct_of_pe": 100.0 * max_ec / pe if pe else 0.0,
            "remaining_pe": max(0, pe - max_ec),
            "wear_evenness": (min_ec / max_ec) if max_ec else 1.0,
            "retired_blocks": st["retired_wear"],
            "host_writes_until_wearout": scale_to(pe),
            "host_mib_until_wearout": scale_to(pe) * ftl.g.page_size / 2**20,
            "host_writes_until_wearout_30k": scale_to(30_000),
            "host_mib_until_wearout_30k": scale_to(30_000) * ftl.g.page_size / 2**20,
            "writes_per_erase_on_hottest": writes_per_erase,
        },
        "stats": dict(st),
        "elapsed_s": run.elapsed,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    op_ratio: float
    policy: str
    wa: float
    erases: int
    gc_erases: int
    copies: int
    wear_max: int
    wear_min: int
    spread: int
    stdev: float
    host_writes: int


def pressure_sweep(geometry: Geometry, base_ops: int, seed: int = 7) -> list[dict]:
    """Sweep over-provisioning from loose to tight.

    At high OP almost every victim is pure garbage, so any reasonable policy
    behaves the same.  The differences that matter -- how much live data a
    policy drags along and how it distributes erases -- only appear when free
    space is tight.  This sweep is what makes the comparison meaningful.
    """
    import math

    tight_geom = Geometry(
        page_size=geometry.page_size,
        pages_per_block=geometry.pages_per_block,
        blocks_per_plane=256,
        planes=geometry.planes,
        pe_cycles=geometry.pe_cycles,
    )
    rows: list[dict] = []
    # tighter OP -> smaller logical space -> shorter run keeps the same
    # number of *backing-store* rewrites (~4x capacity)
    for op in (0.30, 0.20, 0.12, 0.08, 0.05):
        base_cfg = make_cfg("cost", tight_geom, meta_blocks=8, op_ratio=op)
        n_lpn = base_cfg.logical_pages
        n_ops = max(60_000, min(400_000, int(n_lpn * 4.0)))
        wl = WorkloadParams(
            n_lpn=n_lpn, n_ops=n_ops, seed=seed,
            segment_pages=max(200, n_lpn // 24), discard_every=max(2000, n_ops // 8),
            pair_share=0.30,
        )
        for policy in ("greedy", "cost"):
            cfg = make_cfg(policy, tight_geom, meta_blocks=8, op_ratio=op,
                           logical_page_count=n_lpn)
            ftl = FTL(cfg)
            try:
                drive(ftl, FlightLogWorkload(wl))
            except NoSpace:
                pass
            ftl.flush()
            w = ftl.wear_stats()
            rows.append({
                "op_ratio": op,
                "policy": policy,
                "wa": round(ftl.wa(), 3),
                "erases": ftl.stats["erases"],
                "gc_erases": ftl.stats["gc_erases"],
                "copies": ftl.stats["gc_copies"],
                "wear_max": w["max"],
                "wear_min": w["min"],
                "spread": w["spread"],
                "stdev": round(w["stdev"], 2),
                "host_writes": ftl.stats["host_writes"],
            })
            print(f"  OP={op:.0%} {policy:6s} WA={ftl.wa():.3f} "
                  f"erases={ftl.stats['erases']:,} "
                  f"spread={w['spread']} stdev={w['stdev']:.1f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report", default="report/index.html")
    ap.add_argument("--ops", type=int, default=None)
    ap.add_argument("--cuts", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    geometry = QUICK_GEOMETRY if args.quick else MAIN_GEOMETRY
    n_ops = args.ops or (150_000 if args.quick else 3_200_000)
    sample_every = 25_000 if args.quick else 400_000
    cfg0 = make_cfg("cost", geometry)
    wl = WorkloadParams(
        n_lpn=cfg0.logical_pages, n_ops=n_ops, seed=7,
        segment_pages=8192, discard_every=20_000, pair_share=0.30,
    )
    os.makedirs(args.runs_dir, exist_ok=True)
    print(f"geometry: {geometry.describe()}")
    print(f"logical pages: {wl.n_lpn:,}  ops: {wl.n_ops:,}  OP: {cfg0.op_fraction:.1%}")

    results: dict = {
        "geometry": {
            "describe": geometry.describe(),
            "logical_pages": cfg0.logical_pages,
            "logical_mib": cfg0.logical_pages * geometry.page_size / 2**20,
            "op_ratio": cfg0.op_fraction,
            "raw_mib": geometry.capacity_bytes / 2**20,
            "meta_blocks": cfg0.meta_blocks,
            "factory_bad_pct": cfg0.factory_bad_pct,
            "erase_fail_prob": cfg0.erase_fail_prob,
            "pe_cycles": geometry.pe_cycles,
            "page_size": geometry.page_size,
        },
        "workload": asdict(wl),
        "policy": {},
        "block_map": {},
        "valid_dist": {},
        "timeline": {},
        "gc_log": {},
        "policy_config": {
            "GC 策略": "cost = 有效页比例 + 擦除次数；greedy = 只看有效页数",
            "综合权重 gc_wear_weight": cfg0.gc_wear_weight,
            "预留空间比例": f"{cfg0.op_fraction:.1%}",
            "GC 触发水位": f"{cfg0.gc_reserve_pct:.0f}% 空闲超级块",
            "动态磨损均衡": "开启（优先分配擦除次数少的空闲块）",
            "静态磨损均衡": f"开启（极差 > {cfg0.static_wl_min_spread} 且闲置 > {cfg0.static_wl_min_age:,} 页）"
                            if cfg0.static_wl else "关闭",
            "坏块率": f"出厂 {cfg0.factory_bad_pct:.1%}，运行期擦除失败 {cfg0.erase_fail_prob:.1e}",
            "多平面": "开启（同位置双页一次编程）",
            "检查点间隔": f"每 {cfg0.checkpoint_every:,} 主机页",
            "页面大小 / 块内页数": f"{geometry.page_size} B / {geometry.pages_per_block}",
        },
    }

    for policy in ("greedy", "cost"):
        print(f"\n=== {policy} ===")
        run = run_policy(policy, geometry, wl, sample_every=sample_every)
        summ = summarize(run)
        results["policy"][policy] = summ
        results["block_map"][policy] = run.block_map
        results["valid_dist"][policy] = run.valid_dist
        results["gc_log"][policy] = run.gc_recent
        for k, v in run.timeline.items():
            results["timeline"][k] = v
        print(f"writes={summ['host_writes']:,} WA={summ['wa']:.3f} "
              f"erases={summ['erases']:,} gc={summ['gc_erases']:,} "
              f"wl={summ['wl_moves']} bad={summ['runtime_bad']} "
              f"wear_max={summ['wear']['max']} spread={summ['wear']['spread']} "
              f"{summ['elapsed_s']:.1f}s")
        with open(os.path.join(args.runs_dir, f"{policy}.json"), "w") as fh:
            json.dump(summ, fh, indent=1)

    print("\n=== pressure sweep ===")
    results["sweep"] = pressure_sweep(geometry, n_ops)

    print("\n=== power loss ===")
    pl_wl = WorkloadParams(
        n_lpn=cfg0.logical_pages, n_ops=min(n_ops, 120_000), seed=11,
        segment_pages=4096, discard_every=8_000, pair_share=0.30,
    )
    pl = powerloss_sweep(geometry, pl_wl, args.cuts, seed=3)
    results["powerloss"] = pl
    for p, v in pl.items():
        print(f"{p}: {v['cuts'] - v['failures']}/{v['cuts']} ok, "
              f"torn={v['torn_total']} aborts={v['abort_total']} modes={v['recovery_modes']}")

    out = os.path.join(args.runs_dir, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nresults -> {out}")

    from .report import build_report

    build_report(results, args.report)
    print(f"report  -> {os.path.abspath(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
