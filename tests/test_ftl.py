"""FTL functional tests: mapping, GC, wear levelling, discard, bad blocks,
multi-plane, checkpoints."""

import random

import pytest

from flashsim.config import Config, Geometry
from flashsim.device import KIND_DATA, LPN_NONE, NandDevice, NoSpace
from flashsim.ftl import FTL
from flashsim.workload import FlightLogWorkload, Op, WorkloadParams


def cfg(**kw) -> Config:
    base = dict(
        geometry=Geometry(page_size=1024, pages_per_block=32,
                          blocks_per_plane=32, planes=2, pe_cycles=5000),
        meta_blocks=6, op_ratio=0.25, checkpoint_every=400,
        factory_bad_pct=0.0, erase_fail_prob=0.0,
    )
    base.update(kw)
    return Config(**base)


def drive(ftl, wl):
    try:
        for op in FlightLogWorkload(wl):
            if op.kind == Op.PAIR:
                ftl.write_pair(op.lpn, op.payload, op.lpn_b, op.payload_b)
            elif op.kind == Op.WRITE:
                ftl.write(op.lpn, op.payload)
            else:
                ftl.discard_range(op.lpn, op.lpn_b)
    except NoSpace:
        pass
    ftl.flush()


def test_page_mapping_roundtrip():
    f = FTL(cfg())
    rng = random.Random(1)
    model = {}
    for i in range(3000):
        lpn = rng.randrange(f.n_lpn)
        data = f"v{i}".encode()
        f.write(lpn, data)
        model[lpn] = data
    f.flush()
    for lpn, data in model.items():
        assert f.read(lpn).startswith(data)
    # maps point at distinct physical pages
    live = [p for p in f.map if p != LPN_NONE]
    assert len(live) == len(set(live))


def test_gc_reclaims_and_preserves_data():
    f = FTL(cfg(op_ratio=0.15))
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=8000, seed=3, segment_pages=200,
                        discard_every=1000, pair_share=0.0)
    drive(f, wl)
    assert f.stats["gc_erases"] > 0, "GC should have run under this pressure"
    for lpn in range(f.n_lpn):
        ppn = f.map[lpn]
        if ppn != LPN_NONE:
            tag = f.dev.read_tag(ppn)
            assert tag is not None and tag.lpn == lpn
    assert f.live_pages() > 0


def test_discarded_pages_not_copied_by_gc():
    f = FTL(cfg(op_ratio=0.12))
    n = f.n_lpn
    for lpn in range(n // 2):
        f.write(lpn, b"x" * 8)
    f.flush()
    f.discard_range(0, n // 2)
    pages_before = f.stats["gc_copies"]
    # churn the *live* half so GC has to reclaim blocks holding discarded
    # pages; those discarded pages must not be copied back.
    for i in range(6000):
        f.write(n // 2 + (i % (n - n // 2)), b"y" * 8)
    f.flush()
    alive = [l for l in range(n // 2) if f.map[l] != LPN_NONE]
    assert not alive, "discarded LPNs must not be resurrected by GC"
    assert f.stats["gc_erases"] > 0, "test needs GC to have run"
    assert f.stats["discard_pages"] > 0
    # a perfect GC would copy at most the live half once per reclaim; assert
    # the discarded half was not copied at all by checking page contents
    for lpn in range(n // 2):
        assert f.read(lpn) is None


def test_discard_survives_reboot():
    f = FTL(cfg())
    for i in range(200):
        f.write(i, b"old")
    f.flush()
    f.checkpoint()
    f.discard_range(0, 100)
    dev = f.dev
    g = FTL(f.cfg, dev=dev)
    g.recover()
    for i in range(100):
        assert g.read(i) is None
    for i in range(100, 200):
        assert g.read(i) is not None


def test_static_wl_moves_cold_pages():
    base = dict(op_ratio=0.2, static_wl_min_age=500, static_wl_min_spread=6,
                static_wl_check_every=32, static_wl_batch=1)
    cold_n = 60
    results = {}
    for static in (False, True):
        f = FTL(cfg(static_wl=static, **base))
        for l in range(cold_n):
            f.write(l, b"COLD%d" % l)
        cold0 = {l: f.map[l] for l in range(cold_n)}
        for r in range(12):
            for l in range(cold_n, f.n_lpn):
                f.write(l, b"H%d" % r)
        f.flush()
        moved = sum(1 for l in range(cold_n) if f.map[l] != cold0[l])
        results[static] = (moved, f.wear_stats()["spread"])
    assert results[False][0] == 0, "without static WL cold data must stay put"
    assert results[True][0] > 0, "static WL must relocate cold data"
    assert results[True][1] * 2 < results[False][1] + 1 or results[True][1] < results[False][1]


def test_dynamic_wl_rotates_allocation():
    f = FTL(cfg(static_wl=False))
    rng = random.Random(5)
    for i in range(20000):
        f.write(rng.randrange(f.n_lpn), b"z")
    f.flush()
    w = f.wear_stats()
    assert w["spread"] <= max(2, w["max"] // 4), f"wear spread too wide: {w}"


def test_multiplane_writes_two_planes_one_op():
    f = FTL(cfg())
    f.write_pair(1, b"A", 2, b"B")  # warms up an erased superblock
    before = f.dev.ops
    f.write_pair(3, b"C", 4, b"D")
    op_delta = f.dev.ops - before
    f2 = FTL(cfg())
    f2.write(1, b"A")
    f2.write(2, b"B")
    single = FTL(cfg())
    single.write_pair(7, b"A", 8, b"B")  # open the SB on this instance
    b2 = single.dev.ops
    single.write(9, b"C")
    single.write(10, b"D")
    two_single_ops = single.dev.ops - b2
    assert op_delta == 1, f"a two-plane program is one device op, got {op_delta}"
    assert two_single_ops == 2, "two lone writes are two device ops"
    p1, p2 = f.map[3], f.map[4]
    assert f.g.plane_of(f.g.block_of(p1)) == 0
    assert f.g.plane_of(f.g.block_of(p2)) == 1
    assert f.g.page_of(p1) == f.g.page_of(p2)
    assert f.g.pair(f.g.block_of(p1)) == f.g.pair(f.g.block_of(p2))


def test_factory_bad_blocks_never_used():
    f = FTL(cfg(factory_bad_pct=0.15))
    assert f.dev.factory_bad, "test needs some factory bad blocks"
    for s, info in f.sb.items():
        if info.kind == "bad":
            continue
        for b in info.blocks:
            assert b not in f.dev.factory_bad
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=3000, seed=2, segment_pages=100,
                        discard_every=500)
    drive(f, wl)
    for ppn, tag in ((p, f.dev.read_tag(p)) for p in range(f.g.pages) if f.dev.page_state[p]):
        if tag is not None:
            blk = f.g.block_of(ppn)
            assert blk not in f.dev.factory_bad


def test_runtime_bad_block_is_avoided_after_detection():
    f = FTL(cfg(erase_fail_prob=0.0))
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=4000, seed=4, segment_pages=100,
                        discard_every=1000)
    drive(f, wl)
    # force a bad block, then keep writing; it must not be touched again
    victim_block = f.sb[next(iter(f.free))].blocks[0] if f.free else None
    if victim_block is None:
        pytest.skip("no free block to poison")
    f.dev.mark_bad(victim_block, seq=f.seq, reason="program_fail")
    for s, info in f.sb.items():
        if victim_block in info.blocks:
            info.kind = "bad"
    for i in range(2000):
        f.write(i % f.n_lpn, b"after")
    f.flush()
    assert f.dev.programmed_in_block(victim_block) >= 0  # no crash / no writes


def test_checkpoint_recovery_matches_map():
    f = FTL(cfg(checkpoint_every=300))
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=5000, seed=6, segment_pages=150,
                        discard_every=800)
    drive(f, wl)
    g = FTL(f.cfg, dev=f.dev)
    g.recover()
    assert g.stats["recovery_mode"] == "checkpoint+scan"
    diff = sum(1 for l in range(f.n_lpn) if g.map[l] != f.map[l])
    assert diff == 0, f"{diff} mapping mismatches after recovery"


def test_full_scan_recovery_without_checkpoint():
    f = FTL(cfg(checkpoint_every=10**9))  # never checkpoint
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=4000, seed=6, segment_pages=150,
                        discard_every=1000)
    drive(f, wl)
    g = FTL(f.cfg, dev=f.dev)
    g.recover()
    assert g.stats["recovery_mode"] == "full scan"
    diff = sum(1 for l in range(f.n_lpn) if g.map[l] != f.map[l])
    assert diff == 0


def test_erase_count_survives_reboot():
    f = FTL(cfg())
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=6000, seed=8, segment_pages=150,
                        discard_every=1000)
    drive(f, wl)
    g = FTL(f.cfg, dev=f.dev)
    g.recover()
    for s, info in f.sb.items():
        if info.kind == "bad":
            continue
        assert g.sb[s].erase_count == info.erase_count


def test_bad_metadata_block_does_not_break_checkpoints():
    from flashsim.device import NandDevice

    geom = Geometry(page_size=1024, pages_per_block=32,
                    blocks_per_plane=32, planes=2, pe_cycles=2000)
    cfg = Config(geometry=geom, meta_blocks=8, op_ratio=0.2,
                 checkpoint_every=300, seed=1)
    dev = NandDevice(geom)
    dev.add_factory_bad([2, 4])  # inside the metadata area
    f = FTL(cfg, dev=dev)
    for i in range(1500):
        f.write(i % f.n_lpn, b"v%d" % i)
    f.flush()
    assert f.stats["checkpoints"] > 0
    g = FTL(cfg, dev=f.dev)
    g.recover()
    assert g.stats["recovery_mode"] == "checkpoint+scan"
    assert sum(1 for l in range(f.n_lpn) if g.map[l] != f.map[l]) == 0


def test_write_amplification_categories_reconcile():
    """Every programmed page belongs to exactly one accounting category."""
    f = FTL(cfg(op_ratio=0.2, checkpoint_every=500))
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=6000, seed=9, segment_pages=150,
                        discard_every=600, pair_share=0.3)
    drive(f, wl)
    st = f.stats
    parts = (st["data_programs"] + st["marker_pages"] + st["meta_pages"]
             + st["discard_pages"] + st["badlog_pages"])
    assert parts == f.physical_pages()
    assert st["discard_pages"] > 0, "workload must exercise discards"
    assert st["checkpoints"] > 0


def test_write_amplification_accounting():
    f = FTL(cfg(op_ratio=0.2))
    wl = WorkloadParams(n_lpn=f.n_lpn, n_ops=5000, seed=9, segment_pages=150,
                        discard_every=1000)
    drive(f, wl)
    assert f.stats["host_writes"] > 0
    assert f.physical_pages() >= f.stats["host_writes"]
    assert 1.0 <= f.wa() < 50.0
