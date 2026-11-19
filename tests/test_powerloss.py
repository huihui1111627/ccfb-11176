"""Power-loss consistency: cut power during a real workload, reboot, and check
every logical page against the acknowledged state.

These are the acceptance tests for the recovery contract.  A cut may land on
any device operation; the reboot must return the last acknowledged payload
for every page, and an in-flight operation may legally be old / new / gone.
"""

import random

import pytest

from flashsim.config import Config, Geometry
from flashsim.device import NandDevice, NoSpace, PowerLoss
from flashsim.experiment import QUICK_GEOMETRY, make_cfg
from flashsim.ftl import FTL
from flashsim.powerloss import Shadow, check_recovery
from flashsim.workload import FlightLogWorkload, Op, WorkloadParams


def _drive(ftl: FTL, shadow: Shadow, wl: WorkloadParams) -> None:
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


def _small_geometry() -> Geometry:
    return Geometry(page_size=1024, pages_per_block=32, blocks_per_plane=40,
                    planes=2, pe_cycles=400)


def _cfg(policy="cost", **kw):
    base = dict(
        geometry=_small_geometry(),
        meta_blocks=6,
        op_ratio=0.15,
        gc_policy=policy,
        checkpoint_every=400,
        factory_bad_pct=0.01,
        erase_fail_prob=1e-4,
        static_wl=True,
        static_wl_min_age=2000,
        static_wl_check_every=32,
    )
    base.update(kw)
    return Config(**base)


def _wl(cfg: Config, n_ops=4000, seed=5):
    return WorkloadParams(
        n_lpn=cfg.logical_pages, n_ops=n_ops, seed=seed,
        segment_pages=150, discard_every=700, pair_share=0.4,
    )


@pytest.fixture(scope="module")
def clean_ops():
    cfg = _cfg()
    wl = _wl(cfg, n_ops=4000)
    ftl = FTL(cfg)
    try:
        _drive(ftl, Shadow(cfg.logical_pages), wl)
    except NoSpace:
        pass
    return max(1, ftl.dev.ops)


def test_clean_reboot_preserves_everything():
    cfg = _cfg()
    wl = _wl(cfg, n_ops=4000)
    ftl = FTL(cfg)
    shadow = Shadow(cfg.logical_pages)
    try:
        _drive(ftl, shadow, wl)
    except NoSpace:
        pass
    boot = FTL(cfg, dev=ftl.dev)
    boot.recover()
    res = check_recovery(shadow, boot)
    assert res.ok, res.summary()
    assert res.checked > 100


@pytest.mark.parametrize("policy", ["greedy", "cost"])
def test_power_cut_at_many_points(policy, clean_ops):
    rng = random.Random(hash(policy) % 1000 + 7)
    cfg = _cfg(policy)
    wl = _wl(cfg, n_ops=4000, seed=5)
    failures = []
    for i in range(25):
        point = rng.randrange(1, clean_ops)
        dev = NandDevice(cfg.geometry, seed=cfg.seed)
        ftl = FTL(cfg, dev=dev)
        shadow = Shadow(cfg.logical_pages)
        dev.set_power_loss_after(point, seed=rng.randrange(1 << 30))
        try:
            _drive(ftl, shadow, wl)
        except (PowerLoss, NoSpace):
            pass
        boot = FTL(cfg, dev=dev)
        boot.recover()
        res = check_recovery(shadow, boot)
        if not res.ok:
            failures.append((point, res.summary()))
    assert not failures, f"{len(failures)}/25 cuts failed: {failures[:2]}"


def test_power_cut_during_gc_copy():
    """Force the cut exactly on GC relocation pages, where a copy is torn."""
    cfg = _cfg("greedy", op_ratio=0.12)
    wl = _wl(cfg, n_ops=6000, seed=13)
    # run once to find ops that belong to GC; then cut just before them
    probe = FTL(cfg)
    shadow = Shadow(cfg.logical_pages)
    gc_ops = []
    orig = probe._relocate

    def watching(victim, *, reason):
        gc_ops.append(probe.dev.ops)
        return orig(victim, reason=reason)

    probe._relocate = watching
    try:
        _drive(probe, shadow, wl)
    except NoSpace:
        pass
    assert gc_ops, "workload must trigger GC"
    points = gc_ops[:: max(1, len(gc_ops) // 10)][:10]
    for p in points:
        for delta in (0, 1, 2):
            dev = NandDevice(cfg.geometry, seed=cfg.seed)
            ftl = FTL(cfg, dev=dev)
            sh = Shadow(cfg.logical_pages)
            dev.set_power_loss_after(max(1, p + delta), seed=p + delta + 1)
            try:
                _drive(ftl, sh, wl)
            except (PowerLoss, NoSpace):
                pass
            boot = FTL(cfg, dev=dev)
            boot.recover()
            res = check_recovery(sh, boot)
            assert res.ok, f"cut at GC op {p}+{delta}: {res.summary()}"


def test_power_cut_at_checkpoint_rotation():
    """Cut inside checkpoint writes; either snapshot must remain loadable."""
    cfg = _cfg("cost", checkpoint_every=200)
    wl = _wl(cfg, n_ops=3000, seed=17)
    probe = FTL(cfg)
    ck_ops = []
    orig = probe.checkpoint

    def watching(*, force=False):
        ck_ops.append(probe.dev.ops)
        return orig(force=force)

    probe.checkpoint = watching
    try:
        _drive(probe, Shadow(cfg.logical_pages), wl)
    except NoSpace:
        pass
    assert ck_ops
    for p in ck_ops[:4]:
        for delta in range(0, 6):
            dev = NandDevice(cfg.geometry, seed=cfg.seed)
            ftl = FTL(cfg, dev=dev)
            sh = Shadow(cfg.logical_pages)
            dev.set_power_loss_after(max(1, p + delta), seed=p + delta + 3)
            try:
                _drive(ftl, sh, wl)
            except (PowerLoss, NoSpace):
                pass
            boot = FTL(cfg, dev=dev)
            boot.recover()
            res = check_recovery(sh, boot)
            assert res.ok, f"cut in checkpoint at {p}+{delta}: {res.summary()}"


def test_discard_records_survive_bank_rotation():
    """Regression: recycling a metadata bank must not drop live discard
    records; the FTL snapshots first, folding them into the map."""
    import random as _random

    cfg = _cfg("cost", checkpoint_every=10**9, op_ratio=0.25)
    f = FTL(cfg)
    shadow = Shadow(cfg.logical_pages)
    rng = _random.Random(1)
    n = f.n_lpn
    try:
        for i in range(20_000):
            if i % 3 == 0:
                lpn = rng.randrange(n)
                f.write(lpn, b"w%d" % i)
                shadow.ack_write(lpn, b"w%d" % i)
            else:
                lpn = rng.randrange(n)
                if f.map[lpn] != 0xFFFFFFFF:
                    f.discard(lpn)
                    shadow.ack_discard(lpn)
    except NoSpace:
        pass
    g = FTL(cfg, dev=f.dev)
    g.recover()
    res = check_recovery(shadow, g)
    assert res.ok, res.summary()


def test_discard_after_reboot_does_not_overwrite_bank():
    """Regression: recovery must resume *after* discard records that were
    written after the snapshot's commit page, not right after the commit."""
    cfg = _cfg("cost", checkpoint_every=200)
    f = FTL(cfg)
    for i in range(300):
        f.write(i, b"v%d" % i)
    f.flush()
    f.discard_range(0, 50)  # records land after the last commit page
    g = FTL(cfg, dev=f.dev)
    g.recover()
    g.discard(200)          # must not hit an already-programmed bank page
    g.write(201, b"after")
    h = FTL(cfg, dev=g.dev)
    h.recover()
    assert h.read(0) is None
    assert h.read(200) is None
    assert h.read(201).startswith(b"after")


def test_sequence_counter_restored_above_flash():
    cfg = _cfg()
    wl = _wl(cfg, n_ops=2000)
    ftl = FTL(cfg)
    try:
        _drive(ftl, Shadow(cfg.logical_pages), wl)
    except NoSpace:
        pass
    boot = FTL(cfg, dev=ftl.dev)
    boot.recover()
    highest = 0
    for ppn in range(ftl.g.pages):
        if ftl.dev.page_state[ppn]:
            tag = ftl.dev.read_tag(ppn)
            if tag is not None:
                highest = max(highest, tag.seq)
    assert boot.seq > highest, "new writes must supersede every page on flash"
