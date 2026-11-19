"""Device rule enforcement: the simulator must refuse illegal NAND ops."""

import pytest

from flashsim.config import Geometry
from flashsim.device import (
    KIND_DATA,
    FlashError,
    NandDevice,
    PowerLoss,
    pack_tag,
)

G = Geometry(page_size=512, pages_per_block=16, blocks_per_plane=8, planes=2, pe_cycles=100)


def fresh():
    d = NandDevice(G)
    for b in range(G.n_blocks):
        d.erase_block(b)
    return d


def prog(d, block, page, lpn=1, seq=1):
    ppn = G.ppn(block, page)
    d.program_pages([(ppn, b"x" * 16, pack_tag(seq, lpn))], seq=seq)


def test_page_cannot_be_programmed_twice_without_erase():
    d = fresh()
    prog(d, 2, 1)
    with pytest.raises(FlashError, match="programmed twice"):
        prog(d, 2, 1)


def test_out_of_order_program_rejected():
    d = fresh()
    prog(d, 2, 3)
    with pytest.raises(FlashError, match="out-of-order"):
        prog(d, 2, 2)


def test_hole_is_legal_forward_skip_but_not_refill():
    d = fresh()
    prog(d, 2, 1)
    prog(d, 2, 5)          # skip is fine
    with pytest.raises(FlashError, match="out-of-order"):
        prog(d, 2, 3)      # refilling the hole is not


def test_bad_block_rejects_program_and_erase():
    d = fresh()
    d.add_factory_bad([4])
    with pytest.raises(FlashError, match="bad block"):
        prog(d, 4, 1)
    with pytest.raises(FlashError, match="bad block"):
        d.erase_block(4)


def test_torn_multiplane_leaves_first_page_only():
    d = fresh()
    d.set_power_loss_after(1, seed=1234)
    # search a seed/mode that tears (first lands, twin does not)
    torn = False
    for seed in range(1, 40):
        d = fresh()
        d.set_power_loss_after(1, seed=seed)
        try:
            d.program_pages(
                [(G.ppn(2, 1), b"a" * 16, pack_tag(1, 1)),
                 (G.ppn(3, 1), b"b" * 16, pack_tag(2, 2))],
                seq=1,
            )
        except PowerLoss:
            pass
        if d.page_state[G.ppn(2, 1)] and not d.page_state[G.ppn(3, 1)]:
            torn = True
            assert d.read_tag(G.ppn(2, 1)).lpn == 1
            break
    assert torn, "expected at least one seed to tear a two-plane program"


def test_torn_project_can_corrupt_spare_crc():
    d = fresh()
    hit = False
    for seed in range(1, 60):
        d = fresh()
        d.set_power_loss_after(1, seed=seed)
        try:
            d.program_pages([(G.ppn(2, 1), b"a" * 16, pack_tag(1, 1))], seq=1)
        except PowerLoss:
            pass
        if d.page_state[G.ppn(2, 1)] and d.read_tag(G.ppn(2, 1)) is None:
            hit = True
            break
    assert hit, "expected some seed to land a page with a broken tag CRC"


def test_multiplane_requires_same_offset_and_twin_pair():
    d = fresh()
    with pytest.raises(FlashError, match="shared page offset"):
        d.program_pages([(G.ppn(2, 1), b"a", pack_tag(1, 1)),
                         (G.ppn(3, 2), b"b", pack_tag(2, 2))], seq=1)
    d2 = fresh()
    with pytest.raises(FlashError, match="block pair"):
        d2.program_pages([(G.ppn(2, 1), b"a", pack_tag(1, 1)),
                          (G.ppn(5, 1), b"b", pack_tag(2, 2))], seq=1)


def test_erase_clears_pages_and_increments_count():
    d = fresh()
    prog(d, 2, 1)
    before = d.erases[2]
    d.erase_block(2)
    assert d.erases[2] == before + 1
    assert not d.page_state[G.ppn(2, 1)]
    assert d.read_tag(G.ppn(2, 1)) is None
    prog(d, 2, 1, lpn=9, seq=9)  # writable again after erase


def test_tag_crc_detects_flip():
    from flashsim.device import TAG, unpack_tag

    d = fresh()
    prog(d, 2, 1, lpn=7, seq=3)
    ppn = G.ppn(2, 1)
    assert d.read_tag(ppn).lpn == 7
    d.spare[ppn * TAG] ^= 0xFF
    assert d.read_tag(ppn) is None


def test_bad_block_read_still_works_but_writes_refused():
    d = fresh()
    prog(d, 6, 1)
    d.mark_bad(6, seq=5, reason="erase_fail")
    assert d.read_tag(G.ppn(6, 1)).lpn == 1  # data still readable
    with pytest.raises(FlashError):
        prog(d, 6, 2)
