"""Strict model of a bare NAND die plus deterministic power-loss injection.

Rules enforced by :class:`NandDevice` (violations raise :class:`FlashError`):

* erase is block granular and only legal on a block with no programmed pages;
* a page may be programmed exactly once per erase cycle;
* pages inside a block are programmed in strictly increasing order;
* bad blocks (factory or runtime) reject every program and erase;
* a multi-plane program writes the same page offset in the twin block of the
  neighbouring plane, and the pair commits together unless the operation is
  the one torn by the power cut.

Power loss is modelled at *operation* granularity.  The operation at the
cut point is either aborted (nothing lands), torn (one page of a two-plane
command lands, or a page lands with a damaged spare area so its tag CRC is
bad), or it completes and the cut happens right after it.

Tags live in the spare area::

    u32 seq | u32 lpn | u8 flags | u8 kind | u32 crc32(seq..kind)

Unprogrammed pages read as erased; a page is only readable when its tag CRC
is good, so the FTL recovers purely by scanning.
"""

from __future__ import annotations

import struct
import zlib
from array import array
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import Geometry

TAG = 16

FLAG_NONE = 0
FLAG_DISCARD = 1  # payload is a packed list of discarded LPNs

KIND_DATA = 0
KIND_DISCARD = 1
KIND_META = 2

LPN_NONE = 0xFFFFFFFF


class FlashError(RuntimeError):
    """A rule of the NAND device was violated by the caller."""


class NoSpace(FlashError):
    """The FTL ran out of free blocks."""


class PowerLoss(BaseException):
    """Raised at the injection point; subclasses BaseException on purpose."""

    def __init__(self, ops: int):
        super().__init__(f"power loss at operation {ops}")
        self.ops = ops


@dataclass(frozen=True, slots=True)
class PageTag:
    seq: int
    lpn: int
    flags: int
    kind: int

    @property
    def is_data(self) -> bool:
        return self.kind == KIND_DATA and self.flags == FLAG_NONE and self.lpn != LPN_NONE

    @property
    def is_discard(self) -> bool:
        return self.kind == KIND_DISCARD


def pack_tag(seq: int, lpn: int, flags: int = FLAG_NONE, kind: int = KIND_DATA) -> bytes:
    head = struct.pack("<IIBB", seq, lpn, flags, kind)
    return head + struct.pack("<I", zlib.crc32(head) & 0xFFFFFFFF)


def unpack_tag(buf: bytes) -> PageTag | None:
    if len(buf) < 14:
        return None
    head = buf[:10]
    if zlib.crc32(head) & 0xFFFFFFFF != struct.unpack("<I", buf[10:14])[0]:
        return None
    seq, lpn, flags, kind = struct.unpack("<IIBB", head)
    return PageTag(seq, lpn, flags, kind)


class NandDevice:
    """Media state for one bare die."""

    def __init__(self, geom: Geometry, *, seed: int = 0):
        self.g = geom
        self.seed = seed
        n = geom.pages
        self.page_size = geom.page_size
        self.data = bytearray(n * geom.page_size)
        self.spare = bytearray(n * TAG)  # 0xFF == erased
        self.page_state = bytearray(n)  # 0 = erased, 1 = programmed
        self.next_page = array("H", [0]) * geom.n_blocks  # next free offset
        self.block_state = bytearray(geom.n_blocks)  # 0 clean, 1 has data
        self.erases = [0] * geom.n_blocks

        self.factory_bad: set[int] = set()
        self.bad_blocks: dict[int, dict] = {}
        #: optional callback(kind, lpn, ppn, seq) for every page that landed
        #: with an intact spare area.
        self.commit_hook: Callable[[int, int, int, int], None] | None = None

        # power loss
        self._ops = 0
        self._target: int | None = None
        self._rng = (seed * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF
        self.tears = 0
        self.aborts = 0

    # -- deterministic RNG -------------------------------------------------
    def _rnd(self) -> float:
        x = (self._rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        self._rng = x
        return (x >> 11) / float(1 << 53)

    def _rnd_int(self, n: int) -> int:
        return int(self._rnd() * n) if n > 0 else 0

    def set_power_loss_after(self, ops: int | None, seed: int | None = None) -> None:
        self._target = ops
        self._ops = 0
        if seed is not None:
            self._rng = (seed * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF

    @property
    def ops(self) -> int:
        return self._ops

    # -- bad blocks --------------------------------------------------------
    def add_factory_bad(self, blocks: Iterable[int]) -> None:
        for b in blocks:
            if 0 <= b < self.g.n_blocks:
                self.factory_bad.add(b)
                self.bad_blocks.setdefault(
                    b,
                    {
                        "block": b,
                        "plane": self.g.plane_of(b),
                        "erase_count": 0,
                        "seq": 0,
                        "reason": "factory",
                    },
                )

    def is_bad(self, block: int) -> bool:
        return block in self.factory_bad or block in self.bad_blocks

    def mark_bad(self, block: int, *, seq: int = 0, reason: str = "runtime") -> None:
        if block in self.bad_blocks:
            return
        self.bad_blocks[block] = {
            "block": block,
            "plane": self.g.plane_of(block),
            "erase_count": self.erases[block],
            "seq": seq,
            "reason": reason,
        }
        self.block_state[block] = 1

    # -- reads -------------------------------------------------------------
    def page_programmed(self, ppn: int) -> bool:
        return bool(self.page_state[ppn])

    def read_page(self, ppn: int) -> bytes:
        off = ppn * self.page_size
        return bytes(self.data[off : off + self.page_size])

    def read_tag(self, ppn: int) -> PageTag | None:
        if not self.page_state[ppn]:
            return None
        off = ppn * TAG
        return unpack_tag(bytes(self.spare[off : off + TAG]))

    def programmed_in_block(self, blk: int) -> int:
        return int(self.next_page[blk])

    def block_of(self, ppn: int) -> int:
        return self.g.block_of(ppn)

    # -- power loss scheduling --------------------------------------------
    def _is_loss_op(self) -> bool:
        return self._target is not None and self._ops + 1 >= self._target

    def _finish_loss(self) -> None:
        self._ops += 1
        self._target = None
        raise PowerLoss(self._ops)

    def _take_loss(self) -> None:
        self._finish_loss()

    def _land_page(self, ppn: int, payload: bytes, tag: bytes, *, spare_ok: bool = True) -> None:
        off = ppn * self.page_size
        self.data[off : off + len(payload)] = payload
        toff = ppn * TAG
        if spare_ok:
            self.spare[toff : toff + len(tag)] = tag
        else:
            self.spare[toff : toff + TAG] = bytes(self._rnd_int(256) for _ in range(TAG))
        self.page_state[ppn] = 1
        blk = self.g.block_of(ppn)
        self.next_page[blk] = max(self.next_page[blk], self.g.page_of(ppn) + 1)
        self.block_state[blk] = 1
        if spare_ok and self.commit_hook is not None:
            t = unpack_tag(tag)
            if t is not None:
                self.commit_hook(t.kind, t.lpn, ppn, t.seq)

    def _check_writable(self, ppn: int) -> None:
        g = self.g
        blk = g.block_of(ppn)
        if self.is_bad(blk):
            raise FlashError(f"program to bad block {blk} (ppn {ppn})")
        if self.page_state[ppn]:
            raise FlashError(
                f"page {ppn} (block {blk} offset {g.page_of(ppn)}) programmed twice "
                "without erase"
            )
        if g.page_of(ppn) < self.next_page[blk]:
            raise FlashError(
                f"out-of-order program in block {blk}: offset {g.page_of(ppn)} "
                f"< next {self.next_page[blk]}"
            )

    # -- mutating operations ----------------------------------------------
    def program_pages(self, pages: list[tuple[int, bytes, bytes]], *, seq: int = 0) -> None:
        """One program operation touching one or two planes at one offset."""
        if not pages:
            return
        g = self.g
        if len(pages) > 2:
            raise FlashError("at most 2 planes per program operation")
        offsets = {g.page_of(p) for p, _, _ in pages}
        if len(offsets) != 1:
            raise FlashError("multi-plane program needs one shared page offset")
        if len(pages) == 2:
            b1, b2 = g.block_of(pages[0][0]), g.block_of(pages[1][0])
            if b1 == b2 or g.pair(b1) != g.pair(b2):
                raise FlashError("multi-plane program must address a block pair")
        for ppn, _, _ in pages:
            self._check_writable(ppn)

        if self._is_loss_op():
            mode = self._rnd_int(100)
            if mode < 30:
                self.aborts += 1
                self._take_loss()
            elif len(pages) == 2 and mode < 70:
                ppn, payload, tag = pages[0]
                self._land_page(ppn, payload, tag)
                self.tears += 1
                self._finish_loss()
            else:
                ppn, payload, tag = pages[-1]
                self._land_page(ppn, payload, tag, spare_ok=False)
                self.tears += 1
                self._finish_loss()

        for ppn, payload, tag in pages:
            self._land_page(ppn, payload, tag)
        self._ops += 1

    def erase_block(self, blk: int, *, seq: int = 0) -> None:
        g = self.g
        if not 0 <= blk < g.n_blocks:
            raise FlashError(f"block {blk} out of range")
        if self.is_bad(blk):
            raise FlashError(f"erase of bad block {blk}")

        if self._is_loss_op():
            if self._rnd_int(100) < 40:
                self.aborts += 1
                self._take_loss()
            self._apply_erase(blk)
            self.tears += 1
            self._finish_loss()
        self._apply_erase(blk)
        self._ops += 1

    def _apply_erase(self, blk: int) -> None:
        g = self.g
        base = blk * g.pages_per_block
        end = base + g.pages_per_block
        self.page_state[base:end] = bytes(end - base)
        self.spare[base * TAG : end * TAG] = b"\xff" * ((end - base) * TAG)
        self.next_page[blk] = 0
        self.block_state[blk] = 0
        self.erases[blk] += 1

    # -- maintenance -------------------------------------------------------
    def wipe(self) -> None:
        self.__init__(self.g, seed=self.seed)

    def live_bytes(self) -> int:
        return len(self.data)
