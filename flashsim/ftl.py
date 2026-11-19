"""Page-mapped flash translation layer over :class:`NandDevice`.

Topology
--------
The user area is divided into **superblocks**: superblock ``s`` owns the twin
blocks ``(meta + 2s, meta + 2s + 1)``, one per plane.  A multi-plane program
writes the same page offset in both blocks as one NAND operation:

* :meth:`FTL.write_pair` programs two host pages this way;
* GC copies two valid pages this way when both are ready.

Blocks ``[0, meta_blocks)`` are the **metadata area**:

* blocks 0 and 1: mirrored bad-block log (append-only);
* blocks 2 .. meta_blocks-1: mapping-table snapshots and discard records.

Page offset 0 of every user block holds its **erase marker** (block id, erase
count, generation).  Offsets ``1 .. pages_per_block-1`` hold data.  Erase
counts therefore survive a reboot, which is what the bad-block / wear-out
logic needs.

Host writes
-----------
``lpn -> ppn`` is a flat list in RAM.  A returned write is on flash.  A lone
host write is a single-plane program; it leaves a legal hole in the twin plane
that is never refilled because pages are programmed in increasing offsets.

Durability / power loss
-----------------------
Correctness never depends on a clean shutdown:

* every data page carries ``(lpn, seq)`` in its spare area with a CRC;
* GC copies valid pages with a higher ``seq`` **before** the victim is erased,
  so the copy supersedes the original and a crash in between leaves the
  original as a correct fallback;
* discard notifications are appended to the metadata area;
* the full mapping table is periodically snapshotted into the reserved area.

On boot the FTL reads erase markers, replays the bad-block log, loads the
newest *committed* snapshot (commit page CRC over all snapshot pages), scans
pages and discard records newer than that snapshot, and rebuilds the block
state.  If the newest snapshot is torn it falls back to the previous one, and
with no usable snapshot it rebuilds the whole map by scanning tags.  The
sequence counter is restored above the highest seq found on flash, so new
writes always supersede old versions.

GC policies
-----------
``greedy`` : fewest valid pages (wear-blind control arm).
``cost``   : maximise ``dead_fraction / (1 + wear_weight * relative_erases)``
             (cost-benefit: garbage amount per unit of wear; the default).
A superblock with zero valid pages is reclaimed without copying.

Wear levelling
--------------
* dynamic: the free pool is ordered by erase count, so allocations rotate;
* static: when the erase-count spread across closed superblocks exceeds the
  threshold, old cold superblocks (low erase count, data not touched for
  ``static_wl_min_age`` pages) are relocated, so cold data cannot pin a block.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass

from .config import Config, Geometry
from .device import (
    FLAG_DISCARD,
    FLAG_NONE,
    KIND_DATA,
    KIND_DISCARD,
    KIND_META,
    LPN_NONE,
    NandDevice,
    NoSpace,
    pack_tag,
)

# -- on-flash metadata formats -------------------------------------------

EMARK = struct.Struct("<4sIII")  # magic, block, erase_count, generation
EMARK_MAGIC = b"FLEM"

CKPT_PAGE = struct.Struct("<4sIII")  # magic, gen, page_index, crc(data)
CKPT_MAGIC = b"FCKP"
COMMIT = struct.Struct("<4sIIII")  # magic, gen, n_pages, seq, crc(all data)
COMMIT_MAGIC = b"FCMT"
ENT = struct.Struct("<II")  # lpn, ppn
CKGEN = struct.Struct("<4sI")  # commitment marker: magic, gen (seq before ENT)
CKGEN_MAGIC = b"FCGN"

DISC_HDR = struct.Struct("<4sI")  # magic, count
DISC_MAGIC = b"FDSC"

BAD = struct.Struct("<4sIIB")  # magic, block, erase_count, reason
BAD_MAGIC = b"FBBL"
BAD_REASON = {0: "erase_fail", 1: "program_fail", 2: "runtime"}


@dataclass
class SBInfo:
    """One superblock = twin blocks across the planes."""

    index: int
    blocks: tuple[int, ...]
    kind: str = "free"  # free | active | closed | bad | retired
    erase_count: int = 0
    next_off: tuple[int, ...] = ()
    valid: int = 0
    last_write_seq: int = 0

    @property
    def programmed(self) -> int:
        return max(self.next_off) if self.next_off else 0


@dataclass
class GCRecord:
    seq: int
    sb: int
    policy: str
    valid_before: int
    valid_copied: int
    erase_count: int
    scanned: int
    reason: str  # space | static_wl


class FTL:
    def __init__(self, cfg: Config, dev: NandDevice | None = None):
        self.cfg = cfg
        self.g: Geometry = cfg.geometry
        self.dev = dev if dev is not None else NandDevice(cfg.geometry, seed=cfg.seed)
        self.n_lpn = cfg.logical_pages
        self.map: list[int] = [LPN_NONE] * self.n_lpn
        self.seq = 1
        self.discarded: set[int] = set()

        self.stats = {
            "host_writes": 0,
            "host_bytes": 0,
            "data_programs": 0,
            "multipage_ops": 0,
            "gc_copies": 0,
            "gc_erases": 0,
            "gc_invocations": 0,
            "gc_scanned": 0,
            "erases": 0,
            "checkpoints": 0,
            "ckpt_pages": 0,
            "marker_pages": 0,
            "meta_pages": 0,
            "badlog_pages": 0,
            "discard_ops": 0,
            "discard_pages": 0,
            "runtime_bad": 0,
            "retired_wear": 0,
            "wl_moves": 0,
            "wl_copies": 0,
            "recovery_mode": "",
            "recovery_scanned": 0,
            "recovery_ckpt_gen": -1,
            "power_losses": 0,
            "max_erases": 0,
        }
        self.gc_log: list[GCRecord] = []
        self.sb: dict[int, SBInfo] = {}
        self._init_superblocks()
        self.free: list[int] = [s for s in self.sb if self.sb[s].kind == "free"]
        self.active: int | None = None
        self._wb: list[tuple[int, bytes]] = []
        self._sp = 0  # lone-write plane round-robin
        self._badlog_cursor = 0
        self._erases_since_wl = 0
        self._ckpt_seq = 0
        self._ckpt_gen = 0
        self._cur_bank = 0
        self._bank_index = 0
        self._in_gc = False

    # ==================================================================
    # construction
    # ==================================================================
    def _init_superblocks(self) -> None:
        g, cfg = self.g, self.cfg
        rng = random.Random(cfg.seed * 7919 + 11)
        assert (g.n_blocks - cfg.meta_blocks) % g.planes == 0
        n_sb = (g.n_blocks - cfg.meta_blocks) // g.planes
        pool = list(range(cfg.meta_blocks, g.n_blocks))
        n_bad = round(len(pool) * cfg.factory_bad_pct)
        factory = set(rng.sample(pool, n_bad)) if n_bad else set()
        self.dev.add_factory_bad(factory)
        for s in range(n_sb):
            start = cfg.meta_blocks + s * g.planes
            blocks = tuple(range(start, start + g.planes))
            bad = any(b in factory for b in blocks)
            self.sb[s] = SBInfo(
                index=s,
                blocks=blocks,
                kind="bad" if bad else "free",
                next_off=tuple(0 for _ in blocks),
            )
        self.factory_bad = set(factory)

    def _slots(self) -> int:
        return self.g.pages_per_block - 1

    def _size_sb(self) -> int:
        return self._slots() * self.g.planes

    def free_sb(self) -> int:
        return len(self.free)

    def _sb_of_ppn(self, ppn: int) -> int:
        return (self.g.block_of(ppn) - self.cfg.meta_blocks) // self.g.planes

    # ==================================================================
    # erase / allocation
    # ==================================================================
    def _take_free(self) -> int:
        if not self.free:
            if self._in_gc:
                # GC needs a destination while copying.  Only a garbage-only
                # superblock can be given up without copying; anything else
                # means the OP budget is genuinely exhausted.
                s = self._reclaim_garbage_only()
                if s is None:
                    raise NoSpace("free pool exhausted during GC")
            else:
                self.gc(reason="space")
                if not self.free:
                    raise NoSpace("free pool exhausted after GC")
        if self.cfg.dynamic_wl:
            self.free.sort(key=lambda s: (self.sb[s].erase_count, s))
        s = self.free.pop(0)
        self.sb[s].kind = "active"
        return s

    def _reclaim_garbage_only(self) -> int | None:
        cands = [i for i in self._closed_candidates() if i.valid == 0]
        if not cands:
            return None
        victim = min(cands, key=lambda i: (i.erase_count, i.index))
        victim.kind = "free"
        victim.valid = 0
        victim.next_off = tuple(0 for _ in victim.blocks)
        self.free.append(victim.index)
        self.stats["gc_erases"] += 1
        self.gc_log.append(
            GCRecord(seq=self.seq, sb=victim.index, policy=self.cfg.gc_policy,
                     valid_before=0, valid_copied=0, erase_count=victim.erase_count,
                     scanned=1, reason="space")
        )
        self.free.sort(key=lambda s: (self.sb[s].erase_count, s))
        return self.free.pop(0)

    def _fresh_erased(self, info: SBInfo) -> bool:
        """True when every block was erased and carries a valid marker."""
        g = self.g
        for b in info.blocks:
            if self.dev.is_bad(b) or self.dev.programmed_in_block(b) != 1:
                return False
            ppn = g.ppn(b, 0)
            if self.dev.read_tag(ppn) is None:
                return False
            payload = self.dev.read_page(ppn)
            if payload[:4] != EMARK_MAGIC:
                return False
            _, blk, ec, generation = EMARK.unpack(payload[: EMARK.size])
            if blk != b or ec != self.dev.erases[b]:
                return False
        return True

    def _erase_sb(self, s: int) -> bool:
        """Make superblock ``s`` writable.  False if it cannot be used."""
        info = self.sb[s]
        if info.erase_count >= self.g.pe_cycles:
            info.kind = "retired"
            return False
        if self._fresh_erased(info):
            info.next_off = tuple(1 for _ in info.blocks)
            info.valid = 0
            info.kind = "active"
            return True
        for b in info.blocks:
            if self.dev.is_bad(b):
                info.kind = "bad"
                return False
            if rng_hit(self.cfg.erase_fail_prob, self.seq, b):
                self._mark_bad(b, reason="erase_fail")
                info.kind = "bad"
                return False
            self.dev.erase_block(b, seq=self.seq)
            self.stats["erases"] += 1
            self._erases_since_wl += 1
            ec = self.dev.erases[b]
            info.erase_count = max(info.erase_count, ec)
            self._write_marker(b, ec)
        info.next_off = tuple(1 for _ in info.blocks)
        info.valid = 0
        self.stats["max_erases"] = max(self.stats["max_erases"], info.erase_count)
        if info.erase_count >= self.g.pe_cycles:
            info.kind = "retired"
            self.stats["retired_wear"] += 1
            return False
        info.kind = "active"
        return True

    def _write_marker(self, block: int, erase_count: int) -> None:
        g = self.g
        payload = EMARK.pack(EMARK_MAGIC, block, erase_count, erase_count)
        tag = pack_tag(self.seq, LPN_NONE, FLAG_NONE, KIND_META)
        self.dev.program_pages([(g.ppn(block, 0), payload.ljust(g.page_size, b"\xff"), tag)], seq=self.seq)
        self.seq += 1
        self.stats["marker_pages"] += 1

    def _open_sb(self) -> int:
        while True:
            s = self._take_free()
            if self._erase_sb(s):
                self.active = s
                return s

    def _active_sb(self) -> int:
        if self.active is None:
            return self._open_sb()
        info = self.sb[self.active]
        if info.kind == "active" and info.programmed < self.g.pages_per_block:
            return self.active
        info.kind = "closed"
        self.active = None
        return self._open_sb()

    # ==================================================================
    # programming
    # ==================================================================
    def _stage(self, lpn: int, payload: bytes) -> None:
        """Buffer a GC copy so two copies share one multi-plane program."""
        self._wb.append((lpn, payload))
        if len(self._wb) == 2:
            a, b = self._wb
            self._wb.clear()
            self._program((a, b), gc=True)

    def flush(self) -> None:
        if self._wb:
            items = tuple(self._wb)
            self._wb.clear()
            self._program(items, gc=True)

    def _program(self, items: tuple[tuple[int, bytes], ...], *, gc: bool = False) -> None:
        """Program one or two LPNs; a pair goes multi-plane at one offset."""
        g = self.g
        if len(items) > g.planes:
            for it in items:
                self._program((it,), gc=gc)
            return
        s = self._active_sb()
        info = self.sb[s]
        if len(items) == 1:
            plane = self._sp & (g.planes - 1)
            self._sp += 1
            if info.next_off[plane] >= g.pages_per_block:
                others = [i for i, n in enumerate(info.next_off) if n < g.pages_per_block]
                if not others:
                    info.kind = "closed"
                    self.active = None
                    return self._program(items, gc=gc)
                plane = others[0]
            off = info.next_off[plane]
            targets = [(plane, items[0][0], g.ppn(info.blocks[plane], off))]
        else:
            off = max(info.next_off)
            if off >= g.pages_per_block:
                info.kind = "closed"
                self.active = None
                return self._program(items, gc=gc)
            targets = [(i, lpn, g.ppn(info.blocks[i], off)) for i, (lpn, _) in enumerate(items)]
        pages = []
        for i, lpn, ppn in targets:
            payload = items[0][1] if len(items) == 1 else items[i][1]
            pages.append(
                (ppn, payload.ljust(g.page_size, b"\xff"),
                 pack_tag(self.seq, lpn, FLAG_NONE, KIND_DATA))
            )
            self.seq += 1
        self.dev.program_pages(pages, seq=self.seq)
        if len(pages) == 2:
            self.stats["multipage_ops"] += 1
        for i, lpn, ppn in targets:
            old = self.map[lpn]
            if old != LPN_NONE:
                self.sb[self._sb_of_ppn(old)].valid -= 1
            self.map[lpn] = ppn
            info.valid += 1
            no = list(info.next_off)
            no[i] = off + 1
            info.next_off = tuple(no)
            self.stats["data_programs"] += 1
        info.last_write_seq = self.seq

    # ==================================================================
    # host API
    # ==================================================================
    def write(self, lpn: int, payload: bytes) -> None:
        """Write one logical page; returns once it is on flash."""
        if not 0 <= lpn < self.n_lpn:
            raise ValueError(f"lpn {lpn} out of range")
        if len(payload) > self.g.page_size:
            raise ValueError("payload larger than page")
        self.discarded.discard(lpn)
        self.stats["host_writes"] += 1
        self.stats["host_bytes"] += len(payload)
        self._program(((lpn, payload),))
        self._after_host_write()

    def write_pair(self, lpn_a: int, data_a: bytes, lpn_b: int, data_b: bytes) -> None:
        """Program two LPNs into the twin planes in one NAND operation."""
        if lpn_a == lpn_b:
            raise ValueError("a multi-plane pair needs two distinct LPNs")
        if not self.cfg.multi_plane:
            self.write(lpn_a, data_a)
            self.write(lpn_b, data_b)
            return
        if self._wb:
            self.flush()
        self.discarded.discard(lpn_a)
        self.discarded.discard(lpn_b)
        self.stats["host_writes"] += 2
        self.stats["host_bytes"] += len(data_a) + len(data_b)
        self._program(((lpn_a, data_a), (lpn_b, data_b)))
        self._after_host_write()

    def read(self, lpn: int) -> bytes | None:
        ppn = self.map[lpn] if 0 <= lpn < self.n_lpn else LPN_NONE
        if ppn == LPN_NONE or not self.dev.page_programmed(ppn):
            return None
        return self.dev.read_page(ppn)

    # -- discard -----------------------------------------------------------
    DISC_CAP = (4096 - DISC_HDR.size) // 4

    def discard(self, lpns) -> None:
        """Discard one LPN or a range of them; durable when this returns.

        Discard records are written to the metadata area *before* the RAM map
        drops the pages, so a power cut cannot resurrect a discarded page from
        a stale data page.  One record page holds up to ``DISC_CAP`` LPNs, so a
        whole log segment costs a handful of metadata pages.
        """
        if isinstance(lpns, int):
            lpns = (lpns,)
        live = [l for l in lpns if 0 <= l < self.n_lpn and self.map[l] != LPN_NONE]
        if not live:
            return
        chunks = [live[i : i + self.DISC_CAP] for i in range(0, len(live), self.DISC_CAP)]
        self._ensure_discard_room(len(chunks))
        for chunk in chunks:
            body = DISC_HDR.pack(DISC_MAGIC, len(chunk)) + struct.pack(
                f"<{len(chunk)}I", *chunk
            )
            self._write_bank(self._cur_bank, self._bank_index, body,
                             flags=FLAG_DISCARD, kind=KIND_DISCARD)
            self._bank_index += 1
        self.stats["discard_records"] = self.stats.get("discard_records", 0) + len(chunks)
        for l in live:
            old = self.map[l]
            self.sb[self._sb_of_ppn(old)].valid -= 1
            self.map[l] = LPN_NONE
            self.discarded.add(l)
            self.stats["discard_ops"] += 1

    def discard_range(self, lpn: int, count: int) -> None:
        self.discard(range(lpn, min(self.n_lpn, lpn + count)))

    def _after_host_write(self) -> None:
        if self.stats["host_writes"] % self.cfg.checkpoint_every == 0:
            self.checkpoint()
        if self.cfg.static_wl and self._erases_since_wl >= self.cfg.static_wl_check_every:
            self._erases_since_wl = 0
            self._maybe_static_wl()
        if self.free_sb() <= self._reserve_sb():
            self.gc(reason="space")

    def _reserve_sb(self) -> int:
        return max(2, round(self.cfg.gc_reserve_pct / 100 * len(self.sb)))

    # ==================================================================
    # metadata area: banks
    # ==================================================================
    # The snapshot / discard area is split into *banks*.  A bank is a group of
    # whole superblocks (twin blocks across planes) written sequentially, so
    # a bank must be large enough for one full mapping snapshot.  The bank
    # count is what is left of the metadata blocks after the bad-block log.
    #
    # A new checkpoint goes to the bank that does *not* hold the newest one;
    # that bank is erased first.  The newest valid snapshot is therefore never
    # erased, so a cut during rotation falls back to the previous snapshot.
    # Discard records are appended to the same bank as the snapshot they
    # follow, which bounds the bank space a recovery pass must replay.

    def _bank_group_sbs(self) -> int:
        """Superblocks per bank, chosen so one snapshot fits with headroom."""
        snap_pages = self._snapshot_pages() + 1
        per_sb = self._size_sb()
        need_sbs = (snap_pages + per_sb - 1) // per_sb + 1  # +1 for discards
        avail = (self.cfg.meta_blocks - 2) // self.g.planes
        if avail < 2:
            raise ValueError(
                f"meta_blocks={self.cfg.meta_blocks} too small for a "
                f"{self.n_lpn:,}-page snapshot; need >= "
                f"{2 * need_sbs * self.g.planes + 2} blocks"
            )
        return min(need_sbs, max(1, avail // 2))

    def _bank_sbs(self, bank: int) -> list[int]:
        """Metadata superblocks belonging to ``bank`` (indices over meta area)."""
        avail = (self.cfg.meta_blocks - 2) // self.g.planes
        group = self._bank_group_sbs()
        start = (bank * group) % avail
        return [(start + i) % avail for i in range(group)]

    def _bank_count(self) -> int:
        avail = (self.cfg.meta_blocks - 2) // self.g.planes
        group = self._bank_group_sbs()
        return max(1, avail // group)

    def _bank_blocks(self, bank: int) -> tuple[int, ...]:
        blocks: list[int] = []
        for gsb in self._bank_sbs(bank):
            base = 2 + gsb * self.g.planes
            blocks.extend(range(base, base + self.g.planes))
        return tuple(blocks)

    def _bank_slots(self) -> int:
        return self._bank_group_sbs() * self._size_sb()

    def _bank_slot(self, bank: int, index: int) -> tuple[int, int]:
        per = self._slots()
        blk_idx, off = divmod(index, per)
        return self._bank_blocks(bank)[blk_idx], off + 1

    def _next_bank(self, bank: int) -> int:
        n = self._bank_count()
        if n <= 1:
            return bank
        return (bank + 1) % n

    def _write_bank(self, bank: int, index: int, body: bytes, *,
                    flags: int = FLAG_NONE, kind: int = KIND_META) -> None:
        blk, off = self._bank_slot(bank, index)
        if self.dev.is_bad(blk):
            # a bad block loses its whole slot range; the caller's cursor has
            # already been advanced, so the snapshot simply uses fewer slots
            return
        tag = pack_tag(self.seq, LPN_NONE, flags, kind)
        self.dev.program_pages(
            [(self.g.ppn(blk, off), body.ljust(self.g.page_size, b"\xff"), tag)],
            seq=self.seq,
        )
        self.seq += 1
        if kind == KIND_DISCARD:
            self.stats["discard_pages"] += 1
        else:
            self.stats["meta_pages"] += 1

    def _erase_bank(self, bank: int) -> None:
        for blk in self._bank_blocks(bank):
            if self.dev.is_bad(blk):
                continue
            if self.dev.programmed_in_block(blk):
                self.dev.erase_block(blk, seq=self.seq)
                self.seq += 1
                self.stats["erases"] += 1

    def _bank_used(self, bank: int) -> int:
        """Number of meta pages programmed in a bank (0.._bank_slots)."""
        used = 0
        for blk in self._bank_blocks(bank):
            used += self.dev.programmed_in_block(blk)
        return used

    # -- bad block log -----------------------------------------------------
    def _mark_bad(self, block: int, *, reason: str) -> None:
        if self.dev.is_bad(block):
            return
        self.dev.mark_bad(block, seq=self.seq, reason=reason)
        self.stats["runtime_bad"] += 1
        info = self.sb.get((block - self.cfg.meta_blocks) // self.g.planes)
        if info is not None:
            info.kind = "bad"
        self._append_badlog(block, reason)

    def _append_badlog(self, block: int, reason: str, *, erase_count: int | None = None) -> None:
        g = self.g
        rc = {v: k for k, v in BAD_REASON.items()}.get(reason, 2)
        ec = self.dev.erases[block] if erase_count is None else erase_count
        payload = BAD.pack(BAD_MAGIC, block, ec, rc)
        if self._badlog_cursor >= self._slots():
            self._compact_badlog()
        off = self._badlog_cursor + 1
        for blk in (0, 1):
            tag = pack_tag(self.seq, LPN_NONE, FLAG_NONE, KIND_META)
            self.dev.program_pages([(g.ppn(blk, off), payload.ljust(g.page_size, b"\xff"), tag)], seq=self.seq)
            self.seq += 1
            self.stats["badlog_pages"] += 1
        self._badlog_cursor += 1

    def _compact_badlog(self) -> None:
        g = self.g
        for blk in (0, 1):
            if self.dev.programmed_in_block(blk):
                self.dev.erase_block(blk, seq=self.seq)
                self.seq += 1
                self.stats["erases"] += 1
        self._badlog_cursor = 0
        runtime = [b for b in self.dev.bad_blocks if b not in self.dev.factory_bad]
        for b in runtime:
            self._append_badlog(b, self.dev.bad_blocks[b]["reason"],
                                erase_count=self.dev.bad_blocks[b]["erase_count"])

    # ==================================================================
    # checkpoint
    # ==================================================================
    def _snapshot_pages(self) -> int:
        head = CKGEN.size + ENT.size + CKPT_PAGE.size
        ents_per_page = (self.g.page_size - head) // ENT.size
        return (self.n_lpn + ents_per_page - 1) // ents_per_page

    def _ensure_discard_room(self, n: int) -> None:
        """Make sure ``n`` discard-record pages fit in the current bank.

        A discard record is only meaningful together with the mapping state
        it follows, so a bank may not be recycled while it still holds live
        discard records.  If the current bank is full we therefore take a
        fresh snapshot first: the snapshot folds every discard into the map,
        after which the old bank's records are redundant and it can be
        recycled safely.
        """
        if n > self._bank_slots():
            raise NoSpace("discard batch larger than a metadata bank")
        if self._bank_used(self._cur_bank) + n <= self._bank_slots():
            return
        self.checkpoint(force=True)
        if self._bank_used(self._cur_bank) + n > self._bank_slots():
            raise NoSpace("metadata bank too small for one discard batch")

    def checkpoint(self, *, force: bool = False) -> None:
        """Snapshot the mapping table into the next bank.

        The target bank is the one *not* holding the newest committed
        snapshot, so a power cut while rotating leaves the old snapshot
        loadable.  ``CKGEN + ENT(0, seq)`` lead every snapshot page so
        tag-based replay can order snapshot and discard records together.
        """
        if self._wb:
            self.flush()
        need = self._snapshot_pages() + 1
        if need > self._bank_slots():
            raise ValueError("a snapshot does not fit in one metadata bank")
        old_bank = self._cur_bank
        new_bank = self._next_bank(old_bank)
        if new_bank == old_bank:
            raise ValueError("need at least two metadata banks (meta_blocks >= 6)")
        newest = self._newest_commit()
        if newest is not None and newest[1] == new_bank:
            # never erase the bank that holds the newest committed snapshot
            for _ in range(self._bank_count()):
                new_bank = self._next_bank(new_bank)
                if newest[1] != new_bank:
                    break
            else:
                raise NoSpace("no free metadata bank to rotate into")
        # the bank we are about to reuse is the *older* snapshot's bank; the
        # newest committed snapshot stays intact until the new commit lands.
        self._erase_bank(new_bank)
        self._bank_index = 0
        head = CKGEN.size + ENT.size + CKPT_PAGE.size
        ents_per_page = (self.g.page_size - head) // ENT.size
        n_pages = self._snapshot_pages()
        gen = self._ckpt_gen + 1
        ckpt_seq = self.seq
        crc_all = 0
        index = 0
        for page in range(n_pages):
            chunk = bytearray()
            lo = page * ents_per_page
            hi = min(self.n_lpn, (page + 1) * ents_per_page)
            for lpn in range(lo, hi):
                chunk += ENT.pack(lpn, self.map[lpn])
            crc = zlib.crc32(chunk) & 0xFFFFFFFF
            crc_all = zlib.crc32(chunk, crc_all) & 0xFFFFFFFF
            hdr = CKPT_PAGE.pack(CKPT_MAGIC, gen, page, crc)
            self._write_bank(new_bank, index,
                             CKGEN.pack(CKGEN_MAGIC, gen) + ENT.pack(0, ckpt_seq) + hdr + bytes(chunk))
            index += 1
        self._write_bank(new_bank, index, COMMIT.pack(COMMIT_MAGIC, gen, n_pages, ckpt_seq, crc_all))
        self.stats["checkpoints"] += 1
        self.stats["ckpt_pages"] += need
        self._ckpt_gen = gen
        self._ckpt_seq = ckpt_seq
        self._cur_bank = new_bank
        self._bank_index = index + 1

    # ==================================================================
    # recovery
    # ==================================================================
    def recover(self) -> None:
        self.map = [LPN_NONE] * self.n_lpn
        self.discarded = set()
        self.seq = 1
        self._badlog_cursor = 0
        self._read_markers()
        self._replay_badlog()
        ckpt = self._load_checkpoint()
        data_seq: dict[int, int] = {}
        if ckpt is not None:
            entries, gen, seq, bank = ckpt
            for lpn, ppn in entries:
                if ppn != LPN_NONE:
                    self.map[lpn] = ppn
            self._ckpt_gen = gen
            self._ckpt_seq = seq
            self._cur_bank = bank
            # the next free slot is whatever follows the last programmed page;
            # discard records may have landed after the commit page, so
            # "commit index + 1" would overwrite them.
            self._bank_index = min(self._bank_used(bank), self._bank_slots())
            self.stats["recovery_mode"] = "checkpoint+scan"
            self.stats["recovery_ckpt_gen"] = gen
            # Everything at or below ckpt_seq is already folded into the
            # snapshot -- including LPNs the snapshot says are unmapped.
            # Seeding every LPN with ckpt_seq stops an old data page from
            # resurrecting a page that was discarded before the snapshot.
            data_seq = {lpn: seq for lpn in range(self.n_lpn)}
        else:
            self._ckpt_seq = 0
            self._cur_bank = 0
            self._bank_index = 0
            self.stats["recovery_mode"] = "full scan"
            self.stats["recovery_ckpt_gen"] = -1
        disc_seq: dict[int, int] = {}
        maxseq = self._scan_meta(disc_seq)
        maxseq = max(maxseq, self._scan_data(data_seq, self._ckpt_seq))
        # A discard record newer than everything that landed for the LPN
        # unmaps it.  Records at or below ckpt_seq are already reflected in
        # the snapshot's entries; they are skipped, not replayed over newer
        # data.
        for lpn, seq in disc_seq.items():
            if seq > self._ckpt_seq and seq > data_seq.get(lpn, -1):
                self.map[lpn] = LPN_NONE
                self.discarded.add(lpn)
        self.seq = max(max(maxseq, self._ckpt_seq), max(data_seq.values(), default=0),
                       max(disc_seq.values(), default=0)) + 1
        self._rebuild_state()

    def _read_markers(self) -> None:
        g = self.g
        for b in range(self.cfg.meta_blocks, g.n_blocks):
            ppn = g.ppn(b, 0)
            if not self.dev.page_programmed(ppn):
                continue
            payload = self.dev.read_page(ppn)
            if payload[:4] != EMARK_MAGIC:
                continue
            _, blk, ec, generation = EMARK.unpack(payload[: EMARK.size])
            if blk == b:
                self.dev.erases[b] = max(self.dev.erases[b], ec)

    def _replay_badlog(self) -> None:
        g = self.g
        seen = 0
        for off in range(1, g.pages_per_block):
            ppns = [g.ppn(0, off), g.ppn(1, off)]
            page = None
            for ppn in ppns:
                if not self.dev.page_programmed(ppn):
                    continue
                payload = self.dev.read_page(ppn)
                if payload[:4] == BAD_MAGIC:
                    page = payload
                    break
            if page is None:
                continue
            _, block, ec, rc = BAD.unpack(page[: BAD.size])
            if block < self.cfg.meta_blocks or block >= g.n_blocks:
                continue
            if not self.dev.is_bad(block):
                self.dev.mark_bad(block, seq=0, reason=BAD_REASON.get(rc, "runtime"))
                self.dev.bad_blocks[block]["erase_count"] = ec
            seen += 1
        self._badlog_cursor = seen

    def _iter_meta_pages(self):
        for bank in range(self._bank_count()):
            for index in range(self._bank_slots()):
                blk, off = self._bank_slot(bank, index)
                ppn = self.g.ppn(blk, off)
                if self.dev.page_programmed(ppn):
                    yield bank, index, ppn

    def _newest_commit(self) -> tuple[int, int] | None:
        """(generation, bank) of the newest committed snapshot, cheaply."""
        best = None
        for bank, index, ppn in self._iter_meta_pages():
            payload = self.dev.read_page(ppn)
            if payload[:4] == COMMIT_MAGIC:
                gen = COMMIT.unpack(payload[: COMMIT.size])[1]
                if best is None or gen > best[0]:
                    best = (gen, bank)
        return best

    def _load_checkpoint(self):
        commits = []
        for bank, index, ppn in self._iter_meta_pages():
            payload = self.dev.read_page(ppn)
            if payload[:4] == COMMIT_MAGIC:
                _, gen, n_pages, seq, crc = COMMIT.unpack(payload[: COMMIT.size])
                commits.append((gen, n_pages, seq, crc, bank))
        for gen, n_pages, seq, crc, bank in sorted(commits, reverse=True):
            entries = self._read_ckpt_gen(gen, n_pages, crc, bank)
            if entries is not None:
                return entries, gen, seq, bank
        return None

    CKPT_HEAD = CKGEN.size + ENT.size + CKPT_PAGE.size

    def _read_ckpt_gen(self, gen: int, n_pages: int, crc_all: int, bank: int):
        pages: dict[int, bytes] = {}
        h = self.CKPT_HEAD
        ents_per_page = (self.g.page_size - h) // ENT.size
        off0 = CKGEN.size + ENT.size
        for index in range(self._bank_slots()):
            blk, off = self._bank_slot(bank, index)
            ppn = self.g.ppn(blk, off)
            if not self.dev.page_programmed(ppn):
                continue
            payload = self.dev.read_page(ppn)
            if payload[off0 : off0 + 4] != CKPT_MAGIC:
                continue
            _, g_, idx, crc = CKPT_PAGE.unpack(payload[off0:h])
            if g_ != gen or idx >= n_pages:
                continue
            n_ent = min(ents_per_page, self.n_lpn - idx * ents_per_page)
            data = payload[h : h + n_ent * ENT.size]
            if len(data) != n_ent * ENT.size or zlib.crc32(data) & 0xFFFFFFFF != crc:
                return None
            pages[idx] = data
        if len(pages) != n_pages:
            return None
        body = b"".join(pages[i] for i in range(n_pages))
        if zlib.crc32(body) & 0xFFFFFFFF != crc_all:
            return None
        out = []
        for i in range(0, len(body) - ENT.size + 1, ENT.size):
            lpn, ppn = ENT.unpack(body[i : i + ENT.size])
            if lpn < self.n_lpn:
                out.append((lpn, ppn))
        return out

    def _scan_meta(self, disc_seq: dict[int, int]) -> int:
        """Track the highest meta seq and collect discard records by seq."""
        g = self.g
        maxseq = 0
        newest = -1
        for bank, index, ppn in self._iter_meta_pages():
            tag = self.dev.read_tag(ppn)
            if tag is None:
                continue
            maxseq = max(maxseq, tag.seq)
            if tag.kind == KIND_DISCARD:
                self._collect_discard(self.dev.read_page(ppn), tag.seq, disc_seq)
        for blk in (0, 1):
            for off in range(1, g.pages_per_block):
                ppn = g.ppn(blk, off)
                if not self.dev.page_programmed(ppn):
                    continue
                tag = self.dev.read_tag(ppn)
                if tag is not None:
                    maxseq = max(maxseq, tag.seq)
        return maxseq

    def _scan_data(self, data_seq: dict[int, int], ckpt_seq: int) -> int:
        g = self.g
        maxseq = 0
        scanned = 0
        ps = self.dev.page_state
        for s, info in self.sb.items():
            if info.kind == "bad":
                continue
            for blk in info.blocks:
                base = blk * g.pages_per_block
                for ppn in range(base + 1, base + g.pages_per_block):
                    if not ps[ppn]:
                        continue
                    scanned += 1
                    tag = self.dev.read_tag(ppn)
                    if tag is None:
                        continue
                    if tag.seq > maxseq:
                        maxseq = tag.seq
                    if tag.kind != KIND_DATA or tag.lpn >= self.n_lpn:
                        continue
                    lpn = tag.lpn
                    if tag.seq > data_seq.get(lpn, 0):
                        data_seq[lpn] = tag.seq
                        self.map[lpn] = ppn
        self.stats["recovery_scanned"] = scanned
        return maxseq

    def _collect_discard(self, payload: bytes, seq: int, out: dict[int, int]) -> None:
        if payload[:4] != DISC_MAGIC:
            return
        _, count = DISC_HDR.unpack(payload[: DISC_HDR.size])
        if count > (self.g.page_size - DISC_HDR.size) // 4:
            return
        base = DISC_HDR.size
        for i in range(count):
            lpn = struct.unpack("<I", payload[base + 4 * i : base + 4 * i + 4])[0]
            if lpn < self.n_lpn and seq > out.get(lpn, 0):
                out[lpn] = seq

    def _apply_discard_record(self, payload: bytes, seq: int) -> None:
        disc: dict[int, int] = {}
        self._collect_discard(payload, seq, disc)
        for lpn in disc:
            self.map[lpn] = LPN_NONE
            self.discarded.add(lpn)

    def _rebuild_state(self) -> None:
        g = self.g
        self.free = []
        self.active = None
        self.stats["max_erases"] = 0
        for s, info in self.sb.items():
            if any(self.dev.is_bad(b) for b in info.blocks):
                info.kind = "bad"
                continue
            info.valid = 0
            info.next_off = tuple(self.dev.programmed_in_block(b) for b in info.blocks)
            info.erase_count = max(self.dev.erases[b] for b in info.blocks)
            self.stats["max_erases"] = max(self.stats["max_erases"], info.erase_count)
            if info.erase_count >= g.pe_cycles:
                info.kind = "retired"
                continue
            prog = info.programmed
            if prog <= 1:
                info.kind = "free"
                self.free.append(s)
            elif prog < g.pages_per_block:
                info.kind = "active"
            else:
                info.kind = "closed"
        for lpn, ppn in enumerate(self.map):
            if ppn != LPN_NONE:
                self.sb[self._sb_of_ppn(ppn)].valid += 1
        act = [s for s, i in self.sb.items() if i.kind == "active"]
        if act:
            # the most advanced block is the one being appended to
            self.active = max(act, key=lambda s: (self.sb[s].programmed, -s))
        # rebuild the bad log cursor from how many entries were replayed
        self._badlog_cursor = min(self._badlog_cursor, self._slots())

    # ==================================================================
    # GC
    # ==================================================================
    def _closed_candidates(self) -> list[SBInfo]:
        return [
            i
            for s, i in self.sb.items()
            if i.kind in ("closed", "active") and s != self.active and i.programmed > 1
        ]

    def gc(self, reason: str = "space", *, force: bool = False) -> None:
        self.stats["gc_invocations"] += 1
        cands = self._closed_candidates()
        if not cands:
            if self.free:
                return
            raise NoSpace("nothing to reclaim")
        self.stats["gc_scanned"] += len(cands)
        # A victim's valid pages must fit in the free pool, keeping one
        # superblock of headroom for the copy target itself.  Picking a
        # victim that cannot fit is how an FTL wedges itself under pressure.
        room = max(0, (self.free_sb() - 1) * self._size_sb())
        fit = [i for i in cands if i.valid <= room] or []
        zeros = [i for i in cands if i.valid == 0]
        pool = zeros or (fit if fit else cands)
        if zeros:
            victim = min(zeros, key=lambda i: (i.erase_count, i.index))
        elif self.cfg.gc_policy == "greedy":
            victim = min(pool, key=lambda i: (i.valid, i.erase_count, i.index))
        else:
            # Cost-benefit (Rosenblum-style) victim choice:
            #   benefit = 1 - dirty_fraction  (how much of the block is dead)
            #   cost    = 1 + wear_penalty    (how expensive to keep rewriting)
            # pick max benefit/cost.  Wear penalty grows with the *relative*
            # erase count among candidates, so garbage still drives the
            # choice while a hot block loses to a cold one holding the same
            # amount of garbage.
            mx = max(i.erase_count for i in pool) or 1
            w = self.cfg.gc_wear_weight

            def value(i: SBInfo) -> float:
                dead = 1.0 - i.valid / self._size_sb()
                wear = i.erase_count / mx
                return dead / (1.0 + w * wear)

            victim = min(pool, key=lambda i: (-value(i), i.valid, i.index))
        if victim.valid > room and not self.free:
            raise NoSpace("no victim fits in the free pool (OP exhausted)")
        valid_before = victim.valid
        self._in_gc = True
        try:
            copied = self._relocate(victim, reason=reason)
        finally:
            self._in_gc = False
        victim.kind = "free"
        victim.valid = 0
        victim.next_off = tuple(0 for _ in victim.blocks)
        self.free.append(victim.index)
        self.stats["gc_erases"] += 1
        self.gc_log.append(
            GCRecord(
                seq=self.seq,
                sb=victim.index,
                policy=self.cfg.gc_policy,
                valid_before=valid_before,
                valid_copied=copied,
                erase_count=victim.erase_count,
                scanned=len(cands),
                reason=reason,
            )
        )

    def _relocate(self, victim: SBInfo, *, reason: str) -> int:
        g = self.g
        copied = 0
        for i, blk in enumerate(victim.blocks):
            base = blk * g.pages_per_block
            for off in range(1, victim.next_off[i]):
                ppn = base + off
                tag = self.dev.read_tag(ppn)
                if tag is None or tag.kind != KIND_DATA or tag.lpn >= self.n_lpn:
                    continue
                lpn = tag.lpn
                if self.map[lpn] != ppn or lpn in self.discarded:
                    continue
                self._stage(lpn, self.dev.read_page(ppn))
                copied += 1
                self.stats["gc_copies"] += 1
                if reason == "static_wl":
                    self.stats["wl_copies"] += 1
        self.flush()
        return copied

    # ==================================================================
    # static wear levelling
    # ==================================================================
    def _maybe_static_wl(self) -> None:
        closed = [
            i for i in self.sb.values()
            if i.kind == "closed" and i.programmed > 1 and i.valid > 0
        ]
        if len(closed) < 8:
            return
        lo = min(i.erase_count for i in closed)
        hi = max(i.erase_count for i in closed)
        if hi - lo < self.cfg.static_wl_min_spread:
            return
        age_limit = self.seq - self.cfg.static_wl_min_age
        cold = [
            i for i in closed
            if i.last_write_seq < age_limit
        ]
        cold.sort(key=lambda i: (i.erase_count, i.last_write_seq, i.index))
        moved = 0
        for victim in cold:
            if moved >= self.cfg.static_wl_batch:
                break
            if victim.kind != "closed" or victim.valid == 0:
                continue
            if self.free_sb() <= 1:
                return
            before = victim.valid
            self._in_gc = True
            try:
                self._relocate(victim, reason="static_wl")
            finally:
                self._in_gc = False
            victim.kind = "free"
            victim.valid = 0
            victim.next_off = tuple(0 for _ in victim.blocks)
            self.free.append(victim.index)
            self.stats["wl_moves"] += 1
            moved += 1
            self.gc_log.append(
                GCRecord(
                    seq=self.seq,
                    sb=victim.index,
                    policy=self.cfg.gc_policy,
                    valid_before=before,
                    valid_copied=before,
                    erase_count=victim.erase_count,
                    scanned=len(closed),
                    reason="static_wl",
                )
            )

    # ==================================================================
    # reporting
    # ==================================================================
    def live_pages(self) -> int:
        return sum(1 for p in self.map if p != LPN_NONE)

    def wear_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for info in self.sb.values():
            if info.kind == "bad":
                continue
            hist[info.erase_count] = hist.get(info.erase_count, 0) + 1
        return dict(sorted(hist.items()))

    def wear_stats(self) -> dict:
        live = [i for i in self.sb.values() if i.kind not in ("bad", "free")]
        if not live:
            live = [i for i in self.sb.values() if i.kind != "bad"] or list(self.sb.values())
        erases = sorted(i.erase_count for i in live)
        n = len(erases)
        mean = sum(erases) / n if n else 0.0
        return {
            "superblocks": n,
            "min": erases[0] if n else 0,
            "max": erases[-1] if n else 0,
            "mean": mean,
            "median": erases[n // 2] if n else 0,
            "spread": (erases[-1] - erases[0]) if n else 0,
            "stdev": (sum((e - mean) ** 2 for e in erases) / n) ** 0.5 if n else 0.0,
            "p95": erases[int(0.95 * (n - 1))] if n else 0,
        }

    def physical_pages(self) -> int:
        """Every page programmed on the die, by category (mutually exclusive)."""
        return (self.stats["data_programs"] + self.stats["marker_pages"]
                + self.stats["meta_pages"] + self.stats["discard_pages"]
                + self.stats["badlog_pages"])

    def wa(self) -> float:
        hp = self.stats["host_writes"]
        return self.physical_pages() / hp if hp else 0.0


def rng_hit(p: float, seq: int, block: int) -> bool:
    """Deterministic pseudo-random test with probability ``p``."""
    if p <= 0:
        return False
    x = (seq * 0x9E3779B97F4A7C15 + block * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    return (x / float(1 << 64)) < p
