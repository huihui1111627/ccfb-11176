"""Synthetic flight-log workload.

A flight-controller logger writes, at 4 KiB logical-page granularity:

* a high-rate append stream (IMU/attitude/GPS/actuator records grouped into
  log segments; segments are opened sequentially and closed when full);
* a handful of hot status pages (current-flight summary, params, ring head);
* a small set of cold config/calibration pages written at boot and rarely
  after;
* occasional random rewrites (log-index fixups);
* discards of whole old segments after download or under storage pressure;
* part of the append traffic is issued as two-plane pairs, as a logger that
  flushes two adjacent 4 KiB buffers at once would.

The generator is deterministic and cheap: it yields small *abstract* payloads
(the FTL pads them to a page).  A payload carries a monotonic counter, so the
recovery check cannot pass by accident.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum


class Op(IntEnum):
    WRITE = 0
    PAIR = 1
    DISCARD = 2


@dataclass(slots=True)
class Write:
    kind: int
    lpn: int
    payload: bytes
    lpn_b: int = -1
    payload_b: bytes = b""


@dataclass
class WorkloadParams:
    n_lpn: int
    n_ops: int
    seed: int = 1

    segment_pages: int = 1024  # LPNs per log segment
    append_share: float = 0.82
    hot_share: float = 0.10
    cold_share: float = 0.03
    random_share: float = 0.05
    pair_share: float = 0.30  # of appends
    discard_every: int = 4000  # ops between segment discards
    hot_pages: int = 8
    cold_pages: int = 24
    payload_bytes: int = 32

    def __post_init__(self) -> None:
        total = self.append_share + self.hot_share + self.cold_share + self.random_share
        self.append_share /= total
        self.hot_share /= total
        self.cold_share /= total
        self.random_share /= total


PAYLOAD_BYTES = 32


def make_payload(tag: int, counter: int, size: int = PAYLOAD_BYTES) -> bytes:
    body = (tag & 0xFFFFFFFF).to_bytes(4, "little") + (counter & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    return (body * (size // len(body) + 1))[:size]


class FlightLogWorkload:
    def __init__(self, p: WorkloadParams):
        if p.n_lpn < 64:
            raise ValueError("logical space too small")
        self.p = p
        self.rng = random.Random(p.seed)
        self.counter = 0
        n = p.n_lpn
        self.hot = list(range(min(p.hot_pages, max(1, n // 100))))
        n_cold = min(p.cold_pages, max(1, n // 100))
        self.cold = list(range(len(self.hot), len(self.hot) + n_cold))
        self.append_base = len(self.hot) + len(self.cold)
        self.append_end = n
        self.span = self.append_end - self.append_base
        if self.span < p.segment_pages:
            raise ValueError("append region smaller than one segment")
        if p.segment_pages > self.span // 3:
            # keep at least three segments live so a ring wrap never collides
            p.segment_pages = max(8, self.span // 4)
        self._seg_start = self.append_base
        self._pos = 0
        self._live_segments: list[tuple[int, int]] = []  # (start, pages)
        self._ops = 0

    def _next_payload(self, tag: int) -> bytes:
        self.counter += 1
        return make_payload(tag, self.counter, self.p.payload_bytes)

    # -- append cursor -----------------------------------------------------
    def _open_segment(self) -> None:
        seg = self.p.segment_pages
        if self._pos + seg > self.span:
            self._pos = 0
            # a wrapping segment must not overlap the hottest live one
            self._live_segments = [
                sg for sg in self._live_segments
                if sg[0] >= self.append_base + self._pos + seg
            ]
        start = self.append_base + self._pos
        self._live_segments.append((start, seg))
        self._pos += seg
        self._seg_off = 0

    def _append_lpn(self) -> int:
        if (not self._live_segments or self._seg_off >= self.p.segment_pages
                or self._live_segments[-1][0] + self._seg_off >= self.append_end):
            self._open_segment()
        lpn = self._live_segments[-1][0] + self._seg_off
        self._seg_off += 1
        return lpn

    def _oldest_discardable(self) -> tuple[int, int] | None:
        # keep the two newest segments; discard the oldest whole segment
        if len(self._live_segments) <= 2:
            return None
        return self._live_segments.pop(0)

    def _emit_discard(self) -> Write:
        seg = self._oldest_discardable()
        if seg is None:
            return Write(Op.DISCARD, -1, b"")
        start, pages = seg
        return Write(Op.DISCARD, start, b"", lpn_b=pages)

    def __iter__(self):
        p = self.p
        for i in range(p.n_ops):
            r = self.rng.random()
            if r < p.append_share:
                lpn = self._append_lpn()
                if self.rng.random() < p.pair_share and self._seg_off < p.segment_pages:
                    # two adjacent pages in one multi-plane program
                    lpn_b = self._append_lpn()
                    yield Write(Op.PAIR, lpn, self._next_payload(1),
                                lpn_b, self._next_payload(2))
                else:
                    yield Write(Op.WRITE, lpn, self._next_payload(1))
            elif r < p.append_share + p.hot_share:
                lpn = self.hot[self.rng.randrange(len(self.hot))]
                yield Write(Op.WRITE, lpn, self._next_payload(3))
            elif r < p.append_share + p.hot_share + p.cold_share:
                lpn = self.cold[self.rng.randrange(len(self.cold))]
                yield Write(Op.WRITE, lpn, self._next_payload(4))
            else:
                lpn = self.rng.randrange(p.n_lpn)
                yield Write(Op.WRITE, lpn, self._next_payload(5))
            if p.discard_every and i and i % p.discard_every == 0:
                d = self._emit_discard()
                if d.lpn >= 0:
                    yield d
