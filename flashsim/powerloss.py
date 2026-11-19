"""Power-loss test rig.

The rig drives an :class:`~flashsim.ftl.FTL` through a workload, injects a
power cut at a chosen device operation, then reboots a fresh FTL on the same
media and checks the recovery contract:

* every logical page reads back what the last **acknowledged** operation left
  there;
* a page that was never written (or was acknowledged as discarded) reads as
  unmapped;
* an operation that was *in flight* (issued but not acknowledged) when power
  died may leave any of its legal outcomes: the old payload, the new payload,
  or unmapped for a discard.  The FTL is allowed to complete it, drop it, or
  land a torn version -- but it must never produce a payload from some other
  write.

The shadow copy is updated exactly when the FTL call returns; that is the
durability contract.  The cut itself is injected inside the device, so tears
and aborts happen at real operation boundaries and the FTL sees exactly what
a reboot sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .ftl import FTL


class Shadow:
    """Host-side view of what is durable, plus in-flight alternatives."""

    def __init__(self, n_lpn: int):
        self.data: list[bytes | None] = [None] * n_lpn
        # lpn -> set of payloads (None == unmapped) that are legal outcomes
        # for an operation that had not been acknowledged at the cut.
        self.in_flight: dict[int, set[bytes | None]] = {}
        self.acked = 0
        self.acked_discards = 0

    def issue_write(self, lpn: int, payload: bytes) -> None:
        # old and new payload are both legal if the cut lands mid-write
        self.in_flight.setdefault(lpn, set()).update({self.data[lpn], payload})

    def ack_write(self, lpn: int, payload: bytes) -> None:
        self.data[lpn] = payload
        self.in_flight.pop(lpn, None)
        self.acked += 1

    def issue_discard(self, lpn: int) -> None:
        # a discard may or may not have taken effect
        self.in_flight.setdefault(lpn, set()).update({self.data[lpn], None})

    def ack_discard(self, lpn: int) -> None:
        self.data[lpn] = None
        self.in_flight.pop(lpn, None)
        self.acked_discards += 1


@dataclass
class CheckResult:
    ok: bool
    wrong: list[tuple[int, str]] = field(default_factory=list)
    checked: int = 0
    in_flight: int = 0
    detail: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"OK checked={self.checked} in_flight={self.in_flight}"
        return (f"FAIL wrong={len(self.wrong)} checked={self.checked} "
                f"first={self.wrong[:3]} {self.detail}")


def check_recovery(shadow: Shadow, rebooted: FTL) -> CheckResult:
    wrong: list[tuple[int, str]] = []
    checked = 0
    for lpn, expected in enumerate(shadow.data):
        got = rebooted.read(lpn)
        alt = shadow.in_flight.get(lpn)
        if alt is not None:
            if got is None:
                if None in alt:
                    continue
            else:
                n = max(len(p) for p in alt if p is not None) if any(alt) else 0
                if got[:n] in {p for p in alt if p is not None}:
                    continue
            wrong.append((
                lpn,
                f"in-flight got={got[:12] if got else None} allowed={[p[:8] if p else None for p in alt]}",
            ))
            continue
        if expected is None:
            if got is not None:
                wrong.append((lpn, f"expected unmapped, got {got[:12]}"))
            continue
        checked += 1
        if got is None:
            wrong.append((lpn, "expected data, got unmapped"))
        elif got[: len(expected)] != expected:
            wrong.append((lpn, f"expected {expected[:12]} got {got[:12]}"))
    return CheckResult(ok=not wrong, wrong=wrong, checked=checked,
                       in_flight=len(shadow.in_flight))


def reboot(dev, cfg: Config) -> FTL:
    """A fresh FTL on the same media, as after a power cycle."""
    ftl = FTL(cfg, dev=dev)
    ftl.recover()
    return ftl
