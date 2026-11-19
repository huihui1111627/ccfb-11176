"""A bare-NAND simulator and a page-mapped FTL for flight-log storage.

See ``README.md`` and ``report/index.html``.
"""

from .config import Config, Geometry
from .device import FlashError, NandDevice, NoSpace, PowerLoss
from .ftl import FTL
from .workload import FlightLogWorkload, Op, WorkloadParams

__all__ = [
    "Config",
    "Geometry",
    "FTL",
    "NandDevice",
    "FlashError",
    "NoSpace",
    "PowerLoss",
    "FlightLogWorkload",
    "WorkloadParams",
    "Op",
]
