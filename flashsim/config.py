"""NAND geometry and user-facing configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Geometry:
    """Raw NAND die.

    Blocks are numbered ``block = block_in_plane * planes + plane``; the
    "twin" of a block is its neighbour in the next plane, so a two-plane
    program writes the same page offset in ``block`` and ``block ^ 1``.

    Physical page number: ``ppn = block * pages_per_block + page``.
    Page offset 0 of every block is reserved for the erase marker; offsets
    ``1 .. pages_per_block-1`` hold data.
    """

    page_size: int = 4096
    pages_per_block: int = 256
    blocks_per_plane: int = 512
    planes: int = 2
    pe_cycles: int = 30_000

    def __post_init__(self) -> None:
        if self.planes not in (1, 2, 4):
            raise ValueError("planes must be 1, 2 or 4")
        if self.pages_per_block < 8:
            raise ValueError("pages_per_block must be >= 8")

    @property
    def n_blocks(self) -> int:
        return self.blocks_per_plane * self.planes

    @property
    def pages(self) -> int:
        return self.n_blocks * self.pages_per_block

    @property
    def capacity_bytes(self) -> int:
        return self.pages * self.page_size

    def ppn(self, block: int, page: int) -> int:
        return block * self.pages_per_block + page

    def block_of(self, ppn: int) -> int:
        return ppn // self.pages_per_block

    def page_of(self, ppn: int) -> int:
        return ppn % self.pages_per_block

    def plane_of(self, block: int) -> int:
        return block % self.planes

    def block_in_plane(self, block: int) -> int:
        return block // self.planes

    def pair(self, block: int) -> tuple[int, int]:
        b0 = block & ~1
        return b0, b0 | 1

    def describe(self) -> str:
        return (
            f"{self.planes} plane(s) x {self.blocks_per_plane} blocks/plane x "
            f"{self.pages_per_block} pages x {self.page_size}B = "
            f"{self.capacity_bytes / 2**20:.0f} MiB raw, P/E {self.pe_cycles:,}"
        )


@dataclass
class Config:
    geometry: Geometry = field(default_factory=Geometry)

    # -- FTL layout --------------------------------------------------------
    meta_blocks: int = 32          # multiple of planes
    op_ratio: float = 0.07         # over-provisioning fraction of user cap
    logical_page_count: int | None = None

    # -- GC policy ---------------------------------------------------------
    gc_policy: str = "cost"        # "cost" | "greedy"
    gc_wear_weight: float = 4.0    # weight of erase count in the cost score
    gc_reserve_pct: float = 3.0    # GC starts below this % of free superblocks

    # -- wear levelling ----------------------------------------------------
    dynamic_wl: bool = True
    static_wl: bool = True
    static_wl_min_spread: int = 6   # erase-count spread that triggers a move
    static_wl_min_age: int = 50_000  # LPN untouched for this many writes = cold
    static_wl_check_every: int = 128
    static_wl_batch: int = 1

    # -- metadata durability ----------------------------------------------
    checkpoint_every: int = 10_000  # host page writes between checkpoints

    # -- fault injection ---------------------------------------------------
    factory_bad_pct: float = 0.01
    erase_fail_prob: float = 2e-4   # runtime bad block on erase
    multi_plane: bool = True

    # -- power loss --------------------------------------------------------
    seed: int = 0

    def __post_init__(self) -> None:
        if self.gc_policy not in ("cost", "greedy"):
            raise ValueError("gc_policy must be 'cost' or 'greedy'")
        if not 0.0 <= self.op_ratio < 1.0:
            raise ValueError("op_ratio must be in [0, 1)")
        if self.meta_blocks % self.geometry.planes:
            raise ValueError("meta_blocks must be a multiple of planes")

    def clone(self, **kw) -> "Config":
        return replace(self, **kw)

    @property
    def g(self) -> Geometry:
        return self.geometry

    @property
    def user_blocks(self) -> int:
        return self.geometry.n_blocks - self.meta_blocks

    @property
    def pages_per_superblock(self) -> int:
        return self.geometry.pages_per_block * self.geometry.planes

    @property
    def data_pages_per_block(self) -> int:
        return self.geometry.pages_per_block - 1

    @property
    def logical_pages(self) -> int:
        if self.logical_page_count is not None:
            return self.logical_page_count
        # Factory-bad blocks are not usable capacity, so they must not eat
        # into the over-provisioning the caller asked for.
        good_blocks = self.user_blocks * (1.0 - self.factory_bad_pct)
        total = int(good_blocks * self.data_pages_per_block)
        return int(total / (1.0 + self.op_ratio))

    @property
    def op_fraction(self) -> float:
        good_blocks = self.user_blocks * (1.0 - self.factory_bad_pct)
        total = int(good_blocks * self.data_pages_per_block)
        return (total - self.logical_pages) / self.logical_pages
