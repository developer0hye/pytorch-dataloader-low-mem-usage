"""dlmem: measure what DataLoader workers actually cost in memory.

Why this exists
---------------
People diagnose pytorch/pytorch#13246 ("num_workers multiplies memory") by
summing RSS over worker processes.  That number is wrong by construction:
after fork, every copy-on-write page the worker has not touched is counted in
its RSS *again*, so 4 workers of a 1 GB parent show ~4 GB of RSS even when
nothing has been copied.  The kernel already exposes the honest numbers in
/proc/<pid>/smaps_rollup:

  Pss            resident pages, shared ones divided by the number of sharers
  Private_Dirty  pages this process alone owns and has written -- the sum of
                 copied CoW pages and fresh private allocations

This module reads those for the DataLoader's live workers and the parent and
reports per-worker private growth plus the process-family Pss (parent +
workers), which is the real RAM bill.  Linux only; on other platforms every
value is NaN and nothing raises, so it is safe to leave in training scripts.

Private_Dirty growth is *evidence consistent with* CoW duplication, not proof:
decoded samples, allocator retention and queue buffers also land there.  Use
the trend (flat vs growing with the number of samples touched) and compare
representations, do not read a single number as a verdict.

Usage
-----
    from dlmem import worker_memory, format_report

    dl = DataLoader(ds, num_workers=4, persistent_workers=True)
    it = iter(dl); next(it)                 # workers exist after the first batch
    before = worker_memory(dl)
    for batch in it: ...
    after = worker_memory(dl)
    print(format_report(before, after))
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

_KEYS = ("Rss", "Pss", "Private_Dirty", "Private_Clean", "Shared_Clean", "Shared_Dirty")


def smaps_rollup(pid: int) -> dict[str, float]:
    """Return smaps_rollup fields for *pid* in MiB; NaN where unavailable."""
    out = {k: math.nan for k in _KEYS}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in out:
                    out[key] = int(rest.split()[0]) / 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        pass
    return out


@dataclass
class MemorySnapshot:
    parent_pid: int
    parent: dict[str, float]
    workers: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    def worker_mean(self, key: str) -> float:
        vals = [w[key] for w in self.workers.values()]
        return sum(vals) / len(vals) if vals else math.nan

    def worker_sum(self, key: str) -> float:
        return sum(w[key] for w in self.workers.values()) if self.workers else math.nan

    @property
    def family_pss(self) -> float:
        """Parent + workers Pss: the honest total resident cost of this loader."""
        return self.parent["Pss"] + self.worker_sum("Pss")

    @property
    def naive_rss_sum(self) -> float:
        """What `ps`/`top` summation would report. Kept only to show the overcount."""
        return self.parent["Rss"] + self.worker_sum("Rss")


def _worker_pids(dataloader) -> list[int]:
    # DataLoader keeps the live iterator in `_iterator` only when
    # persistent_workers=True; otherwise the caller must pass the iterator.
    it = getattr(dataloader, "_iterator", None) or dataloader
    workers = getattr(it, "_workers", None) or []
    return [w.pid for w in workers if w.pid is not None and w.is_alive()]


def worker_memory(dataloader_or_iter) -> MemorySnapshot:
    """Snapshot memory of the parent and the loader's live worker processes.

    Accepts a DataLoader (with persistent_workers=True) or the iterator
    returned by iter(dataloader).  Workers only exist after the first batch
    has been requested.
    """
    pids = _worker_pids(dataloader_or_iter)
    me = os.getpid()
    return MemorySnapshot(parent_pid=me, parent=smaps_rollup(me),
                          workers={pid: smaps_rollup(pid) for pid in pids})


def format_report(before: MemorySnapshot, after: MemorySnapshot | None = None) -> str:
    """Human-readable comparison. With one snapshot, prints its absolute numbers."""
    lines = []
    n = before.num_workers
    if after is None:
        lines.append(f"workers={n}  parent Pss {before.parent['Pss']:.0f} MiB")
        lines.append(f"  worker mean  Pss {before.worker_mean('Pss'):.0f} MiB  "
                     f"Private_Dirty {before.worker_mean('Private_Dirty'):.0f} MiB  "
                     f"Rss {before.worker_mean('Rss'):.0f} MiB")
        lines.append(f"  family Pss {before.family_pss:.0f} MiB   "
                     f"(naive RSS sum {before.naive_rss_sum:.0f} MiB)")
        return "\n".join(lines)
    d_priv = after.worker_mean("Private_Dirty") - before.worker_mean("Private_Dirty")
    lines.append(f"workers={n}")
    lines.append(f"  worker Private_Dirty  {before.worker_mean('Private_Dirty'):7.0f} -> "
                 f"{after.worker_mean('Private_Dirty'):7.0f} MiB   (+{d_priv:.0f} per worker, "
                 f"+{d_priv * n:.0f} total)")
    lines.append(f"  worker Pss            {before.worker_mean('Pss'):7.0f} -> "
                 f"{after.worker_mean('Pss'):7.0f} MiB")
    lines.append(f"  family Pss            {before.family_pss:7.0f} -> {after.family_pss:7.0f} MiB   "
                 f"(naive RSS sum {after.naive_rss_sum:.0f} MiB)")
    if d_priv * n > 0.25 * max(before.parent["Rss"], 1.0):
        lines.append("  note: worker private memory grew by more than a quarter of the parent's RSS;"
                     " consistent with copy-on-write duplication of Python-object metadata"
                     " (pytorch/pytorch#13246). Compare against a packed representation.")
    return "\n".join(lines)
