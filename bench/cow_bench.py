"""Per-worker private-memory benchmark for DataLoader dataset representations.

Compares how the dataset *metadata* representation drives worker memory:

  list        millions of small Python objects (tuple/str/int)         -- baseline
  numpy       two contiguous numpy arrays (S32 paths, int64 labels)
  sharedlist  PR pytorch/pytorch#191555 SharedList wrapping the same tuples
  usm         PR #191555 `to_shared_dataset()` applied to a dataset holding the
              list (what `DataLoader(use_shared_memory=True)` does)
  fastsharedlist  fastpath.py prototype: PR layout, numpy-view reads
  sharedarray     fastpath.py prototype: typed columns shared via torch storage

across start methods (fork / forkserver / spawn).  Python 3.14 made forkserver
the POSIX default, so fork-only conclusions are not enough.

Why these metrics and not RSS: RSS counts copy-on-write pages once per process
and overstates duplication by the whole parent heap.  We report per worker
Private_Dirty (pages actually copied or freshly allocated) and Pss, plus the
process-family Pss (parent + workers) which is the real resident cost.  The
parent's ru_maxrss captures the packing peak that construction-time
serialization causes (pickle whole list -> pickle per item -> bytearray ->
frombuffer copy).

Usage: python cow_bench.py --mode list --ctx fork [--n 3000000 --workers 4]
"""
import argparse
import gc
import os
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))


def smaps(pid, key):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1024  # MB
    except FileNotFoundError:
        pass
    return float("nan")


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_items(n):
    return [(f"/data/images/{i:09d}.jpg", i) for i in range(n)]


class ListDS(Dataset):
    def __init__(self, n):
        self.items = make_items(n)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        return len(p), l


class NumpyDS(Dataset):
    def __init__(self, n):
        tmp = [f"/data/images/{i:09d}.jpg" for i in range(n)]
        self.paths = np.array(tmp, dtype="S32")
        del tmp
        self.labels = np.arange(n, dtype=np.int64)
        gc.collect()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return len(self.paths[i]), int(self.labels[i])


class SharedListDS(Dataset):
    def __init__(self, n):
        from pr_shared_container import SharedList

        tmp = make_items(n)
        self.items = SharedList(tmp)
        del tmp
        gc.collect()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        return len(p), l


class FastSharedListDS(Dataset):
    """Same tuples as SharedListDS, through the fastpath.py prototype."""

    def __init__(self, n):
        from fastpath import FastSharedList

        tmp = make_items(n)
        self.items = FastSharedList(tmp)
        del tmp
        gc.collect()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        return len(p), l


class SharedArrayDS(Dataset):
    """Typed columns (S32 paths, int64 labels) shared via torch storage."""

    def __init__(self, n):
        from fastpath import SharedArray

        tmp = [f"/data/images/{i:09d}.jpg" for i in range(n)]
        self.paths = SharedArray(np.array(tmp, dtype="S32"))
        del tmp
        self.labels = SharedArray(np.arange(n, dtype=np.int64))
        gc.collect()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return len(self.paths[i]), int(self.labels[i])


def build(mode, n):
    if mode == "list":
        return ListDS(n)
    if mode == "numpy":
        return NumpyDS(n)
    if mode == "sharedlist":
        return SharedListDS(n)
    if mode == "fastsharedlist":
        return FastSharedListDS(n)
    if mode == "sharedarray":
        return SharedArrayDS(n)
    if mode == "usm":
        from pr_shared_container import to_shared_dataset

        ds = ListDS(n)
        return to_shared_dataset(ds)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["list", "numpy", "sharedlist", "usm", "fastsharedlist", "sharedarray"])
    ap.add_argument("--ctx", default="fork", choices=["fork", "forkserver", "spawn"])
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--getitem-samples", type=int, default=20000)
    args = ap.parse_args()

    rss0 = smaps(os.getpid(), "Rss")
    t0 = time.time()
    ds = build(args.mode, args.n)
    build_s = time.time() - t0
    gc.collect()
    rss_after_build = smaps(os.getpid(), "Rss")
    peak_build = maxrss_mb()

    # Single-process __getitem__ latency: the price workers pay per sample.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ds), size=args.getitem_samples)
    t = time.perf_counter()
    for i in idx:
        ds[int(i)]
    getitem_us = (time.perf_counter() - t) / len(idx) * 1e6

    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=True,
                    persistent_workers=True, prefetch_factor=2,
                    multiprocessing_context=args.ctx)
    t_start = time.time()
    it = iter(dl)
    next(it)
    startup_s = time.time() - t_start
    pids = [w.pid for w in dl._iterator._workers]
    pss_start = [smaps(p, "Pss") for p in pids]
    priv_start = [smaps(p, "Private_Dirty") for p in pids]

    t1 = time.time()
    for _ in it:
        pass
    epoch_s = time.time() - t1
    pss_end = [smaps(p, "Pss") for p in pids]
    priv_end = [smaps(p, "Private_Dirty") for p in pids]
    rss_end = [smaps(p, "Rss") for p in pids]
    parent_pss = smaps(os.getpid(), "Pss")
    family_pss = parent_pss + sum(pss_end)

    print(f"mode={args.mode:10s} ctx={args.ctx:10s} N={args.n} workers={args.workers}")
    print(f"  build {build_s:6.1f}s  parent Rss {rss0:.0f}->{rss_after_build:.0f}MB  "
          f"parent peak(ru_maxrss) {peak_build:.0f}MB")
    print(f"  __getitem__ {getitem_us:7.2f} us/item (single process, {args.getitem_samples} random)")
    print(f"  worker startup {startup_s:5.1f}s  epoch {epoch_s:5.1f}s  "
          f"({args.n / epoch_s / 1e6:.2f} M items/s)")
    print(f"  worker Pss           start {np.mean(pss_start):7.0f}MB -> end {np.mean(pss_end):7.0f}MB")
    print(f"  worker Private_Dirty start {np.mean(priv_start):7.0f}MB -> end {np.mean(priv_end):7.0f}MB")
    print(f"  worker Rss (naive)   end   {np.mean(rss_end):7.0f}MB")
    print(f"  sum worker private {sum(priv_end):7.0f}MB   family Pss (parent+workers) {family_pss:7.0f}MB")
    del it, dl


if __name__ == "__main__":
    main()
