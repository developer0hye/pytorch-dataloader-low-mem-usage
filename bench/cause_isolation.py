"""Cause isolation: which write makes a forked DataLoader worker copy the dataset pages?

Goal: separate the *causes* of "memory scales with num_workers":
  list      -> dataset metadata as millions of small Python objects (baseline)
  gcfreeze  -> same, but gc.freeze() before fork (removes GC-header writes only)
  immortal  -> same, but every object made immortal via ctypes (removes refcount
               writes; PEP 683 mechanism, no public API in 3.12 so we poke
               ob_refcnt directly -- experiment only, never do this in prod)
  numpy     -> metadata packed in two contiguous numpy arrays (no per-item
               Python object header inside the shared pages)

We report PSS (proportional set size) per worker, not RSS: RSS counts shared
copy-on-write pages once per process and would overstate duplication.
"""
import ctypes
import gc
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

N = int(os.environ.get("N", "3000000"))
WORKERS = int(os.environ.get("WORKERS", "4"))
MODE = sys.argv[1]


def smaps(pid, key):
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024  # MB
    return float("nan")


class ListDS(Dataset):
    def __init__(self):
        # (path, label) per item: tuple + str + int = 3 Python objects per item.
        self.items = [(f"/data/images/{i:09d}.jpg", i) for i in range(N)]

    def __len__(self):
        return N

    def __getitem__(self, i):
        p, l = self.items[i]  # touching p/l increments their refcounts -> page write
        return len(p), l


class NumpyDS(Dataset):
    def __init__(self):
        tmp = [f"/data/images/{i:09d}.jpg" for i in range(N)]
        self.paths = np.array(tmp, dtype="S32")  # one contiguous buffer
        del tmp
        self.labels = np.arange(N, dtype=np.int64)
        gc.collect()

    def __len__(self):
        return N

    def __getitem__(self, i):
        # Indexing creates a *new* Python object in the worker; the shared
        # buffer itself is only read.
        return len(self.paths[i]), int(self.labels[i])


def immortalize(obj_list):
    # CPython 3.12+: ob_refcnt == 0xFFFFFFFF (low 32 bits saturated) is treated
    # as immortal; Py_INCREF/DECREF become no-ops -> no page write on access.
    IMMORTAL = 0xFFFFFFFF
    n = 0
    for t in obj_list:
        for o in t:
            ctypes.c_ssize_t.from_address(id(o)).value = IMMORTAL
            n += 1
        ctypes.c_ssize_t.from_address(id(t)).value = IMMORTAL
        n += 1
    ctypes.c_ssize_t.from_address(id(obj_list)).value = IMMORTAL
    return n + 1


def main():
    t0 = time.time()
    if MODE == "numpy":
        ds = NumpyDS()
    else:
        ds = ListDS()
    if MODE == "gcfreeze":
        gc.collect()
        gc.freeze()
    if MODE == "immortal":
        immortalize(ds.items)
        gc.collect()
        gc.freeze()  # also keep GC from touching headers, isolate refcount effect
    build_s = time.time() - t0
    main_rss = smaps(os.getpid(), "Rss")

    dl = DataLoader(ds, batch_size=512, num_workers=WORKERS, shuffle=True,
                    persistent_workers=True, prefetch_factor=2)
    it = iter(dl)
    next(it)  # workers are up now
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

    print(f"mode={MODE:9s} N={N} workers={WORKERS} build={build_s:.1f}s epoch={epoch_s:.1f}s "
          f"main_rss={main_rss:.0f}MB")
    print(f"  worker Pss           start {np.mean(pss_start):7.0f}MB -> end {np.mean(pss_end):7.0f}MB")
    print(f"  worker Private_Dirty start {np.mean(priv_start):7.0f}MB -> end {np.mean(priv_end):7.0f}MB")
    print(f"  worker Rss (naive)   end   {np.mean(rss_end):7.0f}MB   "
          f"sum over {WORKERS} workers private={sum(priv_end):.0f}MB")
    del it, dl


if __name__ == "__main__":
    main()
