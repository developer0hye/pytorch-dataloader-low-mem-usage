"""Can PR #191555's containers be made as fast as a plain list?  Prototype + timing.

The PR's SharedList.__getitem__ costs ~6 us against ~0.6 us for a list.  The
work per access is: two `tensor.item()` calls for the offsets, a tensor slice,
`.numpy().tobytes()` (a copy), then pickle.loads.  Only the last step is
inherent.  This file prototypes:

  FastSharedList   same storage layout as the PR (uint8 storage + int64 offsets
                   in torch tensors, so ForkingPickler still shares them under
                   forkserver/spawn), but the worker reads through lazily
                   created numpy views and hands pickle.loads a memoryview.
                   No per-access torch call, no bytes copy.
  SharedArray      fixed-width column (numpy dtype such as S32 / int64) whose
                   bytes live in a torch tensor for sharing, read through a
                   numpy view.  For typed metadata this is *faster* than a list.
  FastSharedDict   PR-style packed keys/values plus a sorted 64-bit
                   deterministic key digest (blake2b of the pickled key) and a
                   permutation, so lookup is one digest + searchsorted + one
                   deserialize instead of a linear pickle.loads scan.
                   Python's str hash is per-process salted, so it cannot be
                   used across spawn/forkserver workers; hence the digest.

Timing is single-process random access; sharing behaviour is checked in
cow_bench.py (mode sharedarray / fastsharedlist).
"""
import hashlib
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from pr_shared_container import SharedDict, SharedList  # noqa: E402


def _pack(items):
    blobs = [pickle.dumps(x, protocol=5) for x in items]
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in blobs], out=offsets[1:])
    storage = np.empty(int(offsets[-1]), dtype=np.uint8)
    # One pass; no intermediate bytearray and no second copy into frombuffer.
    for i, b in enumerate(blobs):
        storage[offsets[i]:offsets[i + 1]] = np.frombuffer(b, dtype=np.uint8)
    return torch.from_numpy(offsets), torch.from_numpy(storage)


class FastSharedList:
    def __init__(self, items):
        self._index, self._storage = _pack(items)
        self._length = int(self._index.numel() - 1)
        self._np = None

    def _views(self):
        # Created lazily *in the reading process*: after unpickling in a spawn
        # worker the tensors point at the shared storage, and .numpy() is a
        # zero-copy view onto it.
        if self._np is None:
            self._np = (self._index.numpy(), self._storage.numpy())
        return self._np

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if i < 0:
            i += self._length
        idx, st = self._views()
        return pickle.loads(memoryview(st[idx[i]:idx[i + 1]]))

    def __getstate__(self):
        return {"_index": self._index, "_storage": self._storage, "_length": self._length}

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._np = None


class SharedArray:
    """Fixed-width typed column, shared via a torch tensor, read via numpy."""

    def __init__(self, array):
        array = np.ascontiguousarray(array)
        self._dtype, self._shape = array.dtype.str, array.shape
        self._storage = torch.from_numpy(array.view(np.uint8).reshape(-1))
        self._np = None

    def _view(self):
        if self._np is None:
            self._np = self._storage.numpy().view(np.dtype(self._dtype)).reshape(self._shape)
        return self._np

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, i):
        return self._view()[i]

    def __getstate__(self):
        return {"_dtype": self._dtype, "_shape": self._shape, "_storage": self._storage}

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._np = None


def _digest(key_blob):
    return int.from_bytes(hashlib.blake2b(key_blob, digest_size=8).digest(), "little", signed=True)


class FastSharedDict:
    def __init__(self, mapping):
        keys = list(mapping)
        kblobs = [pickle.dumps(k, protocol=5) for k in keys]
        digests = np.fromiter((_digest(b) for b in kblobs), dtype=np.int64, count=len(kblobs))
        order = np.argsort(digests, kind="stable")
        self._digests = torch.from_numpy(np.ascontiguousarray(digests[order]))
        self._kindex, self._kstorage = _pack(keys)
        self._vindex, self._vstorage = _pack(mapping[k] for k in keys)
        self._perm = torch.from_numpy(np.ascontiguousarray(order.astype(np.int64)))
        self._length = len(keys)
        self._np = None

    def _views(self):
        if self._np is None:
            self._np = tuple(t.numpy() for t in
                             (self._digests, self._perm, self._kindex, self._kstorage,
                              self._vindex, self._vstorage))
        return self._np

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        dg, perm, ki, ks, vi, vs = self._views()
        kb = pickle.dumps(key, protocol=5)
        h = _digest(kb)
        lo = int(np.searchsorted(dg, h, side="left"))
        hi = int(np.searchsorted(dg, h, side="right"))
        for pos in range(lo, hi):  # digest collisions are resolved by comparing keys
            j = int(perm[pos])
            if pickle.loads(memoryview(ks[ki[j]:ki[j + 1]])) == key:
                return pickle.loads(memoryview(vs[vi[j]:vi[j + 1]]))
        raise KeyError(key)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_np"] = None
        return d


def timeit(fn, idx, reps=1):
    t = time.perf_counter()
    for _ in range(reps):
        for i in idx:
            fn(i)
    return (time.perf_counter() - t) / (len(idx) * reps) * 1e6


if __name__ == "__main__":
    N = int(os.environ.get("N", "3000000"))
    items = [(f"/data/images/{i:09d}.jpg", i) for i in range(N)]
    rng = np.random.default_rng(0)
    idx = [int(i) for i in rng.integers(0, N, 50_000)]

    print(f"N={N} random __getitem__, us/item (lower is better)")
    print(f"  plain list                 {timeit(items.__getitem__, idx):6.2f}")
    paths = np.array([p for p, _ in items], dtype="S32")
    labels = np.arange(N, dtype=np.int64)
    print(f"  numpy S32 (path only)      {timeit(paths.__getitem__, idx):6.2f}")
    t = time.perf_counter(); pr = SharedList(items); tb = time.perf_counter() - t
    print(f"  PR SharedList              {timeit(pr.__getitem__, idx):6.2f}   build {tb:.1f}s")
    del pr
    t = time.perf_counter(); fl = FastSharedList(items); tb = time.perf_counter() - t
    print(f"  FastSharedList             {timeit(fl.__getitem__, idx):6.2f}   build {tb:.1f}s")
    sa = SharedArray(paths); sl = SharedArray(labels)
    print(f"  SharedArray S32 (path)     {timeit(sa.__getitem__, idx):6.2f}")
    print(f"  SharedArray int64 (label)  {timeit(sl.__getitem__, idx):6.2f}")
    assert fl[12345] == items[12345] and sa[12345] == paths[12345]

    for M in (10_000, 100_000, 1_000_000):
        d = {f"/data/images/{i:09d}.jpg": i for i in range(M)}
        keys = random.Random(0).sample(list(d), 200)
        t = time.perf_counter(); fd = FastSharedDict(d); tb = time.perf_counter() - t
        assert all(fd[k] == d[k] for k in keys)
        line = (f"  dict M={M:8d}: dict {timeit(d.__getitem__, keys, 20):6.2f}   "
                f"FastSharedDict {timeit(fd.__getitem__, keys, 20):6.2f}   build {tb:.1f}s")
        if M <= 100_000:
            sd = SharedDict(d)
            line += f"   PR SharedDict {timeit(sd.__getitem__, keys[:20]):10.1f}"
        print(line, flush=True)
