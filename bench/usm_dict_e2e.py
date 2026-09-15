"""End-to-end cost of `use_shared_memory=True` auto-converting a label dict.

A common dataset shape: a list of paths plus a dict mapping path -> label,
looked up once per sample.  PR pytorch/pytorch#191555's `to_shared_dataset`
converts any dict whose pickle is >= 1 MiB into a SharedDict with linear
key scan, so the per-sample lookup becomes O(len(dict)) pickle.loads calls.

We keep the sample count small (default 2000) because the converted run is
slow by construction; the point is the ratio, not the absolute time.
"""
import argparse
import os
import random
import sys
import time

from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from pr_shared_container import to_shared_dataset  # noqa: E402


class PathLabelDS(Dataset):
    def __init__(self, n_samples, n_labels):
        # Samples reference keys spread uniformly over the whole dict.  A first
        # version used the first n_samples keys and measured only a 26x slowdown,
        # because SharedDict's linear scan found them near the front; real
        # datasets look up arbitrary keys, so the expected scan is len(dict)/2.
        keys = [f"/data/images/{i:09d}.jpg" for i in range(n_labels)]
        rng = random.Random(0)
        self.paths = [keys[i] for i in rng.sample(range(n_labels), n_samples)]
        self.label_of = {k: i % 1000 for i, k in enumerate(keys)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return len(p), self.label_of[p]


def run(ds, workers, label):
    dl = DataLoader(ds, batch_size=64, num_workers=workers, shuffle=False)
    t = time.perf_counter()
    for _ in dl:
        pass
    s = time.perf_counter() - t
    print(f"{label:34s} epoch {s:8.2f}s  ({len(ds) / s:8.0f} samples/s)", flush=True)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--labels", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    print(f"samples={a.samples} labels={a.labels} workers={a.workers}", flush=True)
    base = run(PathLabelDS(a.samples, a.labels), a.workers, "plain dict")
    ds = to_shared_dataset(PathLabelDS(a.samples, a.labels))
    print(f"label_of converted to: {type(ds.label_of).__name__}", flush=True)
    conv = run(ds, a.workers, "use_shared_memory / to_shared_dataset")
    print(f"slowdown {conv / base:,.0f}x", flush=True)
