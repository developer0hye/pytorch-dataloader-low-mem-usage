"""SharedDict lookup cost vs dict size (PR pytorch/pytorch#191555).

`to_shared_dataset()` converts any dict attribute whose pickled size is
>= 1 MiB into a SharedDict, and SharedDict.__getitem__ deserializes keys
linearly until it finds a match.  A label map with 1e5 entries therefore
turns one `self.labels[path]` into up to 1e5 pickle.loads calls per sample.
This script measures that so the review can quote numbers, not adjectives.
"""
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from pr_shared_container import SharedDict  # noqa: E402

random.seed(0)
for m in (1_000, 10_000, 100_000):
    d = {f"/data/images/{i:09d}.jpg": i for i in range(m)}
    pickled_mb = len(pickle.dumps(d)) / 2**20
    t = time.perf_counter()
    sd = SharedDict(d)
    build_s = time.perf_counter() - t
    keys = random.sample(list(d), 50)

    t = time.perf_counter()
    for k in keys:
        d[k]
    plain_us = (time.perf_counter() - t) / len(keys) * 1e6

    t = time.perf_counter()
    for k in keys:
        sd[k]
    shared_us = (time.perf_counter() - t) / len(keys) * 1e6

    auto = "YES" if pickled_mb >= 1.0 else "no"
    print(f"M={m:7d} pickled={pickled_mb:6.2f}MiB auto-converted(>=1MiB)={auto:3s} "
          f"build={build_s:5.2f}s  dict lookup {plain_us:7.2f}us  "
          f"SharedDict lookup {shared_us:12.1f}us  ({shared_us / plain_us:,.0f}x)", flush=True)
