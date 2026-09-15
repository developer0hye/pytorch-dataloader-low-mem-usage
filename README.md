# pytorch-dataloader-low-mem-usage

Measurements, reproduction scripts and a small diagnostic for the PyTorch
`DataLoader` worker-memory blow-up
([pytorch/pytorch#13246](https://github.com/pytorch/pytorch/issues/13246)):
with `num_workers > 0` every worker ends up holding a private copy of the
dataset metadata, so RAM grows as `num_workers x dataset_size`.

Everything below was measured, not estimated. Raw logs are in `results/`.

## Why read this

You are probably here because one of these happened:

| Symptom | What to check | Section |
|---|---|---|
| RAM climbs for the first epoch, then plateaus at roughly `N x parent` | per-worker `Private_Dirty` growth, not RSS | [What actually happens](#what-actually-happens) |
| `top` says 8 GB but the box is not swapping | RSS double-counts copy-on-write pages; use Pss | [Measuring it right](#measuring-it-right) |
| Memory is large from the very first batch, no growth | you are on `spawn`/`forkserver` (Python 3.14 default on POSIX) and the dataset is pickled per worker | [Start method matters](#start-method-matters) |
| You are evaluating PR #191555 `SharedList` / `use_shared_memory=True` | it is a different trade-off under `fork` vs `forkserver` | [PR #191555 measured](#pr-191555-measured) |
| You want the memory win without a 10x slower `__getitem__` | read through numpy views, or use a typed column | [Making it fast](#making-it-fast) |

## What actually happens

A forked worker shares the parent's pages copy-on-write. CPython keeps the
reference count *inside* every object, so merely reading a `str` or `tuple`
(`Py_INCREF`) is a write to its page and the kernel copies it. Over one
shuffled epoch a worker touches nearly every metadata page, so it ends up
with a private copy of the whole metadata heap.

Cause isolation, 3M `(path, label)` tuples, 4 workers, one epoch, `fork`
(`bench/cause_isolation.py`, Python 3.12.3, torch 2.12.0, Linux):

| Variant | Worker `Private_Dirty` start -> end | 4 workers total |
|---|---|---|
| `list` of `tuple(str, int)` | 23 -> **515 MB** | 2062 MB |
| same + `gc.freeze()` before fork | 23 -> **515 MB** | 2061 MB |
| same, every object made immortal (ctypes, experiment only) | 11 -> 12 MB | 46 MB |
| metadata in two numpy arrays (`S32`, `int64`) | 9 -> 10 MB | 40 MB |

Read: `gc.freeze()` does nothing here. Making refcount writes disappear
(immortal objects) removes the growth entirely, so the refcount write is the
cause in this workload. Packing the metadata into buffers without per-item
Python objects gets the same result without touching the interpreter.
The immortal variant is a demonstration of the mechanism, not something to
ship: it pokes `ob_refcnt` through `ctypes`, is version specific, and leaks
the objects forever.

## Measuring it right

Sum of worker RSS is the wrong number. After fork every untouched
copy-on-write page is counted again in each worker. In the same run the
kernel's honest figures (`/proc/<pid>/smaps_rollup`) were:

| Metric (per worker, `list`, after one epoch) | Value |
|---|---|
| `Rss` (what `ps`/`top` sum) | 826 MB |
| `Pss` (shared pages divided among sharers) | 574 MB |
| `Private_Dirty` (pages this worker actually owns) | 512 MB |

`dlmem.py` in this repo reads these for a live `DataLoader` and reports
per-worker private growth and the process-family Pss (parent + workers),
which is the real bill:

```python
from dlmem import worker_memory, format_report

dl = DataLoader(ds, num_workers=4, persistent_workers=True)
it = iter(dl); next(it)          # workers exist after the first batch
before = worker_memory(dl)
for batch in it: ...
print(format_report(before, worker_memory(dl)))
```

```
workers=2
  worker Private_Dirty       18 ->      86 MiB   (+68 per worker, +136 total)
  worker Pss                137 ->     181 MiB
  family Pss                526 ->     644 MiB   (naive RSS sum 1342 MiB)
```

`Private_Dirty` growth is evidence consistent with copy-on-write
duplication, not proof; decoded samples and allocator retention land there
too. Compare representations, do not read one number as a verdict.

## Start method matters

Python 3.14 made `forkserver` the default on POSIX. Under `forkserver` and
`spawn` the dataset is pickled into every worker, so there is no
copy-on-write to talk about: the copy is there from the first batch, and a
numpy-packed dataset is copied too.

`bench/cow_bench.py`, 3M items, 4 workers, batch 512, one shuffled epoch:

| Representation | Start method | Worker `Private_Dirty` start -> end | Family Pss (parent + 4 workers) | Worker startup |
|---|---|---|---|---|
| `list` | fork | 19 -> 512 MB | 2976 MB | 0.2 s |
| numpy | fork | 6 -> 7 MB | **608 MB** | 0.2 s |
| `list` | forkserver | 805 -> 805 MB | 4210 MB | 13.9 s |
| numpy | forkserver | 391 -> 391 MB | 2209 MB | 6.0 s |
| `list` | spawn | 806 -> 806 MB | 4216 MB | 13.2 s |
| numpy | spawn | 392 -> 392 MB | 2214 MB | 5.8 s |

Under `forkserver`/`spawn` about 270 MB of each worker's private memory is
the interpreter plus `import torch`; the rest is the pickled dataset.
The 13.9 s startup is the time to pickle 3M tuples into four workers.

## PR #191555 measured

[pytorch/pytorch#191555](https://github.com/pytorch/pytorch/pull/191555)
proposes `SharedList` / `SharedDict` (items pickled into one `torch.uint8`
tensor plus offsets), `to_shared_dataset()` (auto-converts large `list`/`dict`
attributes) and `DataLoader(use_shared_memory=True)`. We benchmarked the PR
module verbatim (`bench/third_party/`, head `66e35fcd`) against the same
3M-item dataset.

| Representation | Start method | Worker private start -> end | Family Pss | Parent RSS after build | Parent peak during build | `__getitem__` | Epoch |
|---|---|---|---|---|---|---|---|
| `list` | fork | 19 -> 512 MB | 2976 MB | 1025 MB | 1018 MB | 0.62 us | 5.0 s |
| numpy | fork | 6 -> 7 MB | 608 MB | 611 MB | 833 MB | 0.43 us | 4.5 s |
| `SharedList` | fork | 6 -> 7 MB | 935 MB | 936 MB | 1728 MB | 6.14 us | 8.0 s |
| `to_shared_dataset` | fork | 6 -> 7 MB | 955 MB | 956 MB | 1824 MB | 6.02 us | 7.8 s |
| `list` | forkserver | 805 -> 805 MB | 4210 MB | 1025 MB | 1018 MB | 0.77 us | 4.5 s |
| numpy | forkserver | 391 -> 391 MB | 2209 MB | 611 MB | 834 MB | 0.45 us | 5.3 s |
| `SharedList` | forkserver | 276 -> 277 MB | 2010 MB | 937 MB | 1728 MB | 5.73 us | 7.7 s |
| `to_shared_dataset` | forkserver | 276 -> 277 MB | 2030 MB | 956 MB | 1828 MB | 5.32 us | 7.5 s |

`spawn` numbers match `forkserver` within noise (see `results/`).

What the table says:

- **Under `fork`, `SharedList` behaves exactly like numpy packing** for worker
  memory (7 MB private). Nothing is in shared memory at that point; the
  `torch.frombuffer` storage lives in ordinary parent pages that stay clean
  because there are no refcounted objects inside them. The PR's RFC describes
  the `ForkingPickler` fd-sharing path, which is not exercised under `fork`
  (the dataset is inherited, not pickled).
- **Under `forkserver`/`spawn`, `SharedList` is the only representation that
  actually shares.** Worker private memory drops to the interpreter baseline
  (277 MB vs 805 MB for `list`, 391 MB for numpy) and worker startup drops from
  13.9 s to 1.3 s because the 3M-item payload travels as a storage handle
  instead of a pickle. With Python 3.14 defaulting to `forkserver`, this is
  the strongest case for the PR and it is not in the PR's own benchmarks.
- **Costs**: `__getitem__` is about 10x slower than a plain list (two
  `tensor.item()` calls, a slice, `.numpy().tobytes()`, `pickle.loads`), which
  turned a 5.0 s metadata-only epoch into 8.0 s at 4 workers. Construction
  peaks at 1.7x the plain-list parent (1728 MB vs 1018 MB): `to_shared_dataset`
  pickles the whole list once just to check the size threshold, then
  `SharedList` pickles every item again, concatenates into a `bytearray`, and
  `torch.frombuffer(bytearray(raw))` copies that once more. The parent also
  ends up larger than with numpy (936 MB vs 611 MB); part of that is
  allocator retention from the deleted source list.

### `SharedDict` auto-conversion

`SharedDict.__getitem__` deserializes keys linearly until it finds a match.
`to_shared_dataset` converts any dict whose pickle is at least 1 MiB, which a
100k-entry `path -> label` map crosses (3.1 MiB). `bench/shareddict_lookup.py`:

| dict size | pickled | auto-converted | `dict` lookup | `SharedDict` lookup | ratio |
|---|---|---|---|---|---|
| 1,000 | 0.03 MiB | no | 0.07 us | 3.0 ms | 42,000x |
| 10,000 | 0.31 MiB | no | 0.11 us | 31 ms | 277,000x |
| 100,000 | 3.12 MiB | **yes** | 0.40 us | **281 ms** | 711,000x |

End to end (`bench/usm_dict_e2e.py`, 2000 samples looking up uniformly
random keys in a 100k-entry dict, 4 workers): the plain-dict epoch took
0.10 s, the `to_shared_dataset` epoch took 144.8 s, a **1,424x** slowdown from
one flag. That is the one part of the PR we think must change before the flag
is safe to recommend: either a hashed index for `SharedDict` or excluding
dicts from auto-conversion. (A first version of the script looked up only
the first 2000 keys and saw 26x; linear scans reward the front of the dict.)

## Making it fast

"Not slower" is a requirement, not a nice-to-have: nobody keeps a memory fix
that costs 60% of epoch time. The overhead in the PR's `__getitem__` is not
inherent to the layout. `bench/fastpath.py` keeps the PR's storage design
(uint8 storage + int64 offsets in `torch` tensors, so `ForkingPickler` still
shares them under `forkserver`/`spawn`) and changes only how a worker reads:
lazily created numpy views instead of `tensor.item()` and tensor slicing,
and `pickle.loads(memoryview(...))` instead of `.numpy().tobytes()`.

Single-process random `__getitem__`, 3M items:

| Container | us/item | vs plain list |
|---|---|---|
| plain `list` | 0.27 | 1.0x |
| numpy `S32` | 0.18 | 0.7x |
| PR `SharedList` | 5.40 | 20x |
| `FastSharedList` (same layout, numpy views) | 1.01 | 3.7x |
| `SharedArray` `S32` / `int64` (typed column, shared via torch storage) | 0.21 | 0.8x |

| dict size | `dict` | `FastSharedDict` (sorted blake2b digest + `searchsorted`) | PR `SharedDict` |
|---|---|---|---|
| 10k | 0.03 us | 4.9 us | 28 ms |
| 100k | 0.04 us | 4.8 us | 259 ms |
| 1M | 0.04 us | 4.8 us | not run |

`FastSharedDict` uses a deterministic digest of the pickled key rather than
`hash()` because CPython salts `str` hashes per process, and `spawn`/
`forkserver` workers do not share the parent's salt.

Worker memory with the fast paths (`bench/cow_bench.py`, same 3M-item setup):

| Representation | Start method | Worker private start -> end | Family Pss | Build peak | `__getitem__` | Epoch |
|---|---|---|---|---|---|---|
| `FastSharedList` | fork | 6 -> 7 MB | 932 MB | 1477 MB | 1.05 us | 6.0 s |
| `SharedArray` | fork | 6 -> 7 MB | **610 MB** | 834 MB | 0.56 us | 5.1 s |
| `FastSharedList` | forkserver | 277 -> 277 MB | 2008 MB | 1476 MB | 1.17 us | 6.1 s |
| `SharedArray` | forkserver | 276 -> 277 MB | **1755 MB** | 834 MB | 0.55 us | 5.1 s |

Compare with the earlier table: `SharedArray` matches numpy under `fork`
(610 vs 608 MB) and beats every other representation under `forkserver`
(1755 MB vs 2010 for `SharedList`, 2209 for numpy, 4210 for `list`) while
being as fast as a plain list. `FastSharedList` keeps the PR's "any picklable
item" generality and cuts its epoch overhead from +60% to +20%, with a lower
construction peak (1477 vs 1728 MB). Both are prototypes in `bench/`, not a
library.

## Reproduce

Linux only (reads `/proc/<pid>/smaps_rollup`). Any recent torch + numpy.

```bash
python -m venv .venv && .venv/bin/pip install torch numpy
PY=.venv/bin/python bench/run_matrix.sh          # 18 runs (6 representations x 3 start methods), ~5 min
.venv/bin/python bench/cause_isolation.py list   # list | gcfreeze | immortal | numpy
.venv/bin/python bench/shareddict_lookup.py
.venv/bin/python bench/usm_dict_e2e.py --samples 2000 --labels 100000   # ~2.5 min, slow by design
.venv/bin/python bench/fastpath.py
```

Runs print one progress line per unit with elapsed time and ETA.

Measured on: 2x AMD EPYC 9554 (256 threads), 1 TB RAM, Ubuntu, Python 3.12.3,
torch 2.12.0+cu130, numpy 2.4.6. CPU only.

## What this repo is not

It does not ship a transparent fix. A `DataLoader`-side fix that needs no
user code change would have to either immortalize the dataset graph before
fork (no safe public API for already-shared objects; CPython 3.15's
`PyUnstable_SetImmortal` is creator-side only, and `fork` is no longer the
default anyway) or run workers as threads (a different concurrency contract:
shared dataset state, global RNGs, no timeout kill). Both are documented in
`docs/reviews/`. What works today is choosing the metadata representation,
and this repo exists to make that choice with numbers.

## More

- Yuxin Wu, [Demystify RAM Usage in Multi-Process Data Loaders](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/), the original write-up of copy-on-read and packed lists (detectron2 `TorchSerializedList`).
- PEP 683, [Immortal Objects](https://peps.python.org/pep-0683/), the motivation section is this exact pre-fork problem.
- `docs/reviews/2026-09-15-codex-design-cross-review.md`, an adversarial review of the "fix it inside DataLoader" designs, including every write source besides refcounts that can dirty a shared page.

## One line

Worker memory is multiplied by refcount writes into copy-on-write pages;
pack metadata into buffers with no per-item Python objects, measure with
Pss, and remember that on Python 3.14's `forkserver` default only
explicitly shared storage is shared at all.
