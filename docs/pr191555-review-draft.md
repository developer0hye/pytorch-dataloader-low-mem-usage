# Draft: review comment for pytorch/pytorch#191555

Status: DRAFT, not posted. Numbers from `results/2026-09-15-a6000-matrix.log`.
Post as a PR conversation comment (not inline) after the owner approves.

---

Thanks for pushing on #13246. I ran the PR module (head `66e35fcd`, vendored
verbatim) through an independent benchmark at a larger scale than the RFC's
60k-row set: 3M `(path, label)` tuples, 4 workers, one shuffled epoch,
Python 3.12.3 / torch 2.12.0, Linux, measuring per-worker `Private_Dirty`
and `Pss` from `/proc/<pid>/smaps_rollup` plus the parent's `ru_maxrss`.
Harness and raw logs: <REPO_URL>. Four things I think are worth folding into
the RFC before the design discussion.

**1. Under `fork`, `SharedList` gives the same worker memory as any packed
buffer, and the mechanism is not shared memory.**

| repr (fork) | worker `Private_Dirty` start -> end | family Pss (parent + 4 workers) |
|---|---|---|
| plain `list` | 19 -> 512 MB | 2976 MB |
| numpy `S32` + `int64` arrays | 6 -> 7 MB | 608 MB |
| `SharedList` | 6 -> 7 MB | 935 MB |
| `to_shared_dataset` | 6 -> 7 MB | 955 MB |

With `fork` the dataset is inherited, not pickled, so the `ForkingPickler`
storage-sharing path in RFC §4.2 never runs; the `torch.frombuffer` tensor
sits in ordinary parent pages that simply stay clean because no refcounted
object lives inside them. That is the same reason numpy works. The RFC's
"placed into shared memory" wording (also in the docstrings) is only true for
`forkserver`/`spawn`, and I think the design doc should say so explicitly.

**2. Under `forkserver`/`spawn`, `SharedList` is the only representation
that really shares, and that is the PR's strongest result. It is missing
from the RFC benchmarks.**

| repr (forkserver) | worker `Private_Dirty` | family Pss | worker startup |
|---|---|---|---|
| plain `list` | 805 MB (from first batch) | 4210 MB | 13.9 s |
| numpy arrays | 391 MB | 2209 MB | 6.0 s |
| `SharedList` | 277 MB (= interpreter + torch import) | 2010 MB | 1.3 s |

`spawn` is identical within noise. Since Python 3.14 defaults to
`forkserver` on POSIX, this is the case most new users will hit, and the
"numpy is enough" advice in the docs stops working there. Worth leading with.

**3. Costs that should be in the design doc.**

- `__getitem__`: 6.1 us vs 0.6 us for a plain list (two `.item()` calls, a
  slice, `.numpy().tobytes()`, `pickle.loads`). For a metadata-only dataset
  this took the epoch from 5.0 s to 8.0 s at 4 workers. Real datasets
  amortize it behind decoding, but it is not free.
- Construction peak: 1728 MB vs 1018 MB for the plain list (1.7x).
  `to_shared_dataset` pickles the entire list once just to test
  `threshold_bytes`, then `SharedList.__init__` pickles every item again,
  builds a `bytearray`, and `torch.frombuffer(bytearray(raw))` copies it a
  third time. Suggestions: estimate size from a sample instead of a full
  pickle; pass `raw` to `frombuffer` directly (it keeps a reference);
  consider `pickle.dumps(item, protocol=5)`.
- Parent RSS after conversion stays at 936 MB vs 611 MB for numpy; part of
  that is pymalloc retaining the freed source list, which users will
  attribute to the container.

**4. `SharedDict` auto-conversion is a performance hazard as written.**

`SharedDict.__getitem__` scans keys linearly with a `pickle.loads` per key.
`to_shared_dataset` converts any dict whose pickle is >= 1 MiB; a 100k-entry
`path -> label` map is 3.1 MiB, so it qualifies:

| dict size | `dict` lookup | `SharedDict` lookup |
|---|---|---|
| 10,000 (not converted) | 0.11 us | 31 ms |
| 100,000 (converted) | 0.40 us | 281 ms |

End to end: a dataset doing one label lookup per sample against that dict
(2000 samples, uniformly random keys, 4 workers) went from a 0.10 s epoch to
144.8 s after `to_shared_dataset`, i.e. 1,424x slower from one flag. I would
either give `SharedDict` a hashed index or exclude dicts from
`to_shared_dataset` until it has one.

**5. Both costs are fixable without changing the storage layout.**

I prototyped two things on top of the PR's exact layout (uint8 storage +
int64 offsets in torch tensors, so `ForkingPickler` sharing under
`forkserver`/`spawn` is unchanged):

- Read through lazily created numpy views (`self._index.numpy()`,
  `self._storage.numpy()`, created in the reading process so they alias the
  shared storage) and call `pickle.loads(memoryview(storage[a:b]))`. No
  per-access `.item()`, no tensor slice, no `tobytes()` copy.
  `__getitem__`: 5.40 us -> 1.01 us (plain list 0.27 us). Epoch at 4 workers:
  8.0 s -> 6.0 s (list 5.0 s). Worker memory identical to `SharedList` in
  every start method; construction peak 1728 -> 1477 MB by packing straight
  into a preallocated numpy buffer instead of `bytearray` + `frombuffer(bytearray(raw))`.
- For `SharedDict`, store a sorted int64 digest of each pickled key
  (blake2b-8; `hash()` cannot be used because `str` hashing is salted per
  process and spawn/forkserver workers do not share the salt) plus a
  permutation, and look up with `np.searchsorted`. Lookup is a flat ~4.8 us
  at 10k, 100k and 1M entries, versus 28 ms / 259 ms for the current scan.
- A typed fixed-width column (`SharedArray`: numpy dtype view over a torch
  storage) is worth offering next to `SharedTensor`: 0.21 us per access
  (faster than a `list`), and under `forkserver` it gives the lowest family
  Pss of anything I measured (1755 MB vs 2010 MB for `SharedList`, 4210 MB
  for `list`).

Code and numbers: <REPO_URL>/blob/main/bench/fastpath.py. I am happy to send
these as a follow-up commit on your branch if you want them.

Smaller notes: `use_shared_memory=True` mutates the user's dataset object
at `DataLoader.__init__` (visible to any other loader sharing it, breaks
`isinstance(x, list)` and identity, and `sample = ds.items[i]; sample.x = ...`
now writes to a temporary); the RFC lists "snapshot semantics" but this
aliasing/mutation surface deserves its own paragraph. Also
`_tensor_to_bytes` still goes through `.numpy()`, which the PR description
says was removed.

Happy to also contribute the harness as a `benchmarks/` script or turn the
forkserver comparison into a test if that helps the RFC move.
