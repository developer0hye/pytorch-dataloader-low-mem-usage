# Summary Verdict

**Partially sound: the measurements establish a strong mechanism and useful remedies, but the proposed package cannot honestly promise a transparent fix for arbitrary datasets.** Explicit packed metadata is the strongest near-term remedy; threads are useful under a different concurrency contract; diagnostics can report growth but cannot identify COW from `Private_Dirty` alone. Arbitrary graph immortalization through `ctypes` should remain a research experiment, because version checks and warnings do not establish runtime correctness. The novelty and upstream strategy also need updating: MONAI already provides thread workers, PyTorch has an open overlapping shared-container proposal, and Python 3.15 has accepted an unstable immortalization C API. These developments support a narrower package focused on explicit contracts and measured benefits. ([MONAI](https://monai.readthedocs.io/en/0.9.0/data.html#threaddataloader), [PyTorch PR #191555](https://github.com/pytorch/pytorch/pull/191555), [Python 3.15 API](https://docs.python.org/3.15/c-api/object.html#c.PyUnstable_SetImmortal))

# Q1 Root Cause Completeness

## The central mechanism is correct; its wording needs tightening

CPython’s `Py_INCREF` and `Py_DECREF` modify object headers unless the applicable immortal fast path suppresses the write. Fork COW operates on **pages**, so accessing one object can privately copy neighboring objects that the worker never accessed. Conversely, a Python operation does not necessarily increment every reachable object: returning a tuple and subsequently traversing its elements are different operations. DataLoader’s fetching, collation and IPC serialization can all contribute reference traffic. ([CPython 3.12.3 `Include/object.h`](https://github.com/python/cpython/blob/v3.12.3/Include/object.h), [PyTorch collation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/collate.py))

For an ordinary map-style shuffled epoch, workers generally receive different indices, not every sample in every worker. Nevertheless, randomly distributed accesses can touch nearly all densely populated object-header pages in each worker. That reconciles disjoint sample assignment with extensive per-worker duplication; it is a page-layout explanation, not evidence that every worker processed every record. ([DataLoader sampling documentation](https://docs.pytorch.org/docs/main/data.html#single-and-multi-process-data-loading))

## Other write sources

No finite list can enumerate every write performed by arbitrary Python callbacks or native extensions. The following covers the relevant interpreter mechanisms and separates them from application-specific mutations.

| Write source | Does it fire for read-only list-of-`(str, int)` access? | Consequence for immortalization |
|---|---|---|
| **Cyclic GC bookkeeping** | Potentially during collections triggered by temporary allocations. In 3.12, `update_refs`, `subtract_refs` and generation-list operations in `Modules/gcmodule.c` modify GC metadata. Already-untracked atomic tuples need less GC involvement. | Immortal headers alone are not equivalent to removing objects from GC tracking. Your successful treatment also included `gc.freeze()`, which suppresses this traversal for frozen objects. |
| **Dictionary mutation/version tags** | Ordinary lookup does **not** update `ma_version_tag`. Insertion, replacement, deletion and watcher operations can. Accessing `self.samples` therefore does not imply a version write on every lookup. | Not a bulk residual cause here. Cache-populating datasets can trigger it. In 3.14 the field is `_ma_watcher_tag`; treating `ma_version_tag` as a stable cross-version field is wrong. |
| **Dictionary-key version caching** | Specialization may lazily assign `PyDictKeysObject.dk_version` through `_PyDictKeys_GetVersionForCurrentState`. This differs from dictionary mutation versioning. | Survives object immortalization, but normally affects a few namespace/key-table pages rather than every record. |
| **String hash caching** | Indexing or unpacking the tuple does not hash the path. A first `hash(path)`, set insertion or dictionary lookup using that path can. | `unicode_hash` writes the cached hash even if the string is immortal. Hashing every previously unhashed path can restore dataset-sized COW. |
| **Unicode representation caching** | ASCII paths usually avoid the relevant extra representation. A native consumer calling `PyUnicode_AsUTF8AndSize` on a non-ASCII string can materialize a UTF-8 cache. Ordinary UTF-8 encoding APIs do not all have identical caching behavior. | Another per-string write that immortality does not eliminate; workload-dependent. |
| **List resize/reallocation** | No resize occurs when indexing the source list. Batch construction creates and grows other lists. `list_resize` changes the item buffer, size and allocation state when an actual mutation occurs. | Source mutation remains a write. Temporary allocations can dirty neighboring inherited pages. |
| **Tuple resize and hash caching** | Tuple reads do not resize tuples. `_PyTuple_Resize` is a restricted construction operation, not normal read behavior. Additionally, 3.14’s `tuple_hash` caches into `PyTupleObject.ob_hash`. | No residual source for the stated 3.12 indexing workload; first hashing of inherited tuples is an additional 3.14 counterexample. |
| **Small-integer cache** | Returning cached small integers does not incur normal header refcount updates in 3.12+. Labels outside the cached range remain ordinary objects. | Cached integers are already covered by CPython’s immortality. It is wrong to generalize this to all `int` objects. |
| **Interned strings** | Reading an existing interned string does not automatically mutate the intern table. Interning a new string, or removing a mortal interned string at deallocation, does. | Interning is separate from merely accessing strings. Ordinary runtime paths should not be assumed interned, and interned does not universally mean immortal across builds. |
| **Allocator and free-list bookkeeping** | Yes: fetching, collation and serialization allocate and release temporary objects. Pymalloc updates pool counters, free-block links and arena/pool lists; type free lists also update linkage. | Immortalizing live dataset objects does not isolate their pages from allocator writes to neighboring free blocks. Allocation history and fragmentation matter. |
| **Shared-key instance dictionaries** | Existing attribute reads can use the values array without materializing `__dict__`. Reading `obj.__dict__`, certain generic paths, or attribute writes can materialize a dictionary. Divergence may modify shared keys or cause conversion to combined storage; not every new attribute necessarily forces combination. | Can scale with the number of record instances. Internal key-table counts such as `dk_refcnt` are not `PyObject.ob_refcnt` and are not disabled by immortalizing the instance. |
| **Type lookup/version caches** | Attribute lookup can populate type caches and assign `tp_version_tag`, notably through `_PyType_Lookup` and `assign_version_tag`. | Usually bounded by the number of types, not records. Immortal types can still receive internal cache writes. |
| **Adaptive bytecode/inline caches** | Yes, potentially even for plain `self.samples[index]`: `LOAD_ATTR` and subscription operations are specialization candidates. | Writes affect executing code/cache pages, ordinarily a small footprint. Saying these caches are irrelevant to plain attribute access is too strong. |
| **Tier-two executors/JIT state** | Not a mechanism in the stated 3.12 baseline. Later tier-two execution can maintain executor state when enabled. | `_PyExecutorObject` is not another name for the original 3.11 inline cache. It introduces additional runtime state, not an inevitable per-record mutation. |
| **Interpreter, library and OS-runtime state** | Yes: evaluation stacks, iterator positions, exception state, RNG state, queue bookkeeping, locks, lazy imports and native allocation caches change during execution. | Dataset immortality cannot eliminate these writes or new private allocations. They explain a residual floor without refuting the main result. |

Sources for the table: [3.12 GC implementation](https://github.com/python/cpython/blob/v3.12.3/Modules/gcmodule.c), [GC freeze contract](https://docs.python.org/3.14/library/gc.html#gc.freeze), [3.12 dictionaries](https://github.com/python/cpython/blob/v3.12.3/Objects/dictobject.c), [3.14 dictionary layout](https://github.com/python/cpython/blob/v3.14.0/Include/cpython/dictobject.h), [Unicode implementation](https://github.com/python/cpython/blob/v3.12.3/Objects/unicodeobject.c), [3.14 tuples](https://github.com/python/cpython/blob/v3.14.0/Objects/tupleobject.c), [pymalloc](https://github.com/python/cpython/blob/v3.12.3/Objects/obmalloc.c), [type caches](https://github.com/python/cpython/blob/v3.12.3/Objects/typeobject.c), [PEP 659](https://peps.python.org/pep-0659/), [executor structures](https://github.com/python/cpython/blob/3.14/Include/internal/pycore_optimizer.h).

**Field-name correction:** Python 3 `str` caches its hash in `PyASCIIObject.hash`; `ob_shash` is associated with `bytes`, not the `str` layout in this experiment. ([Unicode layout](https://github.com/python/cpython/blob/v3.12.3/Include/cpython/unicodeobject.h))

## Reconciliation with the measurements

Accepting the results, the combined immortalization-and-freezing treatment leaves approximately 1 MB of growth per worker. Thus, **the other mechanisms do not produce substantial additional private memory in this workload and observation window**. Their existence is not a valid objection to the measured effect.

Three qualifications matter:

- **Approximately 45× is the endpoint ratio**, `2062/46`, not the growth ratio. The rounded per-worker growth changes from approximately 492 MB to approximately 1 MB; that denominator is too coarsely rounded for a precise growth-reduction claim.
- `gc.freeze()` alone being ineffective does not establish that GC is harmless once refcount writes disappear. Different writes can hit the same pages and mask one another. An immortalization-only result was not supplied.
- Generalization depends on operations, not just container names. Read-only dictionaries whose keys were already hashed may behave well. Per-record cached properties, `__dict__` materialization, previously unhashed strings and lazy image decoding can behave very differently. Pillow’s `Image.load()` and serialization paths explicitly materialize image data. ([Pillow `Image.py`](https://github.com/python-pillow/Pillow/blob/main/src/PIL/Image.py))

Finally, `Private_Dirty` is much better than summed RSS for this experiment, but it is **not a COW-event counter**: fresh private allocations contribute too. Total process-family PSS is the complementary measure for actual resident-memory cost. ([Linux `/proc` documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html))

# Q2 Thread Backend

## What breaks or requires a new contract

### 1. Worker-local dataset state

Process workers have separate Python dataset state. Thread workers share the dataset, transforms, caches, open handles and callable objects unless the backend explicitly clones them.

A shallow copy does not isolate nested state. A deep copy restores much of the memory duplication being avoided and may fail on handles or locks. Parent mutations also become immediately visible to workers, and returned Python objects may alias the original dataset instead of crossing a serialization boundary. These are observable semantic changes, even with the GIL enabled. ([DataLoader dataset-replica contract](https://docs.pytorch.org/docs/main/data.html#single-and-multi-process-data-loading))

### 2. `get_worker_info()` and IterableDataset sharding

`torch/utils/data/_utils/worker.py` stores `_worker_info` in a **module global**. Running `_worker_loop` in multiple threads would overwrite the same variable. A replacement needs thread-local, loader-specific worker context; merely defining a new package-local `get_worker_info()` would not fix datasets that imported PyTorch’s function earlier.

IterableDataset examples commonly shard using `worker_info.id`, `num_workers`, or `worker_init_fn` mutations to `worker_info.dataset`. The last approach races on a shared instance. Sharing an iterator is also different from creating separate iterators over one source; the backend must define which model it supports. ([Worker implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/worker.py), [IterableDataset examples](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.IterableDataset))

### 3. RNG seeding and reproducibility

`_BaseDataLoaderIter` derives `_base_seed` using a random `int64` draw with `loader.generator`. `_worker_loop` seeds Python and Torch with `base_seed + worker_id`, and NumPy with `_generate_state(base_seed, worker_id)`.

Those module/default RNGs are shared between threads. Repeatedly reseeding them does not create independent worker streams; it changes other workers’ state and potentially the training thread’s state. RNG locking can prevent corruption while leaving sample-to-random-number assignment scheduling-dependent.

Explicit `random.Random`, NumPy `Generator` and `torch.Generator` instances are a solution only when dataset code uses them. Arbitrary existing transforms do not automatically accept injected generators. Serializing “save RNG state → seed → execute → restore” around all work defeats concurrent execution and remains vulnerable to unrelated threads. ([Seed derivation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/dataloader.py), [Worker seeding](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/worker.py))

### 4. `worker_init_fn` and thread-pool settings

The callback currently executes after worker setup and before fetching. Existing callbacks may mutate dataset replicas, seed libraries, open process-local files or configure process-wide libraries. Preserving the callback signature does not preserve that isolation.

`torch.set_num_threads(1)` also needs backend-specific treatment. ATen’s native backend manages shared intra-op pool configuration; OpenMP has different thread-setting behavior. Calling it independently from each worker is not a portable per-worker resource budget. The pinning thread already calls it, but that does not prove arbitrary thread workers can safely reproduce process initialization. ([ATen native backend](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/ParallelNative.cpp), [OpenMP backend](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/ParallelOpenMP.cpp))

### 5. Pinning and collation

A separate pin-memory stage remains useful. It must preserve device selection, queue backpressure, exception delivery and shutdown behavior, and prevent workers from reusing a buffer while pinning or training still consumes it.

There is a subtle existing assumption: `collate_tensor_fn` treats non-`None` `get_worker_info()` as a reason to allocate shared-memory tensor storage using `_new_shared`. Thread workers do not need that IPC optimization. Thus, making worker information thread-aware without auditing its consumers is incomplete. ([Pinning implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/pin_memory.py), [Collation implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/collate.py))

### 6. Ordering, persistence and fetching

`in_order=True` requires task numbering and a bounded reorder buffer. A slow early batch can retain later completed batches; threads do not eliminate this memory pressure.

`persistent_workers=True` preserves worker state across iterations. A thread backend must define restart, exhaustion, early-break and stale-result behavior. It must also retain map-style batched `__getitems__` support, IterableDataset exhaustion handling, `drop_last` behavior and custom collation. These are substantial parts of `_MultiProcessingDataLoaderIter` and the fetchers, not incidental details. ([Iterator implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/dataloader.py), [Fetchers](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/fetch.py))

### 7. Exceptions, timeout and cancellation

Python exceptions can be wrapped and re-raised in the caller, as DataLoader does with `ExceptionWrapper`. A timeout can stop waiting for a result.

**A timeout cannot safely terminate an arbitrary Python thread.** A worker stuck in native decoding or I/O can continue holding resources after the iterator reports failure. Process workers can be terminated during shutdown; threads cannot provide equivalent isolation. A native crash in a thread can kill the training process.

### 8. KeyboardInterrupt and signals

`_worker_loop` installs worker signal handling and catches `KeyboardInterrupt`; the multiprocessing iterator also uses process liveness and, on supported platforms, SIGCHLD handling. These mechanisms cannot be mechanically transferred to threads.

Python signal handlers execute in the main thread. The thread backend needs cooperative cancellation, bounded joins and an explicit policy for unresponsive workers. Suppressing `KeyboardInterrupt` in a worker thread does not recreate multiprocessing shutdown behavior. ([Worker loop](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/_utils/worker.py), [Python signal rules](https://docs.python.org/3/library/signal.html#signals-and-threads))

### 9. Native libraries and CUDA

The proposed **“h5py without SWMR is not thread-safe” example is incorrect**. h5py serializes libhdf5 calls using its interpreter-wide `phil` lock, including on free-threaded builds. SWMR concerns a writer and concurrent readers of a file; it is not a thread-safety switch. The practical objection is often lost parallelism, plus application-level handle/state management. ([h5py threading](https://docs.h5py.org/en/stable/threads.html), [SWMR](https://docs.h5py.org/en/stable/swmr.html))

Shared decoder instances, mutable image objects and video-reader cursors require library-specific review; there is no defensible blanket statement that all decoders are unsafe. CUDA work likewise needs explicit device, stream and lifetime handling. Autograd, inference mode and autocast have thread-local aspects; parent-thread settings cannot simply be assumed inherited. ([PyTorch autograd threading notes](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/autograd.md))

## Should free-threaded builds automatically select threads?

**No.** PEP 703 removes a serialization mechanism; it does not grant thread safety or worker-state isolation to datasets.

The factual support picture is more nuanced than “PyTorch does not support no-GIL”:

- PyTorch’s compatibility matrix lists experimental free-threaded support, including 3.14t for 2.12. Its `torch._C` initialization explicitly declares `Py_MOD_GIL_NOT_USED` under `Py_GIL_DISABLED`. Claiming that Torch necessarily re-enables the GIL on import is therefore wrong. ([Compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [`Module.cpp`](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/Module.cpp))
- PyTorch announced removal of 3.13t wheels from nightlies and 2.13, directing users toward 3.14t. A new package should not treat these two interpreter targets as equally durable. ([Release announcement](https://dev-discuss.pytorch.org/t/dropping-python-3-13t-free-threaded-support-in-nightlies-and-in-the-future-pytorch-2-13-release/3386))
- Build support and actual GIL state differ. Unsupported extensions can re-enable it; `sys._is_gil_enabled()` checks runtime state. ([CPython free-threading guide](https://docs.python.org/3.14/howto/free-threading-python.html))
- Open discussion [#180550](https://github.com/pytorch/pytorch/issues/180550) identifies unresolved application-design questions around threaded distributed collectives. TorchCodec also has a free-threaded-wheel issue, [#1266](https://github.com/meta-pytorch/torchcodec/issues/1266). Neither is proof that ordinary eager CPU Torch is generally broken.
- I could not verify an exhaustive current inventory of Torch’s remaining GIL-dependent paths or certify 3.14t reliability across Torch, torchvision, decoders and third-party extensions. The historical [3.13 tracking issue](https://github.com/pytorch/pytorch/issues/130249) should not be presented as today’s failure list.

Use an explicit thread opt-in and a tested support matrix. On ordinary GIL builds, threads can still help I/O and native operations that release the GIL, but Python-heavy transforms can contend with the training loop.

# Q3 Auto-Serialize

**There is no safe general heuristic that turns “large list/dict attribute” into “semantically replaceable serialized storage.”** Size and shape can nominate candidates; they cannot establish the dataset’s access contract.

## What a defensible selection policy looks like

Require an explicit field declaration or dataset adapter, then validate a narrow schema—for example, an exact list of independent immutable metadata records. Reject custom container subclasses and custom serialization behavior initially. Sampling types or estimating memory is useful for recommendations, not proof of compatibility.

Even a list of immutable tuples can be used through identity checks, external aliases, slices or list-specific methods. Consequently, this is an **explicit snapshot representation**, not transparent optimization.

## Specific failure modes

- **Aliases across attributes:** `samples` and `imgs` may refer to the same list. Independent conversion breaks that relationship. A graph memo can preserve aliases among converted top-level containers, but it cannot find and update every external reference. Unconverted aliases can also keep the entire original graph alive.
- **Aliases across items:** Independent `pickle.dumps(item)` calls use independent memos. Shared nested objects become separate reconstructions. Repeated access to one index also returns fresh objects, changing identity and persistent mutation behavior.
- **Cycles:** Pickle supports many recursive graphs; “cycles break pickle” is false. The problem is splitting one graph into separate pickle records. A record referring back to the parent list can serialize much of the dataset repeatedly, preserve the wrong topology after loading, or cause extreme expansion. Deep recursion and custom reducers add other failure modes. ([Pickle semantics](https://docs.python.org/3.14/library/pickle.html))
- **Dictionary semantics:** `__len__`, `__iter__` and `__getitem__` are not a complete drop-in mapping contract. Preserve insertion order, key lookup semantics, membership and views—or explicitly exclude them. A retained Python key-to-offset dictionary can itself remain a large COW source. Replacing it with a packed index is a separate data-structure design problem.
- **Picklability:** Open handles, locks and lambdas commonly fail standard pickle. Failure must leave the dataset unchanged. Custom `__reduce__` and `__getstate__` can execute code or allocate substantial memory during packing.
- **PIL/OpenCV classification:** “PIL objects are unpicklable” is too broad. Pillow `Image.__getstate__` calls `tobytes()`, loading image data and serializing pixels; this can succeed while producing a disastrous memory/time expansion. OpenCV-produced NumPy arrays and stateful decoder objects must likewise be distinguished. ([Pillow source](https://github.com/python-pillow/Pillow/blob/main/src/PIL/Image.py))
- **Silent cache loss:** With `sample = packed[i]; sample["decoded"] = image`, the mutation succeeds on a temporary reconstruction and disappears on the next access. Rejecting writes to the outer wrapper does not detect this. `packed[i] = ...` may instead fail visibly.
- **Incomplete traversal:** `dataset.__dict__` misses slots, class attributes, closures, globals, wrapped child datasets and extension-owned state. Conversely, replacing attributes mutates a user-owned dataset that may be shared with another loader.

## Peak memory is not bounded by “up to 2×”

Let original live storage be `O`, serialized payload `B`, offsets `I`, and per-record temporary overhead `H`. A straightforward concatenate-based pack can temporarily require approximately:

**`O + 2B + I + H`**, before allocator retention and any copy into shared storage.

`B` need not be smaller than `O`; cross-record aliases can make it much larger. Millions of temporary bytes objects and NumPy views make `H` significant. This is an analytical estimate, not an additional measurement. Detectron2’s construction sequence demonstrates why both individual serialized buffers and their concatenation can coexist. ([Detectron2 implementation](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py))

Finally, a normal NumPy buffer is not automatically shared under spawn. Use explicit shared tensor storage or an explicitly reopened mmap representation for cross-start-method benefits. The supplied fixed-width NumPy result demonstrates a useful representation, but it does **not** measure the proposed per-item pickle implementation’s footprint or throughput.

# Q4 Immortalize Hack

## Exact version differences

These are source-level facts from the indicated release tags, not a supported recipe for writing object headers.

| Build | Immortal representation and `_Py_IsImmortal` |
|---|---|
| **3.12.3, GIL, 64-bit** | `_Py_IMMORTAL_REFCNT = UINT_MAX = 0xFFFFFFFF`. `_Py_IsImmortal` tests whether the low 32 bits interpreted as signed are negative. |
| **3.12.3, GIL, 32-bit** | `_Py_IMMORTAL_REFCNT = UINT_MAX >> 2 = 0x3FFFFFFF`; the check is equality. |
| **3.13.0, GIL** | Same numerical sentinels and 64-bit signed-low-word/32-bit equality distinction, with casts in the definitions. |
| **3.14.0, GIL, 64-bit** | `_Py_IMMORTAL_INITIAL_REFCNT = 3ULL << 30 = 0xC0000000`; minimum is `1ULL << 31 = 0x80000000`. `_Py_IsImmortal` checks the sign of the 32-bit count. |
| **3.14.0, GIL, 32-bit** | Initial `5L << 28 = 0x50000000`; minimum `1L << 30 = 0x40000000`; check is `>= minimum`. Static immortals use separate initial/minimum values, `0x70000000`/`0x60000000`. |
| **3.13t/3.14t** | `_Py_IMMORTAL_REFCNT_LOCAL = UINT32_MAX = 0xFFFFFFFF`. The check atomically reads `ob_ref_local` and compares for equality. It does not interpret the first machine word as `ob_refcnt`. |

Sources: [3.12.3 header](https://github.com/python/cpython/blob/v3.12.3/Include/object.h), [3.13.0 header](https://github.com/python/cpython/blob/v3.13.0/Include/object.h), [3.14.0 refcount header](https://github.com/python/cpython/blob/v3.14.0/Include/refcount.h).

A further 3.14 distinction is critical: `_Py_IsImmortal()` can be true below `_Py_IMMORTAL_INITIAL_REFCNT`, while `Py_INCREF` still increments toward that initial threshold. Therefore, **“the immortality predicate is true” does not necessarily establish “all increfs stop writing.”**

### Header layout and free threading

In 3.14’s 64-bit GIL layout, the first word contains a 32-bit `ob_refcnt`, `ob_overflow` and `ob_flags`, also accessible as `ob_refcnt_full`. An old machine-word `ctypes` store can overwrite more than the count. Writing `0xFFFFFFFF` is also outside the intended initial-value convention and can trip debug refcount checks. ([3.14 object layout](https://github.com/python/cpython/blob/v3.14.0/Include/object.h), [refcount checks](https://github.com/python/cpython/blob/v3.14.0/Include/refcount.h))

Free-threaded `_object` instead includes `ob_tid`, `ob_mutex`, `ob_gc_bits`, `ob_ref_local`, `ob_ref_shared` and `ob_type`; 3.14 also uses `ob_flags`. `ob_ref_shared` reserves its low two bits for state flags. The owning thread uses the local count; other threads generally update the shared count. This is biased reference counting, not an address-compatible variation of the GIL header. ([PEP 703](https://peps.python.org/pep-0703/), [3.13 layout](https://github.com/python/cpython/blob/v3.13.0/Include/object.h))

The runtime helper also performs more work than one store: 3.14 `_Py_SetImmortal` untracks GC objects, and `_Py_SetImmortalUntracked` updates build-specific flags and fields. Arbitrary concurrent immortalization has had a documented race, [CPython #113956](https://github.com/python/cpython/issues/113956). ([Helper implementation](https://github.com/python/cpython/blob/v3.14.0/Objects/object.c))

## Immortal does not mean immutable

All Q1 cache and payload mutations remain relevant. `Py_SET_TYPE` changes `ob_type`; permitted instance-class changes and extension operations are separate from reference counting. Generic dictionary materialization, type-version assignment and string hash caching have no universal “immortal means read-only” rule.

The traversal boundary is also unsound as a completeness claim. Stopping at functions skips closure-held data; stopping at classes skips class-held data. `gc.get_referents()` follows GC traversal support, which is not a universal enumeration of extension-owned references. Full reachable-graph discovery cannot be promised for arbitrary datasets. ([GC introspection contract](https://docs.python.org/3.14/library/gc.html#gc.get_referents))

Lifetime consequences extend beyond leaked metadata: immortalizing wrappers can prevent destructor-driven release of files, decoder state or other native resources. Replacing loaders repeatedly can accumulate immortal graphs in the parent. Restoring the original counts later is invalid because intervening reference traffic was skipped.

## Public and semi-public APIs

- **PEP 683** established the mechanism, not a general Python-level dataset immortalization API. ([PEP 683](https://peps.python.org/pep-0683/))
- **`gc.freeze()`** changes GC participation; it does not disable ordinary reference counting.
- **3.14 `sys._is_immortal()`** and **`PyUnstable_IsImmortal()`** inspect immortality; they do not set it. `sys._is_interned()` is likewise not a setter. ([`sys`](https://docs.python.org/3.14/library/sys.html#sys._is_immortal), [C API](https://docs.python.org/3.14/c-api/object.html#c.PyUnstable_IsImmortal))
- **`_Py_SetImmortal`** is an internal helper. The public `Py_SET_REFCNT` documentation describes overflow immortalization on free-threaded builds, but that is not a safe recursive graph API. ([Reference-counting API](https://docs.python.org/3.14/c-api/refcounting.html))
- **`_testinternalcapi`:** I could not verify a generally available arbitrary-object immortalization setter across both 3.13 and 3.14 test-module builds. It would be incorrect to publish an unverified function name as a dependency.
- **3.15 has `PyUnstable_SetImmortal()` already.** It expects an object uniquely referenced by the calling thread, intended for use by its creator shortly after creation; it untracks GC objects and returns whether immortalization occurred. This does not authorize retrofitting an already-shared dataset graph. ([Accepted API](https://docs.python.org/3.15/c-api/object.html#c.PyUnstable_SetImmortal))

The proposal’s long-term narrative is therefore outdated. Public exposure was debated in [September 2024](https://discuss.python.org/t/exposing-public-apis-for-immortal-objects/64264), including objections about arbitrary object safety. A narrower unstable C API was subsequently [implemented for 3.15](https://discuss.python.org/t/add-an-unstable-c-api-for-immortalizing-objects/105461). The remaining proposal would be about safe bulk/graph immortalization and its ownership contract.

Also distinguish 3.13t’s broad automatic immortalization at first additional-thread startup from 3.14t’s narrower immortality and increased use of deferred/per-thread reference counting. Neither automatically immortalizes an arbitrary dataset list. ([3.13 guide](https://docs.python.org/3.13/howto/free-threading-python.html), [3.14 guide](https://docs.python.org/3.14/howto/free-threading-python.html))

# Q5 Prior Art

| Project | Confirmed mechanism and relevance |
|---|---|
| **Detectron2** | The proposal identifies the right pattern, but current `detectron2/data/common.py` uses private `_TorchSerializedList`, selected by `DatasetFromList(serialize=True)`. Tensor storage supports sharing through Torch’s multiprocessing reducers for spawn/forkserver as well as fork. It does not discover arbitrary dataset attributes. ([Source](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py)) |
| **Yuxin Wu’s `NumpySerializedList`/`TorchSerializedList` work** | The 2022 article and accompanying repository already explain copy-on-read, RSS versus USS/PSS, packing and start-method differences. It should be treated as central prior art, not a peripheral implementation reference. ([Article and code links](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/)) |
| **MMEngine** | `BaseDataset(serialize_data=True)` packs its known `data_list` using `_serialize_data`, with `data_bytes` and `data_address`. This is an established dataset-owned schema. Its ordinary NumPy buffers do not, by themselves, establish shared backing under spawn. ([Source](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py)) |
| **Hugging Face Datasets** | Arrow-backed storage and memory mapping avoid materializing the full corpus as a parent Python-object graph. Conversion/formatting still creates working objects; in-memory construction is not universally mmap-backed. ([Arrow documentation](https://huggingface.co/docs/datasets/about_arrow)) |
| **FFCV** | Uses its own dataset representation and optimized loading/preprocessing pipeline. Relevant alternative architecture, requiring data/pipeline adaptation rather than an arbitrary Dataset fix. ([Project](https://ffcv.io/)) |
| **NVIDIA DALI** | Provides native CPU/GPU loading and preprocessing pipelines and framework integration. It changes the pipeline boundary; it does not make arbitrary Python metadata COW-safe. ([Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)) |
| **WebDataset** | Tar-shard streaming avoids needing a giant eagerly materialized per-sample metadata graph. Sharding and shuffle-buffer semantics need to be compared with exact map-style random access. ([Project](https://github.com/webdataset/webdataset)) |
| **Ray Data** | Stores blocks—typically Arrow tables—in a shared-memory object store and schedules distributed operations over them. This is explicit data representation and execution infrastructure, not transparent fork-heap sharing. ([Internals](https://docs.ray.io/en/latest/data/data-internals.html)) |
| **TensorDict `MemoryMappedTensor`** | Explicit mmap-backed tensor storage with process-sharing facilities. Useful for structured numerical metadata; it does not preserve arbitrary Python-object identity and mutation semantics. ([API](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html)) |
| **Instagram/Cinder** | Meta’s pre-fork workload was a major motivation and production proving ground for immortal objects, related directly to PEP 683. A runtime-owned deployment is substantially different from a third-party package overwriting unknown object headers. The historical Cinder runtime repository now redirects to MetaPython. ([Meta account](https://engineering.fb.com/2023/08/15/developer-tools/immortal-objects-for-python-instagram-meta/), [Runtime repository](https://github.com/facebookincubator/MetaPython)) |

**Material omissions:**

1. **MONAI `ThreadDataLoader(use_thread_workers=True)`** already exposes thread workers and explicitly warns that some datasets and random transforms are not thread-safe. A new implementation needs a concrete compatibility or maintenance advantage. ([API](https://monai.readthedocs.io/en/0.9.0/data.html#threaddataloader))
2. **Meta SPDL** explicitly explores threaded loading, GIL-releasing preprocessing and reduced memory. Its published work is relevant to both benchmarking and architecture. ([Paper](https://arxiv.org/abs/2504.20067))
3. **PyTorch PR #191555**, open in the retrieved source, substantially overlaps layer 2: shared serialized containers, automatic dataset-field conversion and a DataLoader flag. It is a proposal, not an accepted or verified complete solution. ([PR](https://github.com/pytorch/pytorch/pull/191555))

Exact searches for `lowmem_dataloader` and `immortal_dataloader` returned no indexed matches. I did **not** verify an existing PyPI package that transparently solves this exact problem while preserving arbitrary Dataset semantics. That bounded search result does not justify “first ever”; MONAI and the overlapping PyTorch PR already invalidate broader novelty claims.

# Q6 Upstream Strategy

The following are engineering judgments, not maintainer commitments.

| Layer | Realistic core prospects | Better initial landing |
|---|---|---|
| **1: Threads** | Plausible as an explicit new backend, but low near-term odds for a fully compatible addition. Automatic selection is particularly difficult to defend. | TorchData nodes or a standalone backend with a deliberately narrower contract. |
| **4: Diagnostics** | Highest relative prospects, especially as opt-in instrumentation or documentation. An unconditional causal warning based on private-memory growth has a much weaker case. | Standalone diagnostic first; then a small instrumentation/documentation proposal. |
| **2: Auto-serialize** | Low for arbitrary introspection inside the DataLoader constructor. Explicit shared containers or a dataset protocol are more defensible. | Standalone storage adapters, possibly TorchData utilities. Coordinate with existing work. |
| **3: `ctypes` immortalization** | Extremely low as specified. Permanent lifetime changes and interpreter-layout manipulation impose risks beyond DataLoader’s ownership. | Research-only experiment; separate CPython discussion. |

There are specific roadmap signals:

- TorchData announced removal of DataPipes and DataLoaderV2, deprecated in 0.8 and removed in 0.9. “Target DataLoader v2” is therefore obsolete advice. ([Status update](https://meta-pytorch.org/data/0.9/index.html))
- The later [Polylithic RFC #1334](https://github.com/meta-pytorch/data/issues/1334) explicitly explores threaded, composable loading. `ParallelMapper(method="thread")` now supplies an existing architectural home. It does not claim process-worker compatibility. ([Nodes API](https://meta-pytorch.org/data/main/torchdata.nodes.html))
- Issue #13246’s editor note already recommends alternative representations and distinguishes NumPy’s fork-only benefit. I could verify that note, but not a complete historical maintainer-comment audit. ([Issue](https://github.com/pytorch/pytorch/issues/13246))
- On overlapping PR #191555, Alban Desmaison explicitly described the change as subtle and complex and said it might require an RFC and deeper design discussion. That is evidence of review burden, not rejection or endorsement. ([Discussion](https://github.com/pytorch/pytorch/pull/191555))

## Why diagnostics are not automatically upstream-ready

`Private_Dirty` growth includes decoded samples, caches, allocator retention and queue-related allocations. Spawn can start with a large duplicate and show almost no growth—as your measurements demonstrate. A first-epoch rule also lacks a natural boundary for infinite IterableDatasets or interrupted iterations.

A defensible diagnostic should:

- Report **observed private-memory growth**, with COW as a possible explanation.
- Record the actual multiprocessing context and sample count.
- Sample from the parent at a bounded frequency; tolerate missing `/proc` access and worker exit.
- Distinguish startup footprint from subsequent growth.
- Account for prefetch and warm-up; avoid duplicate warnings across ranks.
- Offer process-family PSS alongside worker-private figures.

These requirements follow from Linux’s accounting definitions and DataLoader’s buffering/lifecycle behavior; they require validation before an automatic warning is justified. ([Linux accounting](https://www.kernel.org/doc/html/latest/filesystems/proc.html), [DataLoader](https://docs.pytorch.org/docs/main/data.html))

# Q7 Rankings

These rank the proposed layers as engineering investments. Required restrictions are stated explicitly; no ranking makes the original blanket compatibility claims acceptable.

## A. Technical soundness

1. **Layer 4 — Diagnostics.** Reading memory counters is technically straightforward and does not alter dataset semantics. Its interpretation must remain observational rather than claiming to identify COW.
2. **Layer 2 — Serialization.** Packed backing storage has a strong mechanism and established implementations. Automatic field replacement remains unsound without an explicit snapshot contract.
3. **Layer 1 — Threads.** Shared-address-space execution removes inter-worker COW duplication. Reproducing process isolation, global RNG behavior and cancellation is impossible in general, making the proposed compatibility surface substantially harder.
4. **Layer 3 — Immortalization hack.** The measured mechanism is persuasive, but arbitrary graph traversal and raw header writes lack a safe runtime contract. Version gating addresses only part of that problem.

## B. Likelihood of acceptance into `pytorch/pytorch`

1. **Layer 4 — Diagnostics.** An opt-in observer or focused documentation improvement has the smallest semantic burden. Automatic warnings still need evidence about false positives and overhead.
2. **Layer 1 — Threads.** An explicit backend is a coherent capability with ecosystem precedent. Core inclusion requires a worker-context design and careful compatibility boundaries; TorchData is the easier first venue.
3. **Layer 2 — Auto-serialize.** There is already concrete upstream interest through an overlapping proposal, but constructor-driven dataset mutation crosses ownership boundaries. Explicit containers are a stronger candidate than the proposed keyword.
4. **Layer 3 — Immortalization hack.** There is no credible near-term case for shipping arbitrary `ctypes` immortalization inside stable DataLoader. A future runtime-supported contract would constitute a materially different proposal.

## C. Real user value

1. **Layer 2 — Serialization.** For large immutable metadata, it directly addresses the demonstrated failure while retaining process workers. Explicit sharing can also extend the benefit to spawn and forkserver.
2. **Layer 1 — Threads.** It can reduce interpreter/process overhead and enable efficient I/O or native preprocessing across platforms. Its value is substantial for audited pipelines but can reverse for Python-heavy or stateful datasets.
3. **Layer 4 — Diagnostics.** Correct measurements can prevent wasted debugging and misleading RSS conclusions. It does not itself recover memory, and a late warning cannot save a job that already exhausted RAM.
4. **Layer 3 — Immortalization hack.** It has research value and a striking result for controlled pre-fork workloads. Permanent retention, resource lifetime changes and platform restrictions make it the weakest general-user feature.

# Q8 Recommended MVP and Flat Errors in the Plan

## Recommended scope and ordering

1. **Publish the supplied measurements with precise claims.** Preserve their status as verified observations. Label the approximately 45× endpoint reduction correctly and distinguish fixed-width NumPy packing from per-item pickle serialization.
2. **Start with diagnostics plus explicit packed metadata storage.** A drop-in import can preserve ordinary DataLoader defaults; memory-changing behavior should require explicitly selected fields or a supported adapter. Initially support independent metadata records and documented snapshot semantics.
3. **Design storage for all intended process start methods.** Prefer explicitly shared tensor buffers or reopenable mmap storage. Specify ownership, cleanup, interrupted packing and original-alias retention.
4. **Validate compatibility and end-to-end value before expanding.** Required evidence includes output equivalence, identity/mutation limitations, packing peak memory, startup time, throughput, repeated epochs, early termination and total parent-plus-worker memory. These are proposed future evaluations, not work performed in this review.
5. **Develop threads as a separate opt-in backend.** Begin with audited map-style datasets and explicit RNG/state restrictions. Compare against MONAI, TorchData and SPDL before rebuilding their capabilities.
6. **Exclude immortalization from the supported MVP.** Keep the experiment available as research evidence. Engage CPython about the gap between the accepted creator-side C API and safe pre-fork graph treatment.
7. **Split upstream proposals.** Instrumentation, explicit shared containers and thread-worker semantics are separate reviewable changes. Coordinate with #191555 rather than presenting layer 2 as an unexplored direction.

One additional portability omission is significant: **Python 3.14 changed the POSIX default start method to forkserver where supported; fork is no longer the default anywhere.** Immortalizing objects in the training parent does not make them appear in a separately started forkserver’s heap. Fork-specific behavior must be explicitly scoped. ([Multiprocessing documentation](https://docs.python.org/3.14/library/multiprocessing.html#contexts-and-start-methods))

## Flat errors or unsupported premises

- **“Users need no code changes” is incompatible with arbitrary semantic preservation.** An import-only change cannot generally preserve both process isolation and thread sharing, or both live containers and serialized snapshots.
- **A free-threaded interpreter is not a thread-safety certificate.**
- **`gc.freeze()` does not suppress normal reference counting.**
- **`ob_shash` is the wrong field for Python 3 `str`.**
- **Plain attribute access can trigger specialization/cache writes.**
- **Pickle supports cycles; per-record graph splitting is the actual problem.**
- **PIL images are not categorically unpicklable.**
- **SWMR is not h5py’s thread-safety mechanism.**
- **Packing has no universal 2× peak-memory bound.**
- **NumPy storage alone does not provide spawn sharing.**
- **The same `ctypes` header write is not valid across 3.12–3.14 builds.**
- **Private-memory growth is not proof of COW, and flat growth is not proof of low memory use.**
- **An immortalization C API is no longer merely a long-term proposal; a constrained unstable API exists in 3.15.**

**Recommended product claim:** reduce worker memory duplication for explicitly supported metadata representations, provide trustworthy diagnostics, and offer an opt-in thread backend for audited datasets. That is useful, testable and defensible; a universal transparent DataLoader fix is not established by this design.

