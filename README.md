# lmdbobjectstore

Lightweight thread-safe Python object store on top of **LMDB**.
Provides a dict-like API (with buffering and atomic multi-put), automatic map-size growth, and fast zero-copy reads without having to handle LMDB's lower-level details.

---

## Features

* **Dict-like interface**: `store[key] = obj`, `obj = store.get(key)`, `del store[key]`, `key in store`.
* **Atomic multi-put**: `put_many(items)` writes large amounts of data **atomically** in a single transaction.
* **Write buffering**: small writes are batched in memory; `flush()` persists them (auto-flush on size threshold).
* **Auto map-size growth**: retries on `MapFullError` by growing the LMDB map (2× or +64MiB), up to an optional cap.
* **Fast reads**: Uses LMDB zero-copy reads to minimize copies; `get_many()` efficiently combines buffered and DB reads.
* **Flexible keys**: bytes/bytearray/memoryview always supported; `str` keys allowed when `key_encoding='utf-8'` option is used.

---

## Installation

```bash
pip install lmdbobjectstore
```

### Requirements

* Python **3.10+**
* The [`lmdb`](https://pypi.org/project/lmdb/) Python package (wheels available for major platforms)

---

## Quick start

```python
from lmdb_object_store import LmdbObjectStore

# Create or open an LMDB-backed object store
with LmdbObjectStore(
    "path/to/db",
    batch_size=1000,                # flush buffer when it reaches this many entries
    autoflush_on_read=True,         # flush pending writes before reads
    key_encoding="utf-8",           # allow str keys (encoded with UTF-8)
    # Any lmdb.open(...) kwargs may be passed here, e.g.:
    map_size=128 * 1024 * 1024,     # 128 MiB initial map size
    subdir=True,                    # create directory layout
    readonly=False,
    # max_map_size is recognized (cap for auto-resize):
    max_map_size=4 * 1024 * 1024 * 1024,  # 4 GiB cap
) as store:

    # Put / get like a dict (values are pickled)
    store["user:42"] = {"name": "Ada", "plan": "pro"}
    print(store.get("user:42"))  # {'name': 'Ada', 'plan': 'pro'}

    # Existence checks
    if "user:42" in store:        # __contains__ has NO side effects (no flush)
        assert store.exists("user:42", flush=False) is True

    # Delete
    del store["user:42"]

    # Batch write, atomically (single transaction)
    items = {f"k{i}": {"i": i} for i in range(10_000)}
    store.put_many(items)

    # Fetch many at once
    found, not_found = store.get_many(["k1", "kX"], decode_keys=True)
    # found -> {'k1': {'i': 1}}, not_found -> ['kX']

# Clean close with strict error policy if needed:
store = LmdbObjectStore("path/to/db", key_encoding="utf-8")
try:
    # ... work with store ...
    store.close(strict=True)  # re-raise if final flush fails (after cleanup)
finally:
    # idempotent
    try: store.close()
    except Exception: pass
```

---

## API Overview

### Constructor

```python
LmdbObjectStore(
    db_path: str,
    batch_size: int = 1000,
    *,
    autoflush_on_read: bool = True,
    key_encoding: str | None = None,
    key_errors: str = "strict",
    str_normalize: str | None = None,
    **lmdb_kwargs,
)
```

* **db\_path**: LMDB environment path (passed to `lmdb.open`).
* **batch\_size**: pending buffer size threshold to auto-flush.
* **autoflush\_on\_read**: if `True`, flushes buffer before reads (`get`, `get_many`, `exists(flush=None)`).
* **key\_encoding**: enable `str` keys (e.g. `"utf-8"`). If `None`, only bytes-like keys are allowed.
* **key\_errors**: error strategy for encoding `str` keys (`"strict"`, `"ignore"`, `"replace"`, ...).
* **str\_normalize**: Unicode normalization for `str` keys (e.g., `"NFC"`, `"NFKC"`).
* **lmdb\_kwargs**: forwarded to `lmdb.open(...)` (e.g., `map_size`, `subdir`, `readonly`, etc).
  Special: **`max_map_size`** (cap for automatic map growth) is also recognized.

### Put / Get

```python
store.put(key, obj)                          # buffer write
obj = store.get(key, default=None)           # read; from buffer if present, else DB
store.flush()                                # persist the write buffer
```

* Values are serialized with `pickle` (highest protocol).
* `get()` uses zero-copy buffers internally; unpickling happens once per value.

### Atomic multi-put

```python
store.put_many(items: Mapping[Any, Any] | Iterable[tuple[Any, Any]])
```

* Writes **all** items in **one LMDB write transaction**.
* If a `MapFullError` occurs, the store will **grow the map** (2× or +64MiB) up to `max_map_size` and **retry from the beginning**.
* **Note**: `put_many()` first flushes any pending buffered writes; the atomic transaction only includes the `items` passed to this call.

### Get many

```python
found, not_found = store.get_many(
    keys: Sequence[Any],
    *,
    decode_keys: bool = False,
    decode_not_found: bool | None = None,   # None → follow decode_keys
)
```

* Returns a tuple:

  * `found`: `{key: value}` for keys found (key type is `bytes` by default, or `str` if `decode_keys=True`).
  * `not_found`: list of input keys not found (decoded to `str` if `decode_not_found=True`).
* Efficiently merges results from the write buffer and DB; `autoflush_on_read` applies unless overridden via other APIs.

### Existence & containment

```python
store.exists(key, *, flush: bool | None = None) -> bool
key in store  # __contains__ → NO flush, purely checks current state
```

* `flush=None` (default) follows `autoflush_on_read`.
* `flush=False` guarantees **no** implicit flush (useful for side-effect-free checks).

### Deletion

```python
del store[key]       # KeyError if not present
store.delete(key)    # schedules deletion via buffer (dict-like semantics)
```

### Lifecycle

```python
with LmdbObjectStore(...) as store:
    ...
# or
store.close(strict: bool = False)
```

* `close(strict=True)` re-raises the last flush error **after** closing the environment; otherwise it logs and completes.

---

## Concurrency Model

* Internally uses `RLock` + `Condition` and a simple **reader count** to coordinate:

  * Multiple concurrent readers are allowed.
  * Writers hold the lock (buffer mutation + flush/commit).
  * `close()` sets a “closing” flag and **waits** until the active reader count reaches zero.
* Designed for **thread-safety within a single process**. While LMDB itself supports multi-process access, this wrapper's locking is process-local; if you need multi-process writes, coordinate at a higher level.

---

## Map Size & Auto-Resize

* Initial size is given by `map_size` (forwarded to `lmdb.open`).
* When a write/commit hits `MapFullError`:

  * The store **grows** to `max(current*2, current+64MiB)`, capped at `max_map_size` if provided,
  * and **retries** the operation.
* If the cap is reached and still insufficient, the error is propagated.

---

## Performance Tips

* **Batch writes**: Keep `batch_size` large enough for your workload; call `flush()` at logical boundaries.
* **Use `put_many()`** for bulk inserts—single transaction with fewer fsyncs.
* **Avoid unnecessary decodes**: If you don't need `str` keys on output, leave `decode_*` parameters off.
* **Key type**: If possible, pass bytes keys directly (saves encoding overhead).

---

## Error Handling

* **Unpickling failures** are raised as a `RuntimeError` (with key context) when reading from buffer/DB.
* **Missing keys**: `__getitem__` and `del` raise `KeyError`; `get()` returns `default`.
* **Final flush failure on close**: re-raised if `strict=True`; otherwise logged.

---

## Security Note

This library uses Python `pickle` for value serialization. **Never unpickle data from untrusted sources**.

---

## Configuration Reference

* `batch_size: int` – buffer size threshold to auto-flush (default: 1000).
* `autoflush_on_read: bool` – flush before reads (default: `True`).
* `key_encoding: Optional[str]` – enable `str` keys (e.g., `"utf-8"`). If `None`, only bytes-like keys are accepted.
* `key_errors: str` – encoding error handling (`"strict"`, `"ignore"`, `"replace"`).
* `str_normalize: Optional[str]` – Unicode normalization for `str` keys (`"NFC"`, `"NFKC"`, ...).
* `lmdb_kwargs` – forwarded to `lmdb.open(...)`:

  * `map_size`, `subdir`, `readonly`, `lock`, ...
  * `max_map_size` (recognized by this wrapper to cap auto-growth).

---