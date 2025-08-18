"""
LmdbObjectStore: A thread-safe, buffered object store using LMDB.

This class provides a convenient and efficient way to store and
retrieve arbitrary Python objects in an LMDB database, with support
for write buffering, automatic map resizing, and flexible key handling.

Features
--------
- Thread-safe with reader-writer lock pattern for concurrent reads and writes.
- Write buffering: put/delete operations are buffered in memory
    and flushed in batches for performance.
- Automatic LMDB map resizing on MapFullError, with optional maximum map size.
- Flexible key handling: supports bytes, bytearray, memoryview,
    and (optionally) str keys with configurable encoding and Unicode normalization.
- Optional autoflush on read to ensure data consistency.
- Supports context manager protocol for safe resource management.
- Implements dict-like interface: __getitem__, __setitem__, __delitem__, __contains__.

Usage Notes
-----------
- Use `put` to store objects, `get` to retrieve them, and `delete` to remove them.
- LMDB sorts keys in lexicographical byte order. For numerical keys,
    use fixed-width zero-padded strings or big-endian binary representations.
- Setting autoflush_on_read=False can improve write throughput at the cost of
    possible stale reads.
- The store can be used as a context manager to ensure proper closing and flushing.

Example
-------
    with LmdbObjectStore("mydb", key_encoding="utf-8") as db:
        db["foo"] = {"bar": 123}
        value = db["foo"]
"""

import logging
import pickle
import threading
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import lmdb

log = logging.getLogger(__name__)


class LmdbObjectStore:
    """
    LmdbObjectStore: A thread-safe, buffered object store using LMDB.

    Thread-safe, buffered object store using LMDB for storing and retrieving
    Python objects.

    Parameters
    ----------
        batch_size (int, optional): Number of buffered operations before
            automatic flush. Defaults to 1000.
        autoflush_on_read (bool, optional): If True, flushes buffer before
            read operations for consistency. Defaults to True.
        key_encoding (Optional[str], optional): Encoding for string keys
            (e.g., 'utf-8'). If None, str keys are not allowed.
        **lmdb_kwargs: Additional keyword arguments for lmdb.open(), such as
            map_size, subdir, readonly, etc.
    """

    _DELETION_SENTINEL = object()

    def __init__(
        self,
        db_path: str,
        batch_size: int = 1000,
        *,
        autoflush_on_read: bool = True,
        key_encoding: str | None = None,
        key_errors: str = "strict",
        str_normalize: str | None = None,
        **lmdb_kwargs,
    ):
        """
        Initialize the LmdbObjectStore.

        Parameters
        ----------
            db_path (str): Path to the LMDB database file.
            batch_size (int, optional): Buffer size before an automatic
                flush. Defaults to 1000.
            autoflush_on_read (bool, optional): If True, automatically
                flushes the buffer before read operations. Defaults to True.
            key_encoding (Optional[str], optional): Encoding for string keys
                (e.g., 'utf-8'). If None, str keys are disallowed. Defaults to None.
            key_errors (str, optional): Error handling for string encoding.
                Defaults to "strict".
            str_normalize (Optional[str], optional): Unicode normalization
                form for string keys (e.g., 'NFC'). Defaults to None.
            **lmdb_kwargs: Additional keyword arguments to pass to
                lmdb.open(). `max_map_size` is also a valid option here.
        """
        self.db_path = db_path
        self.max_map_size = lmdb_kwargs.pop("max_map_size", None)
        self.env = lmdb.open(db_path, **lmdb_kwargs)
        self.batch_size = batch_size
        self.autoflush_on_read = autoflush_on_read
        self.write_buffer = {}

        # Thread safety attributes
        self._is_closed = False
        self._closing = False
        self._readers = 0
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)

        # Key normalization policy
        self.key_encoding = key_encoding
        self.key_errors = key_errors
        self.str_normalize = str_normalize
        self._readonly = bool(lmdb_kwargs.get("readonly", False))

    def _norm_key(self, key: Any) -> bytes:
        if key is None:
            raise TypeError("Key cannot be None.")
        if isinstance(key, bytes):
            return key
        if isinstance(key, bytearray | memoryview):
            return bytes(key)
        if isinstance(key, str):
            if self.key_encoding is None:
                raise TypeError(
                    "str keys are not allowed"
                    " (set key_encoding='utf-8' etc. to enable)."
                )
            s = (
                unicodedata.normalize(self.str_normalize, key)
                if self.str_normalize
                else key
            )
            return s.encode(self.key_encoding, self.key_errors)
        raise TypeError(
            "Key must be bytes-like"
            f"{' or str' if self.key_encoding else ''}; "
            f"got {type(key).__name__}"
        )

    def _ensure_open(self):
        if self._closing or self._is_closed:
            raise lmdb.Error("Database is closed or in the process of closing.")

    def _ensure_writable(self):
        if self._readonly:
            raise lmdb.Error(
                "Environment is read-only; write operations are not allowed."
            )

    def _format_key_for_display(self, key: Any) -> str:
        """
        Format a key for display in error messages.

        Returns a human-readable representation that distinguishes between
        text and binary keys while keeping messages concise.

        Parameters
        ----------
        key : Any
            The key to format for display.

        Returns
        -------
        str
            A formatted string representation of the key.
        """
        if isinstance(key, bytes):
            # Try to decode as UTF-8 for text-like keys
            try:
                decoded = key.decode("utf-8")
                # Check if it contains only printable characters
                if decoded.isprintable() and len(decoded) <= 50:
                    return f"'{decoded}'"
            except UnicodeDecodeError:
                pass
            # For binary data, show hex representation with size limit
            if len(key) <= 16:
                return f"0x{key.hex()}"
            else:
                return f"0x{key[:8].hex()}...({len(key)} bytes)"
        elif isinstance(key, str):
            # String keys are displayed with quotes
            if len(key) <= 50:
                return f"'{key}'"
            else:
                return f"'{key[:47]}...'"
        else:
            # For other types, use repr with length limit
            key_repr = repr(key)
            if len(key_repr) <= 50:
                return key_repr
            else:
                return f"{key_repr[:47]}..."

    def _key_not_found_error(self, key: Any) -> KeyError:
        """
        Generate a consistent KeyError for missing keys.

        Parameters
        ----------
        key : Any
            The key that was not found.

        Returns
        -------
        KeyError
            A KeyError with a formatted message.
        """
        return KeyError(f"Key not found: {self._format_key_for_display(key)}")

    def _unpickle_error(
        self, key: Any, context: str, original_error: Exception
    ) -> RuntimeError:
        """
        Generate a consistent RuntimeError for unpickling failures.

        Parameters
        ----------
        key : Any
            The key that failed to unpickle.
        context : str
            Context description (e.g., "buffered", "DB").
        original_error : Exception
            The original exception that caused the unpickling failure.

        Returns
        -------
        RuntimeError
            A RuntimeError with a formatted message.
        """
        return RuntimeError(
            f"Failed to unpickle {context} key "
            f"{self._format_key_for_display(key)}: {original_error}"
        )

    def _grow_mapsize_for_retry(self) -> None:
        """
        Grow the LMDB map size when encountering MapFullError.

        Calculates new size as the larger of double current size or current + 64MB.
        Respects max_map_size limit if configured.

        Raises
        ------
        lmdb.MapFullError
            If the current size is already at max_map_size limit.
        """
        info = self.env.info()
        current_size = info["map_size"]

        # Check if we're already at the maximum allowed size
        if self.max_map_size is not None and current_size >= self.max_map_size:
            log.error(
                "Cannot resize: map size (%d) is already at its configured "
                "maximum (%d).",
                current_size,
                self.max_map_size,
            )
            raise lmdb.MapFullError("Map size at configured maximum")

        # Calculate new size: double or add 64MB, whichever is larger
        new_size = max(current_size * 2, current_size + 64 * 1024 * 1024)

        # Cap at max_map_size if configured
        if self.max_map_size is not None:
            new_size = min(new_size, self.max_map_size)

        log.warning(
            "MapFullError: growing mapsize from %d to %d", current_size, new_size
        )
        self.env.set_mapsize(new_size)

    def _iter_normalized_pickled(
        self, items: Mapping[Any, Any] | Iterable[tuple[Any, Any]]
    ) -> Iterator[tuple[bytes, bytes | object]]:
        """
        Normalize and pickle items lazily without creating large intermediate buffers.

        Parameters
        ----------
        items : Mapping[Any, Any] | Iterable[tuple[Any, Any]]
            Items to normalize and pickle.

        Yields
        ------
        tuple[bytes, bytes | object]
            Normalized key and pickled value (or DELETION_SENTINEL).
        """
        it = items.items() if isinstance(items, Mapping) else items

        for k, v in it:
            norm_key = self._norm_key(k)
            # Respect existing deletion semantics: DELETION_SENTINEL is passed through
            if v is self._DELETION_SENTINEL:
                yield norm_key, self._DELETION_SENTINEL
            else:
                yield norm_key, pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)

    def _flush(self):
        if not self.write_buffer:
            return

        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for key, value in self.write_buffer.items():
                        if value is self._DELETION_SENTINEL:
                            txn.delete(key)
                        else:
                            txn.put(key, value)
                break  # Success - exit the retry loop

            except lmdb.MapFullError:
                # Use the helper method to grow the map size
                self._grow_mapsize_for_retry()
                # Continue the loop to retry the write operation

        self.write_buffer.clear()

    def put(self, key: Any, obj: Any):
        """
        Store an object in the database under the given key.

        Parameters
        ----------
        key : Any
            The key under which to store the object. Must be bytes-like or str
                (if key_encoding is set).
        obj : Any
            The Python object to store. It will be pickled.

        Raises
        ------
        TypeError
            If the key is of an unsupported type.
        lmdb.Error
            If the database is closed or in read-only mode.
        """
        norm_key = self._norm_key(key)
        pickled_value = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        with self.lock:
            self._ensure_open()
            self._ensure_writable()
            self.write_buffer[norm_key] = pickled_value
            if len(self.write_buffer) >= self.batch_size:
                self._flush()

    def get_many(
        self,
        keys: Sequence[Any],
        *,
        decode_keys: bool = False,
        decode_not_found: bool | None = None,
    ) -> tuple[dict[bytes | str, Any], list[bytes | str]]:
        """
        Retrieve multiple objects from the database by their keys.

        Parameters
        ----------
        keys : Sequence[Any]
            The keys of the objects to retrieve. Must be bytes-like or str
            (if key_encoding is set).
        decode_keys : bool, optional
            If True, decode the keys using the specified key_encoding.
        decode_not_found : bool | None, optional
            If True, decode the not found keys using the specified key_encoding.

        Returns
        -------
        tuple[dict[bytes | str, Any], list[bytes | str]]
            A tuple containing a dictionary of found objects and a list of
            keys that were not found.
        """
        if (decode_keys or (decode_not_found is True)) and not self.key_encoding:
            raise ValueError("Decoding requested but key_encoding is not set.")
        if decode_not_found is None:
            decode_not_found = decode_keys
        norm_keys = [self._norm_key(k) for k in keys]

        found_pickled: dict[bytes, bytes] = {}
        keys_to_check_in_db: list[bytes] = []

        with self.lock:
            self._ensure_open()
            seen_for_db = set()
            for k in norm_keys:
                if k in self.write_buffer:
                    v = self.write_buffer[k]
                    if v is not self._DELETION_SENTINEL and k not in found_pickled:
                        found_pickled[k] = v
                else:
                    if k not in seen_for_db:
                        keys_to_check_in_db.append(k)
                        seen_for_db.add(k)

            if keys_to_check_in_db and self.autoflush_on_read:
                self._flush()

            if keys_to_check_in_db:
                self._readers += 1

        found_objects = {}
        for k, v in found_pickled.items():
            try:
                found_objects[k] = pickle.loads(v)
            except Exception as e:
                raise self._unpickle_error(k, "buffered", e) from e

        if keys_to_check_in_db:
            try:
                with self.env.begin(buffers=True) as txn:
                    for k in keys_to_check_in_db:
                        pv = txn.get(k)
                        if pv is not None and k not in found_objects:
                            try:
                                found_objects[k] = pickle.loads(pv)
                            except Exception as e:
                                raise self._unpickle_error(k, "DB", e) from e
            finally:
                with self.lock:
                    self._readers -= 1
                    if self._readers == 0 and self._closing:
                        self.condition.notify_all()

        not_found = [
            orig_k
            for orig_k, norm_k in zip(keys, norm_keys, strict=False)
            if norm_k not in found_objects
        ]

        if decode_keys and self.key_encoding:
            found_objects = {
                k.decode(self.key_encoding, self.key_errors): v
                for k, v in found_objects.items()
            }
        if decode_not_found and self.key_encoding:
            not_found = [
                (
                    k.decode(self.key_encoding, self.key_errors)
                    if isinstance(k, bytes | bytearray | memoryview)
                    else k
                )
                for k in not_found
            ]

        return found_objects, not_found

    def put_many(
        self,
        items: Mapping[Any, Any] | Iterable[tuple[Any, Any]],
    ) -> None:
        """
        Store multiple key-value pairs in the database atomically.

        This method writes all items in a single LMDB transaction, ensuring
        atomicity. If the operation fails (e.g., due to insufficient space),
        no items are written. The database map size is automatically resized
        if needed.

        Parameters
        ----------
        items : Mapping[Any, Any] | Iterable[tuple[Any, Any]]
            Items to store. Can be a dict-like mapping or an iterable of
            (key, value) tuples. Duplicate keys follow last-write-wins semantics.

        Raises
        ------
        TypeError
            If any key is of an unsupported type.
        lmdb.Error
            If the database is closed or in read-only mode.
        lmdb.MapFullError
            If the map cannot be resized enough to fit all items.

        Notes
        -----
        - Always materializes non-reusable iterables to allow retry.
        - Always flushes any existing write buffer before executing.
        - The operation is always atomic (all-or-nothing).
        - Automatically retries with map resizing on MapFullError.
        - For non-reusable iterables, items are materialized once to allow retry.
        """
        with self.lock:
            self._ensure_open()
            self._ensure_writable()

            if self.write_buffer:
                self._flush()

            # Handle non-reusable iterables: materialize only when needed
            # (normalized + pickled)
            is_mapping = isinstance(items, Mapping)
            materialized: list[tuple[bytes, bytes | object]] | None = None
            if not is_mapping:
                materialized = list(self._iter_normalized_pickled(items))

            while True:
                try:
                    with self.env.begin(write=True) as txn:
                        iterator = (
                            materialized
                            if materialized is not None
                            else self._iter_normalized_pickled(
                                items
                            )  # Mapping is reusable
                        )
                        for nk, pv in iterator:
                            if pv is self._DELETION_SENTINEL:
                                txn.delete(nk)
                            else:
                                txn.put(nk, pv)
                    break  # Commit successful
                except lmdb.MapFullError:
                    # Retry entire operation after resizing map
                    self._grow_mapsize_for_retry()

    def get(self, key: Any, default: Any | None = None) -> Any | None:
        """
        Retrieve an object from the database by its key.

        Parameters
        ----------
        key : Any
            The key of the object to retrieve. Must be bytes-like or str
            (if key_encoding is set).
        default : Any | None, optional
            The value to return if the key is not found. Defaults to None.

        Returns
        -------
        Any | None
            The object associated with the key, or default if not found.

        Raises
        ------
        TypeError
            If the key is of an unsupported type.
        RuntimeError
            If the stored value cannot be unpickled.
        """
        norm_key = self._norm_key(key)
        with self.lock:
            self._ensure_open()
            if norm_key in self.write_buffer:
                value = self.write_buffer[norm_key]
                if value is self._DELETION_SENTINEL:
                    return default
                try:
                    return pickle.loads(value)
                except Exception as e:
                    raise self._unpickle_error(norm_key, "buffered", e) from e

            if self.autoflush_on_read:
                self._flush()
            self._readers += 1

        try:
            with self.env.begin(buffers=True) as txn:
                value = txn.get(norm_key)
                if value is None:
                    return default
                try:
                    return pickle.loads(value)
                except Exception as e:
                    raise self._unpickle_error(norm_key, "DB", e) from e
        finally:
            with self.lock:
                self._readers -= 1
                if self._readers == 0 and self._closing:
                    self.condition.notify_all()

    def delete(self, key: Any):
        """
        Mark an object for deletion from the database by its key.

        This method does not immediately remove the object from the database,
        but marks it for deletion. The actual removal will occur during the
        next flush operation.

        Parameters
        ----------
        key : Any
            The key of the object to delete. Must be bytes-like or str
            (if key_encoding is set).

        Raises
        ------
        TypeError
            If the key is of an unsupported type.
        lmdb.Error
            If the database is closed or in read-only mode.
        """
        norm_key = self._norm_key(key)
        with self.lock:
            self._ensure_open()
            self._ensure_writable()
            self.write_buffer[norm_key] = self._DELETION_SENTINEL
            if len(self.write_buffer) >= self.batch_size:
                self._flush()

    def exists(self, key: Any, *, flush: bool | None = None) -> bool:
        """
        Check if a key exists in the database (including buffered writes).

        Parameters
        ----------
        key : Any
            The key to check for existence. Must be bytes-like or str
            (if key_encoding is set).
        flush : bool | None, optional
            Whether to flush the write buffer before checking.
            If None (default), uses the autoflush_on_read setting.
            If True, always flushes before checking.
            If False, never flushes before checking.

        Returns
        -------
        bool
            True if the key exists and is not marked for deletion, False otherwise.
        """
        norm_key = self._norm_key(key)
        with self.lock:
            self._ensure_open()
            if norm_key in self.write_buffer:
                return self.write_buffer[norm_key] is not self._DELETION_SENTINEL

            # Determine whether to flush based on parameter or default setting
            should_flush = self.autoflush_on_read if flush is None else flush
            if should_flush:
                self._flush()
            self._readers += 1

        try:
            with self.env.begin(buffers=True) as txn:
                return txn.get(norm_key) is not None
        finally:
            with self.lock:
                self._readers -= 1
                if self._readers == 0 and self._closing:
                    self.condition.notify_all()

    def flush(self):
        """
        Flush all buffered write and delete operations to the LMDB database.

        Raises
        ------
        lmdb.Error
            If the database is closed or in read-only mode.
        """
        with self.lock:
            self._ensure_open()
            self._ensure_writable()
            self._flush()

    def close(self, *, strict: bool = False):
        """
        Close the LMDB object store, flushing any pending writes and releasing
        resources.

        Parameters
        ----------
        strict : bool, optional
            If True, raises an exception if the final flush fails.
            If False (default), logs the error and continues closing.

        Raises
        ------
        RuntimeError
            If strict=True and the final flush fails.
        lmdb.Error
            If an error occurs during closing the environment.
        """
        flush_error: Exception | None = None

        with self.lock:
            if self._is_closed or self._closing:
                return

            self._closing = True

            if not self._readonly:
                try:
                    self._flush()
                except Exception as e:
                    log.error(f"Error during final flush on close: {e}", exc_info=True)
                    flush_error = e

            while self._readers > 0:
                self.condition.wait()

            try:
                self.env.sync()
            except Exception:
                log.warning("env.sync() failed during close()", exc_info=True)

            self.env.close()
            self._is_closed = True

        # Raise the flush error after cleanup if in strict mode
        if flush_error is not None and strict:
            raise RuntimeError("Final flush failed during close()") from flush_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, key: Any, value: Any):
        self.put(key, value)

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, default=self._DELETION_SENTINEL)
        if value is self._DELETION_SENTINEL:
            raise self._key_not_found_error(key)
        return value

    def __delitem__(self, key: Any):
        if not self.exists(key, flush=False):
            raise self._key_not_found_error(key)
        self.delete(key)

    def __contains__(self, key: Any) -> bool:
        return self.exists(key, flush=False)
