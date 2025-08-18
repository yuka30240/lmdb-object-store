"""
Comprehensive tests for put_many bulk operations in LmdbObjectStore.

Tests various bulk insertion scenarios including:
- Atomic transactions with dictionaries and lists
- Duplicate key handling (last-write-wins)
- Map resizing during bulk operations
- Error handling and rollback behavior
- Integration with write buffer
- Various input types and edge cases
"""

import lmdb
import pytest

from lmdb_object_store import LmdbObjectStore


def make_path(tmp_path, subdir: bool):
    if subdir:
        p = tmp_path / "dbdir"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    else:
        return str(tmp_path / "dbfile")


def test_put_many_atomic_with_dict(tmp_path):
    """Test atomic put_many with dictionary input."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path), subdir=False, key_encoding="utf-8", batch_size=10
    ) as db:
        items = {
            "key1": "value1",
            "key2": {"nested": "data"},
            "key3": [1, 2, 3],
            "key4": 42,
        }

        # Atomic insert (default)
        db.put_many(items)

        # Verify all items were stored
        for key, value in items.items():
            assert db.get(key) == value

        # Verify they're in the database (not just buffer)
        with db.env.begin() as txn:
            assert txn.get(b"key1") is not None


def test_put_many_atomic_with_list_of_tuples(tmp_path):
    """Test atomic put_many with list of tuples input."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path), subdir=False, key_encoding="utf-8", batch_size=10
    ) as db:
        items = [
            ("key1", "value1"),
            ("key2", {"nested": "data"}),
            ("key3", [1, 2, 3]),
            ("key4", 42),
        ]

        db.put_many(items)

        # Verify all items were stored
        for key, value in items:
            assert db.get(key) == value


def test_put_many_always_atomic(tmp_path):
    """Test that put_many is always atomic (single transaction)."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),
        subdir=False,
        key_encoding="utf-8",
        batch_size=3,  # Small batch size doesn't affect put_many
    ) as db:
        items = {f"key{i}": f"value{i}" for i in range(10)}

        # Track flushes - put_many should flush buffer once at start
        flush_count = 0
        original_flush = db._flush

        def counting_flush():
            nonlocal flush_count
            flush_count += 1
            original_flush()

        db._flush = counting_flush

        # put_many always executes atomically
        db.put_many(items)

        # Should have triggered exactly 1 flush (initial buffer flush)
        # or 0 if buffer was empty
        assert flush_count <= 1

        # Verify all items were stored
        for key, value in items.items():
            assert db.get(key) == value


def test_put_many_duplicate_keys_last_wins(tmp_path):
    """Test that duplicate keys follow last-write-wins semantics."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(str(path), subdir=False, key_encoding="utf-8") as db:
        # List with duplicate keys
        items = [
            ("key1", "first"),
            ("key2", "value2"),
            ("key1", "second"),  # Duplicate
            ("key3", "value3"),
            ("key1", "third"),  # Another duplicate
        ]

        db.put_many(items)

        # Last value should win
        assert db.get("key1") == "third"
        assert db.get("key2") == "value2"
        assert db.get("key3") == "value3"


def test_put_many_empty_input(tmp_path):
    """Test put_many with empty input."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(str(path), subdir=False, key_encoding="utf-8") as db:
        # Should handle empty dict gracefully
        db.put_many({})

        # Should handle empty list gracefully
        db.put_many([])

        # Add one item to verify DB is working
        db.put("test", "value")
        assert db.get("test") == "value"


def test_put_many_with_map_resize(tmp_path):
    """Test that put_many handles MapFullError with automatic resize."""
    path = make_path(tmp_path, subdir=False)

    # Start with very small map
    with LmdbObjectStore(
        str(path),
        subdir=False,
        key_encoding="utf-8",
        map_size=1024 * 1024,  # 1MB
        max_map_size=100 * 1024 * 1024,  # 100MB max
        batch_size=100,
    ) as db:
        # Create large items that will exceed initial map size
        large_value = "x" * (100 * 1024)  # 100KB string
        items = {
            f"key{i}": large_value
            for i in range(20)  # 20 * 100KB = 2MB total
        }

        # put_many should resize and succeed
        db.put_many(items)

        # Verify all items were stored
        for key, _value in items.items():
            assert db.get(key) == large_value


def test_put_many_atomic_rollback_on_error(tmp_path):
    """Test that atomic put_many rolls back on error."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),
        subdir=False,
        key_encoding="utf-8",
        map_size=1024 * 1024,  # 1MB
        max_map_size=1024 * 1024,  # Can't resize
    ) as db:
        # First add a key
        db.put("existing", "value")
        db.flush()

        # Create items that will cause MapFullError
        large_value = "x" * (500 * 1024)  # 500KB
        items = {
            f"key{i}": large_value
            for i in range(5)  # Will exceed 1MB limit
        }

        # put_many should fail
        with pytest.raises(lmdb.MapFullError):
            db.put_many(items)

        # Original key should still exist
        assert db.get("existing") == "value"

        # New keys should not exist (transaction rolled back)
        for i in range(5):
            assert db.get(f"key{i}") is None


def test_put_many_with_bytes_keys(tmp_path):
    """Test put_many with bytes keys."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(str(path), subdir=False) as db:
        items = {
            b"key1": "value1",
            b"key2": {"data": 123},
            b"key3": [1, 2, 3],
        }

        db.put_many(items)

        for key, value in items.items():
            assert db.get(key) == value


def test_put_many_mixed_types(tmp_path):
    """Test that put_many() can handle diverse Python object types as values."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(str(path), subdir=False, key_encoding="utf-8") as db:
        items = {
            "string": "text value",
            "integer": 42,
            "float": 3.14159,
            "list": [1, 2, 3, 4, 5],
            "dict": {"nested": {"data": "structure"}},
            "tuple": (1, "two", 3.0),
            "none": None,
            "bool": True,
            "set": {1, 2, 3},  # Sets are pickleable
        }

        db.put_many(items)

        # Verify all types are correctly stored and retrieved
        assert db.get("string") == "text value"
        assert db.get("integer") == 42
        assert db.get("float") == 3.14159
        assert db.get("list") == [1, 2, 3, 4, 5]
        assert db.get("dict") == {"nested": {"data": "structure"}}
        assert db.get("tuple") == (1, "two", 3.0)
        assert db.get("none") is None
        assert db.get("bool") is True
        assert db.get("set") == {1, 2, 3}


def test_put_many_generator_input(tmp_path):
    """Test put_many with generator as input."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(str(path), subdir=False, key_encoding="utf-8") as db:
        # Generator of items
        def item_generator():
            for i in range(10):
                yield (f"key{i}", f"value{i}")

        db.put_many(item_generator())

        # Verify all items were stored
        for i in range(10):
            assert db.get(f"key{i}") == f"value{i}"


def test_put_many_preserves_existing_buffer(tmp_path):
    """Test that atomic put_many doesn't affect existing write_buffer."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path), subdir=False, key_encoding="utf-8", batch_size=100
    ) as db:
        # Add items to buffer
        db.put("buffered1", "value1")
        db.put("buffered2", "value2")

        # Buffer should have items
        assert len(db.write_buffer) == 2

        # put_many flushes buffer first, then executes atomically
        db.put_many({"atomic1": "avalue1", "atomic2": "avalue2"})

        # Buffer should be empty after put_many (it flushes first)
        assert len(db.write_buffer) == 0

        # Atomic items should be in DB
        assert db.get("atomic1") == "avalue1"
        assert db.get("atomic2") == "avalue2"

        # Buffered items should still be retrievable
        assert db.get("buffered1") == "value1"
        assert db.get("buffered2") == "value2"


def test_put_many_readonly_raises_error(tmp_path):
    """Test that put_many raises error in readonly mode."""
    path = make_path(tmp_path, subdir=False)

    # First create a database with some data
    with LmdbObjectStore(str(path), subdir=False) as db:
        db.put(b"initial", "value")

    # Open in readonly mode
    with LmdbObjectStore(str(path), subdir=False, readonly=True) as db:
        with pytest.raises(lmdb.Error):
            db.put_many({b"key1": "value1"})


def test_put_many_closed_db_raises_error(tmp_path):
    """Test that put_many raises error on closed database."""
    path = make_path(tmp_path, subdir=False)

    db = LmdbObjectStore(str(path), subdir=False, key_encoding="utf-8")
    db.close()

    with pytest.raises(lmdb.Error, match="closed"):
        db.put_many({"key": "value"})


def test_put_many_generator_with_map_resize(tmp_path):
    """Test put_many with generator input that exceeds initial map size."""
    path = str(tmp_path / "db")
    with LmdbObjectStore(
        path, key_encoding="utf-8", map_size=512 * 1024, max_map_size=8 * 1024 * 1024
    ) as db:

        def gen():
            for i in range(50):
                yield (f"k{i}", "x" * 20000)

        db.put_many(gen())
        for i in range(50):
            assert db.get(f"k{i}") == "x" * 20000
