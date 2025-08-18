"""
Comprehensive tests for deletion sentinel behavior in LmdbObjectStore.

Tests various deletion scenarios including:
- Deletion sentinel handling in buffer
- Interactions between deletions and reads
- Deletion persistence
- Edge cases with deletions
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


def test_deletion_sentinel_basic(tmp_path):
    """Test basic deletion sentinel behavior."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=10, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Add and immediately delete
        db["key1"] = "value1"
        db.delete("key1")

        # Should not exist
        assert db.get("key1") is None
        assert db.exists("key1") is False
        assert "key1" not in db

        # Buffer should contain deletion sentinel
        assert len(db.write_buffer) == 1
        assert db.write_buffer[b"key1"] is db._DELETION_SENTINEL


def test_delete_nonexistent_key(tmp_path):
    """Test that delete() handles non-existent keys gracefully while del raises KeyError."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, batch_size=10, key_encoding="utf-8") as db:
        # Delete non-existent key using delete()
        db.delete("nonexistent")  # Should not raise

        # But using del should raise KeyError
        with pytest.raises(KeyError) as exc_info:
            del db["nonexistent"]
        assert "Key not found" in str(exc_info.value)


def test_deletion_then_recreation(tmp_path):
    """Test deleting a key then recreating it."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=10, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Create, delete, recreate in buffer
        db["key"] = "value1"
        db.delete("key")
        db["key"] = "value2"

        # Should get new value
        assert db.get("key") == "value2"
        assert db.exists("key") is True

        # Flush and verify
        db.flush()
        assert db.get("key") == "value2"


def test_deletion_across_flush_boundaries(tmp_path):
    """Test deletion behavior across flush boundaries."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, batch_size=10, key_encoding="utf-8") as db:
        # Write to DB
        db["key1"] = "value1"
        db["key2"] = "value2"
        db.flush()

        # Delete key1 (goes to buffer)
        db.delete("key1")
        assert db.exists("key1") is False  # Should check buffer

        # Flush deletion
        db.flush()

        # Verify deletion persisted
        assert db.get("key1") is None
        assert db.get("key2") == "value2"


def test_deletion_with_get_many(tmp_path):
    """Test get_many with deletion sentinels."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=20, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Setup: some in DB, some in buffer
        db["a"] = 1
        db["b"] = 2
        db["c"] = 3
        db.flush()

        # Delete some, add new ones
        db.delete("a")  # Delete from DB
        db.delete("x")  # Delete non-existent
        db["d"] = 4  # New in buffer
        db.delete("d")  # Delete from buffer
        db["e"] = 5  # New in buffer

        # get_many should handle all cases correctly
        found, not_found = db.get_many(["a", "b", "c", "d", "e", "x", "y"])

        assert found == {b"b": 2, b"c": 3, b"e": 5}
        assert set(not_found) == {"a", "d", "x", "y"}


def test_deletion_sentinel_not_leaked(tmp_path):
    """Ensure deletion sentinel is never returned to user."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, batch_size=10, key_encoding="utf-8") as db:
        db["key"] = "value"
        db.delete("key")

        # Various ways to access should all show key as missing
        assert db.get("key") is None
        assert db.get("key", "default") == "default"

        with pytest.raises(KeyError):
            _ = db["key"]

        assert "key" not in db
        assert db.exists("key") is False


def test_mass_deletion_pattern(tmp_path):
    """Test pattern of mass deletion followed by selective recreation."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, batch_size=10, key_encoding="utf-8") as db:
        # Create initial dataset
        for i in range(20):
            db[f"item_{i:02d}"] = f"value_{i}"
        db.flush()

        # Mass delete even items
        for i in range(0, 20, 2):
            db.delete(f"item_{i:02d}")

        # Recreate some deleted items with new values
        db["item_00"] = "new_value_00"
        db["item_10"] = "new_value_10"

        # Flush all changes
        db.flush()

        # Verify final state
        assert db.get("item_00") == "new_value_00"
        assert db.get("item_01") == "value_1"
        assert db.get("item_02") is None  # Deleted
        assert db.get("item_10") == "new_value_10"
        assert db.get("item_11") == "value_11"
        assert db.get("item_12") is None  # Deleted


def test_deletion_with_readonly_mode(tmp_path):
    """Test that deletions are blocked in readonly mode."""
    path = make_path(tmp_path, subdir=False)

    # First, create some data
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        db["key1"] = "value1"
        db["key2"] = "value2"

    # Open in readonly mode
    with LmdbObjectStore(path, readonly=True, key_encoding="utf-8") as db:
        # Verify data exists
        assert db.get("key1") == "value1"

        # Try to delete - should fail
        with pytest.raises(lmdb.Error):
            db.delete("key1")

        with pytest.raises(lmdb.Error):
            del db["key1"]


def test_complex_deletion_sequence(tmp_path):
    """Test complex sequence of operations with deletions."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path,
        batch_size=5,  # Small batch to trigger multiple flushes
        key_encoding="utf-8",
        autoflush_on_read=False,
    ) as db:
        # Operation sequence that tests various edge cases
        operations = [
            ("put", "a", 1),
            ("put", "b", 2),
            ("delete", "a", None),
            ("put", "c", 3),
            ("put", "a", 4),  # Recreate deleted key
            ("flush", None, None),  # Manual flush
            ("delete", "b", None),  # Delete from DB
            ("put", "d", 5),
            ("delete", "d", None),  # Delete from buffer
            ("put", "e", 6),
            ("put", "f", 7),  # Should trigger auto-flush
            ("delete", "c", None),  # Delete from DB after flush
            ("put", "g", 8),
        ]

        for op, key, value in operations:
            if op == "put":
                db[key] = value
            elif op == "delete":
                db.delete(key)
            elif op == "flush":
                db.flush()

        # Final state check
        expected = {
            "a": 4,  # Recreated after delete
            "b": None,  # Deleted
            "c": None,  # Deleted
            "d": None,  # Deleted before flush
            "e": 6,
            "f": 7,
            "g": 8,
        }

        for key, expected_value in expected.items():
            assert db.get(key) == expected_value


def test_deletion_sentinel_with_batch_overflow(tmp_path):
    """Test deletion sentinels when batch overflows."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path,
        batch_size=3,  # Very small batch
        key_encoding="utf-8",
        autoflush_on_read=False,
    ) as db:
        # This sequence should cause overflow with mixed operations
        db["a"] = 1  # 1
        db.delete("a")  # Still 1 (same key)
        db["b"] = 2  # 2
        db["c"] = 3  # 3 - should trigger flush

        # After flush, buffer should be empty
        assert len(db.write_buffer) == 0

        # Verify "a" was properly deleted
        assert db.get("a") is None
        assert db.get("b") == 2
        assert db.get("c") == 3
