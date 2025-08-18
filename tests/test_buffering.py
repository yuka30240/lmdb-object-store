"""
Comprehensive tests for write buffering behavior in LmdbObjectStore.

Tests various scenarios including:
- Buffer overflow and automatic flushing
- Mixed operations (put/delete) in buffer
- Buffer state consistency
- Edge cases around batch_size boundaries
"""

from lmdb_object_store import LmdbObjectStore


def make_path(tmp_path, subdir: bool):
    if subdir:
        p = tmp_path / "dbdir"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    else:
        return str(tmp_path / "dbfile")


def test_buffer_overflow_triggers_autoflush(tmp_path):
    """Test that buffer automatically flushes when reaching batch_size."""
    path = make_path(tmp_path, subdir=False)
    batch_size = 5

    with LmdbObjectStore(
        path,
        batch_size=batch_size,
        key_encoding="utf-8",
        autoflush_on_read=False,  # Ensure flushes are only from overflow
    ) as db:
        # Fill buffer exactly to batch_size - 1
        for i in range(batch_size - 1):
            db[f"key_{i}"] = f"value_{i}"

        assert len(db.write_buffer) == batch_size - 1

        # One more put should trigger flush
        db[f"key_{batch_size - 1}"] = f"value_{batch_size - 1}"
        assert len(db.write_buffer) == 0  # Buffer should be flushed

        # Verify all values were persisted
        for i in range(batch_size):
            assert db.get(f"key_{i}") == f"value_{i}"


def test_buffer_overflow_with_mixed_operations(tmp_path):
    """Test buffer overflow with mixed put and delete operations."""
    path = make_path(tmp_path, subdir=False)
    batch_size = 5

    with LmdbObjectStore(
        path, batch_size=batch_size, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Pre-populate some data
        for i in range(10):
            db[f"pre_{i}"] = i
        db.flush()

        # Mix puts and deletes to reach batch_size
        db["new_1"] = "value_1"  # 1
        db.delete("pre_0")  # 2
        db["new_2"] = "value_2"  # 3
        db.delete("pre_1")  # 4

        assert len(db.write_buffer) == 4

        # This should trigger flush
        db["new_3"] = "value_3"  # 5 -> triggers flush

        assert len(db.write_buffer) == 0

        # Verify state
        assert db.get("new_1") == "value_1"
        assert db.get("new_2") == "value_2"
        assert db.get("new_3") == "value_3"
        assert db.get("pre_0") is None
        assert db.get("pre_1") is None
        assert db.get("pre_2") == 2  # Should still exist


def test_buffer_state_with_overwrites(tmp_path):
    """Test buffer behavior when same key is written multiple times."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=10, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Multiple writes to same key
        db["key"] = "value_1"
        assert len(db.write_buffer) == 1

        db["key"] = "value_2"
        assert len(db.write_buffer) == 1  # Should still be 1

        db["key"] = "value_3"
        assert len(db.write_buffer) == 1

        # Should get latest value
        assert db.get("key") == "value_3"

        # Now delete it
        db.delete("key")
        assert len(db.write_buffer) == 1  # Still 1 entry
        assert db.get("key") is None

        # Write again after delete
        db["key"] = "value_4"
        assert len(db.write_buffer) == 1
        assert db.get("key") == "value_4"


def test_buffer_consistency_across_operations(tmp_path):
    """Test that buffer maintains consistency across various operations."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=20, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Write some initial data
        for i in range(5):
            db[f"key_{i}"] = f"initial_{i}"

        # Delete some
        db.delete("key_1")
        db.delete("key_3")

        # Overwrite some
        db["key_0"] = "updated_0"
        db["key_2"] = "updated_2"

        # Add new ones
        db["key_5"] = "new_5"
        db["key_6"] = "new_6"

        # Check buffer size (should have 7 unique keys)
        assert len(db.write_buffer) == 7

        # Verify get operations work correctly with buffer
        assert db.get("key_0") == "updated_0"
        assert db.get("key_1") is None
        assert db.get("key_2") == "updated_2"
        assert db.get("key_3") is None
        assert db.get("key_4") == "initial_4"
        assert db.get("key_5") == "new_5"
        assert db.get("key_6") == "new_6"

        # exists() should also work correctly
        assert db.exists("key_0") is True
        assert db.exists("key_1") is False
        assert db.exists("key_99") is False


def test_exact_batch_size_boundary(tmp_path):
    """Test behavior at exact batch_size boundary."""
    path = make_path(tmp_path, subdir=False)
    batch_size = 3

    with LmdbObjectStore(
        path, batch_size=batch_size, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Fill exactly to batch_size
        db["a"] = 1
        db["b"] = 2
        db["c"] = 3  # This should trigger flush

        assert len(db.write_buffer) == 0

        # Add more
        db["d"] = 4
        db["e"] = 5
        assert len(db.write_buffer) == 2

        # One more to trigger another flush
        db["f"] = 6
        assert len(db.write_buffer) == 0


def test_buffer_with_get_many(tmp_path):
    """Test that get_many() correctly handles mix of persisted and buffered data."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=20, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Some data in DB
        db["db_1"] = "value_db_1"
        db["db_2"] = "value_db_2"
        db.flush()

        # Some data in buffer
        db["buf_1"] = "value_buf_1"
        db["buf_2"] = "value_buf_2"

        # Delete one from DB (in buffer)
        db.delete("db_1")

        # get_many with mix of DB, buffer, and missing keys
        keys = [
            "db_1",
            "db_2",
            "buf_1",
            "buf_2",
            "missing_1",
            "buf_1",
        ]  # Note duplicate
        found, not_found = db.get_many(keys)

        assert found[b"db_2"] == "value_db_2"
        assert found[b"buf_1"] == "value_buf_1"
        assert found[b"buf_2"] == "value_buf_2"
        assert b"db_1" not in found  # Deleted

        assert not_found == [
            "db_1",
            "missing_1",
        ]  # Preserves order, no duplicates for found keys


def test_buffer_persistence_on_close(tmp_path):
    """Test that buffer is flushed on close."""
    path = make_path(tmp_path, subdir=False)

    # Write data and close without explicit flush
    db = LmdbObjectStore(
        path, batch_size=10, key_encoding="utf-8", autoflush_on_read=False
    )

    db["key_1"] = "value_1"
    db["key_2"] = "value_2"
    db["key_3"] = "value_3"

    assert len(db.write_buffer) == 3
    db.close()  # Should flush buffer

    # Reopen and verify data persisted
    with LmdbObjectStore(path, key_encoding="utf-8") as db2:
        assert db2.get("key_1") == "value_1"
        assert db2.get("key_2") == "value_2"
        assert db2.get("key_3") == "value_3"


def test_buffer_with_context_manager_exception(tmp_path):
    """Test that buffer is flushed even if exception occurs in context manager."""
    path = make_path(tmp_path, subdir=False)

    try:
        with LmdbObjectStore(
            path, batch_size=10, key_encoding="utf-8", autoflush_on_read=False
        ) as db:
            db["key_1"] = "value_1"
            db["key_2"] = "value_2"
            assert len(db.write_buffer) == 2
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected

    # Verify data was still persisted
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        assert db.get("key_1") == "value_1"
        assert db.get("key_2") == "value_2"


def test_large_batch_size_performance(tmp_path):
    """Test with large batch_size to verify buffer can grow appropriately."""
    path = make_path(tmp_path, subdir=False)
    batch_size = 1000

    with LmdbObjectStore(
        path, batch_size=batch_size, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        # Fill buffer with many items
        for i in range(batch_size - 1):
            db[f"key_{i:04d}"] = {"index": i, "data": f"value_{i}"}

        assert len(db.write_buffer) == batch_size - 1

        # Verify we can read from buffer
        assert db.get("key_0500")["index"] == 500

        # Trigger flush
        db[f"key_{batch_size - 1:04d}"] = {"index": batch_size - 1}
        assert len(db.write_buffer) == 0

        # Verify some samples persisted correctly
        assert db.get("key_0000")["index"] == 0
        assert db.get("key_0999")["index"] == 999


def test_autoflush_on_read_true_triggers_flush(tmp_path):
    """Test that reading with autoflush_on_read=True triggers a flush."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(
        path, subdir=False, key_encoding="utf-8", autoflush_on_read=True, batch_size=100
    ) as db:
        db["x"] = 1  # buffered
        # read another key → triggers flush because autoflush_on_read=True
        assert db.get("missing") is None
        # buffer should be empty now
        assert len(db.write_buffer) == 0
        # data should be persisted
        assert db.get("x") == 1


def test_autoflush_on_read_false_does_not_flush(tmp_path):
    """Test that reading with autoflush_on_read=False does not trigger a flush."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(
        path,
        subdir=False,
        key_encoding="utf-8",
        autoflush_on_read=False,
        batch_size=100,
    ) as db:
        db["x"] = 1  # buffered
        # read another key → must NOT flush
        assert db.get("missing") is None
        assert len(db.write_buffer) == 1
        # yet reading "x" must return from buffer
        assert db.get("x") == 1
