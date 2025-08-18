"""
Comprehensive tests for edge cases and error scenarios in LmdbObjectStore.

Tests various edge cases including:
- Invalid inputs and type errors
- Large data handling
- Special characters and binary data
- Error recovery
- Resource cleanup
"""

import tempfile

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


def test_none_key_handling(tmp_path):
    """Test that None keys are properly rejected."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # None key should raise TypeError
        with pytest.raises(TypeError, match="Key cannot be None"):
            db.put(None, "value")

        with pytest.raises(TypeError, match="Key cannot be None"):
            db.get(None)

        with pytest.raises(TypeError, match="Key cannot be None"):
            db[None] = "value"

        with pytest.raises(TypeError, match="Key cannot be None"):
            _ = db[None]

        with pytest.raises(TypeError, match="Key cannot be None"):
            db.delete(None)

        with pytest.raises(TypeError, match="Key cannot be None"):
            db.exists(None)


def test_invalid_key_types(tmp_path):
    """Test rejection of invalid key types."""
    path = make_path(tmp_path, subdir=False)

    # Without key_encoding, only bytes-like allowed
    with LmdbObjectStore(path) as db:
        # These should work
        db[b"bytes"] = 1
        db[bytearray(b"bytearray")] = 2
        db[memoryview(b"memoryview")] = 3

        # These should fail
        with pytest.raises(TypeError, match="str keys are not allowed"):
            db["string"] = 4

        with pytest.raises(TypeError, match="Key must be bytes-like"):
            db[123] = 5

        with pytest.raises(TypeError, match="Key must be bytes-like"):
            db[{"dict": "key"}] = 6

        with pytest.raises(TypeError, match="Key must be bytes-like"):
            db[[1, 2, 3]] = 7


def test_empty_keys(tmp_path):
    """Test handling of empty keys."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # Empty bytes key
        db[b""] = "empty_bytes"
        assert db[b""] == "empty_bytes"

        # Empty string key
        db[""] = "empty_string"
        assert db[""] == "empty_string"

        # They should be the same (both encode to b"")
        assert db[b""] == "empty_string"


def test_very_long_keys(tmp_path):
    """Test handling of very long keys (LMDB has key size limits)."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # LMDB typically has a max key size of 511 bytes
        long_key = "k" * 500  # Should work
        db[long_key] = "value"
        assert db[long_key] == "value"

        # Very long key might fail
        very_long_key = "k" * 1000
        try:
            db[very_long_key] = "value"
            # If it succeeds, we should be able to read it
            assert db[very_long_key] == "value"
        except (lmdb.BadValsizeError, ValueError):
            # Expected for keys that are too long
            pass


def test_special_characters_in_keys(tmp_path):
    """Test keys with special characters."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        special_keys = [
            "key\x00null",  # Null byte
            "key\ttab",  # Tab
            "key\nnewline",  # Newline
            "key\r\nwindows",  # Windows newline
            "🔑🗝️",  # Emoji
            "key/with/slashes",
            "key\\with\\backslashes",
            "key with spaces",
            "key[with]brackets",
            "key{with}braces",
            "key|with|pipes",
        ]

        for i, key in enumerate(special_keys):
            db[key] = f"value_{i}"
            assert db[key] == f"value_{i}"


def test_unpickleable_objects(tmp_path):
    """Test storing objects that can't be pickled."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # Lambda functions can't be pickled
        with pytest.raises(Exception):
            db["lambda"] = lambda x: x + 1

        # Open file handles can't be pickled
        with tempfile.NamedTemporaryFile() as f, pytest.raises(Exception):
            db["file"] = f


def test_large_values(tmp_path):
    """Test that large values (KB to MB range) can be stored and retrieved correctly."""
    path = make_path(tmp_path, subdir=False)

    # Start with reasonable map size
    with LmdbObjectStore(
        path,
        key_encoding="utf-8",
        map_size=50 * 1024 * 1024,  # 50MB
        max_map_size=100 * 1024 * 1024,  # 100MB limit
    ) as db:
        # Store progressively larger values
        sizes = [
            1024,  # 1KB
            1024 * 1024,  # 1MB
            10 * 1024 * 1024,  # 10MB
        ]

        for size in sizes:
            key = f"large_{size}"
            # Create large object (list of integers)
            large_value = list(range(size // 4))  # Roughly size bytes when pickled

            db[key] = large_value
            db.flush()  # Force write to test map sizing

            # Verify we can read it back
            retrieved = db[key]
            assert len(retrieved) == len(large_value)
            assert retrieved[0] == 0
            assert retrieved[-1] == large_value[-1]


def test_corrupted_pickle_data_in_buffer(tmp_path):
    """Test handling of corrupted pickle data in write buffer."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # Manually insert corrupted data into buffer
        db.write_buffer[b"corrupted"] = b"NOT_A_VALID_PICKLE"

        # Should raise when trying to read
        with pytest.raises(RuntimeError, match="Failed to unpickle buffered key"):
            _ = db["corrupted"]


def test_database_reopening_patterns(tmp_path):
    """Test various patterns of closing and reopening database."""
    path = make_path(tmp_path, subdir=False)

    # Pattern 1: Write, close, reopen, read
    db1 = LmdbObjectStore(path, key_encoding="utf-8")
    db1["key1"] = "value1"
    db1.close()

    db2 = LmdbObjectStore(path, key_encoding="utf-8")
    assert db2["key1"] == "value1"
    db2.close()

    # Pattern 2: Multiple reopens
    for i in range(5):
        with LmdbObjectStore(path, key_encoding="utf-8") as db:
            db[f"key_{i}"] = f"value_{i}"
            assert db["key1"] == "value1"  # Original still there

    # Pattern 3: Readonly after write
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        db["final"] = "value"

    with LmdbObjectStore(path, key_encoding="utf-8", readonly=True) as db:
        assert db["final"] == "value"
        assert (
            len(
                [
                    k
                    for k in ["key_0", "key_1", "key_2", "key_3", "key_4", "final"]
                    if db.exists(k)
                ]
            )
            == 6
        )


def test_context_manager_exception_handling(tmp_path):
    """Test context manager handles exceptions properly."""
    path = make_path(tmp_path, subdir=False)

    class CustomError(Exception):
        pass

    # Test exception during operation
    try:
        with LmdbObjectStore(path, key_encoding="utf-8") as db:
            db["key1"] = "value1"
            raise CustomError("Test error")
    except CustomError:
        pass

    # Database should still be usable and data should be saved
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        assert db["key1"] == "value1"

    # Test exception in __enter__
    # Create a database with invalid path to trigger error
    try:
        with LmdbObjectStore(
            "/invalid/path/that/does/not/exist", key_encoding="utf-8"
        ) as db:
            pass
    except (lmdb.Error, OSError):
        pass  # Expected


def test_get_many_edge_cases(tmp_path):
    """Test edge cases for get_many method."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        db["a"] = 1
        db["b"] = 2
        db.flush()

        # Empty keys list
        found, not_found = db.get_many([])
        assert found == {}
        assert not_found == []

        # All missing keys
        found, not_found = db.get_many(["x", "y", "z"])
        assert found == {}
        assert not_found == ["x", "y", "z"]

        # Lots of duplicates
        keys = ["a"] * 100 + ["missing"] * 100
        found, not_found = db.get_many(keys)
        assert found == {b"a": 1}
        assert not_found == ["missing"] * 100

        # Mixed key types when key_encoding is set
        db[b"bytes_key"] = "bytes_value"
        found, not_found = db.get_many(["a", b"bytes_key", "missing"])
        assert found == {b"a": 1, b"bytes_key": "bytes_value"}
        assert not_found == ["missing"]


def test_unicode_edge_cases(tmp_path):
    """Test Unicode edge cases."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, key_encoding="utf-8", key_errors="strict", str_normalize="NFC"
    ) as db:
        # Various Unicode scenarios
        test_cases = [
            ("ascii", "simple ascii"),
            ("émoji🎉", "emoji and accents"),
            ("한글", "Korean"),
            ("中文", "Chinese"),
            ("עברית", "Hebrew"),
            ("العربية", "Arabic"),
            ("\u200b\u200c\u200d", "zero-width characters"),
            ("a\u0301", "combining characters"),  # à in decomposed form
        ]

        for key, value in test_cases:
            db[key] = value
            assert db[key] == value

        # Test normalization worked
        # a\u0301 (a + combining acute accent) normalizes to á (precomposed)
        assert db["á"] == "combining characters"  # NFC normalized form
        # Also test that the decomposed form works
        assert db["a\u0301"] == "combining characters"  # Should find same entry


def test_map_resize_limits(tmp_path):
    """Test map resizing with various limit configurations."""
    path = make_path(tmp_path, subdir=False)

    # Test 1: No max_map_size (should resize freely)
    with LmdbObjectStore(
        path,
        map_size=1024 * 1024,  # 1MB
        key_encoding="utf-8",
    ) as db:
        # This should trigger resize
        large_data = "x" * (2 * 1024 * 1024)
        db["large"] = large_data
        db.flush()
        assert db["large"] == large_data

    # Test 2: max_map_size equals initial size (no resize allowed)
    db2_path = tmp_path / "db2"
    db2_path.mkdir(exist_ok=True)
    path2 = make_path(db2_path, subdir=False)
    with LmdbObjectStore(
        path2,
        map_size=1024 * 1024,  # 1MB
        max_map_size=1024 * 1024,  # Same as initial
        key_encoding="utf-8",
    ) as db:
        large_data = "x" * (2 * 1024 * 1024)
        db["large"] = large_data

        with pytest.raises(lmdb.MapFullError):
            db.flush()


def test_readonly_operations_comprehensive(tmp_path):
    """Comprehensive test of readonly mode restrictions."""
    path = make_path(tmp_path, subdir=False)

    # First create some data
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        db["key1"] = "value1"
        db["key2"] = "value2"

    # Open readonly and test all write operations fail
    with LmdbObjectStore(path, readonly=True, key_encoding="utf-8") as db:
        # Read operations should work
        assert db.get("key1") == "value1"
        assert db.exists("key1") is True
        assert "key1" in db
        found, not_found = db.get_many(["key1", "key2", "missing"])
        assert len(found) == 2

        # All write operations should fail
        with pytest.raises(lmdb.Error):
            db.put("new", "value")

        with pytest.raises(lmdb.Error):
            db["new"] = "value"

        with pytest.raises(lmdb.Error):
            db.delete("key1")

        with pytest.raises(lmdb.Error):
            del db["key1"]

        with pytest.raises(lmdb.Error):
            db.flush()


def test_double_close(tmp_path):
    """Test that calling close() multiple times is safe."""
    path = make_path(tmp_path, subdir=False)

    db = LmdbObjectStore(path, key_encoding="utf-8")
    db["key"] = "value"

    # First close
    db.close()

    # Second close should be safe (no-op)
    db.close()

    # Third close should still be safe
    db.close()

    # Operations should fail after close
    with pytest.raises(lmdb.Error):
        db["key2"] = "value2"


def test_operations_after_close_error(tmp_path):
    """Test that operations after close raise lmdb.Error."""
    path = make_path(tmp_path, subdir=False)
    db = LmdbObjectStore(path, subdir=False, key_encoding="utf-8")
    db["k"] = 1
    db.close()
    with pytest.raises(lmdb.Error):
        db.put("x", 2)
    with pytest.raises(lmdb.Error):
        db.flush()


def test_unpickle_failure_raises_runtimeerror(tmp_path):
    """Test that unpickling failure raises RuntimeError with key info."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db:
        # Write invalid (non-pickle) bytes directly via LMDB to simulate corruption
        with db.env.begin(write=True) as txn:
            txn.put(b"bad", b"NOT_A_PICKLE")
        with pytest.raises(RuntimeError):
            _ = db.get("bad")


def test_empty_str_key_without_encoding_is_rejected(tmp_path):
    """Test that empty string key without encoding raises TypeError."""
    path = str(tmp_path / "db")
    with LmdbObjectStore(path) as db, pytest.raises(TypeError):
        db.put("", 1)
