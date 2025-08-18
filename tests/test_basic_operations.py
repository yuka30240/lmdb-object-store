"""
Comprehensive tests for core CRUD operations and key handling in LmdbObjectStore.

Tests fundamental scenarios including:
- Basic CRUD operations (put, get, delete, exists)
- Key normalization (bytes, bytearray, memoryview, str)
- Unicode normalization (NFC, NFD)
- get_many with order preservation and duplicates
- Readonly mode restrictions
- Map size management and auto-resizing
- Mapping protocol and error messages
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


@pytest.mark.parametrize("subdir", [False, True])
@pytest.mark.parametrize("autoflush", [True, False])
def test_put_get_delete_exists_basic(tmp_path, subdir, autoflush):
    """Test core CRUD operations with parametrized subdir and autoflush configurations."""
    path = make_path(tmp_path, subdir)
    with LmdbObjectStore(
        path,
        batch_size=10,
        key_encoding="utf-8",
        subdir=subdir,
        autoflush_on_read=autoflush,
        map_size=2 * 1024 * 1024,
    ) as db:
        # put via __setitem__
        db["u:1"] = {"n": "Alice"}
        db["u:2"] = {"n": "Bob"}

        # exists (buffer hit)
        assert db.exists("u:1") is True

        # get buffered
        assert db["u:1"]["n"] == "Alice"

        # get default for missing
        assert db.get("u:999", default=None) is None

        # delete buffered and check exists
        db.delete("u:1")
        assert db.exists("u:1") is False

        # Force flush
        db.flush()

        # After flush, values persist/missing as expected
        assert db.get("u:2") == {"n": "Bob"}
        assert db.get("u:1", "DELETED") == "DELETED"

    # Reopen readonly and re-check persistence
    with LmdbObjectStore(
        path, readonly=True, key_encoding="utf-8", subdir=subdir
    ) as db:
        assert db.get("u:2") == {"n": "Bob"}
        assert db.get("u:1", "DELETED") == "DELETED"
        assert ("u:2" in db) is True
        assert ("u:1" in db) is False


def test_key_normalization_bytes_like_and_str(tmp_path):
    """Test key normalization for bytes, bytearray, memoryview, and str."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db:
        # bytes, bytearray, memoryview all normalize to same bytes
        key_b = b"k:1"
        key_ba = bytearray(b"k:1")
        key_mv = memoryview(b"k:1")
        key_s = "k:1"

        db.put(key_b, {"v": 1})
        assert db.get(key_b) == {"v": 1}
        assert db.get(key_ba) == {"v": 1}
        assert db.get(key_mv) == {"v": 1}
        assert db.get(key_s) == {"v": 1}


def test_key_normalization_str_forbidden_without_encoding(tmp_path):
    """Test that using str keys without key_encoding raises TypeError."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False) as db, pytest.raises(TypeError):
        db.put("str-key", 1)


def test_unicode_normalization_optional(tmp_path):
    """Test that unicode normalization is optional and works correctly."""
    c_composed = "é"  # U+00E9
    c_decomposed = "e\u0301"  # U+0065 U+0301

    # No normalization: keys differ
    path1 = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(
        path1, subdir=False, key_encoding="utf-8", str_normalize=None
    ) as db:
        db.put(c_decomposed, {"x": 1})
        assert db.get(c_decomposed) == {"x": 1}
        assert db.get(c_composed) is None  # different bytes

    # With NFC normalization: keys unify
    path2 = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(
        path2, subdir=False, key_encoding="utf-8", str_normalize="NFC"
    ) as db:
        db.put(c_decomposed, {"x": 2})
        assert db.get(c_composed) == {"x": 2}


def test_get_many_order_and_duplicates(tmp_path):
    """Test that get_many() preserves order and handles duplicate keys correctly."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db:
        db["a"] = 1
        db["b"] = 2
        db.flush()

        # include duplicates and missing
        keys = ["a", "missing_z", "b", "a", "missing_a", "missing_z"]
        found, not_found = db.get_many(keys)

        # found's keys are bytes
        assert all(isinstance(k, bytes) for k in found)
        # values correct
        assert found[b"a"] == 1
        assert found[b"b"] == 2

        # not_found preserves order and duplication
        assert not_found == ["missing_z", "missing_a", "missing_z"]


def test_get_many_decoding_variants(tmp_path):
    """Test get_many with different decoding options."""
    path = str(tmp_path / "db")
    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        db["a"] = 1
        db.flush()

        found, not_found = db.get_many(["a", "x"], decode_keys=True)
        assert list(found.keys()) == ["a"]
        assert not_found == ["x"]

        found, not_found = db.get_many(
            [b"a", b"y"], decode_keys=False, decode_not_found=True
        )
        assert list(found.keys()) == [b"a"]
        assert not_found == ["y"]

    with LmdbObjectStore(str(tmp_path / "db2")) as db2, pytest.raises(ValueError):
        db2.get_many([b"a"], decode_keys=True)


def test_readonly_guard(tmp_path):
    """Test readonly mode restrictions."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db:
        db["k"] = 1
        db.flush()

    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8", readonly=True) as ro:
        assert ro.get("k") == 1
        with pytest.raises(lmdb.Error):
            ro.put("k2", 2)
        with pytest.raises(lmdb.Error):
            ro.delete("k")
        with pytest.raises(lmdb.Error):
            ro.flush()


def test_mapfull_auto_resize_allows_large_write(tmp_path):
    """Test that mapfull error triggers auto-resize and allows large write."""
    path = make_path(tmp_path, subdir=False)
    # tiny map to force resize
    with LmdbObjectStore(
        path, subdir=False, key_encoding="utf-8", map_size=1 * 1024 * 1024
    ) as db:
        big_value = b"x" * (2 * 1024 * 1024)  # 2 MiB
        db.put("big", big_value)  # buffered
        # Force flush (should trigger MapFullError internally, then resize+retry)
        db.flush()
        # Confirm persisted
        got = db.get("big")
        assert isinstance(got, bytes) and len(got) == len(big_value)


def test_mapfull_max_map_size_cap_raises(tmp_path):
    """Test that hitting max_map_size cap raises MapFullError."""
    path = make_path(tmp_path, subdir=False)
    # start with 1 MiB and cap at 1 MiB -> resize forbidden
    with LmdbObjectStore(
        path,
        subdir=False,
        key_encoding="utf-8",
        map_size=1 * 1024 * 1024,
        max_map_size=1 * 1024 * 1024,
    ) as db:
        big_value = b"x" * (2 * 1024 * 1024)
        db.put("big", big_value)
        with pytest.raises(lmdb.MapFullError):
            db.flush()  # cannot resize -> re-raise MapFullError


def test_mapping_protocol_and_error_messages(tmp_path):
    """Test mapping protocol and error messages for key not found."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False) as db:
        # Test error message contains key information for binary keys
        bad_key = b"\xff\xfe"
        with pytest.raises(KeyError) as ei:
            _ = db[bad_key]

        error_msg = str(ei.value)
        # Check that error message is informative
        assert "Key not found" in error_msg
        assert "0x" in error_msg  # Binary key shown as hex

        # Test with string key (using key encoding)
        with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db_str:
            with pytest.raises(KeyError) as ei:
                _ = db_str["missing_string_key"]

            error_msg = str(ei.value)
            assert "Key not found" in error_msg
            assert "missing_string_key" in error_msg

        # normal set/get/contains/del
        key = b"k"
        db[key] = {"v": 1}
        assert key in db
        assert db[key] == {"v": 1}
        del db[key]
        assert key not in db
