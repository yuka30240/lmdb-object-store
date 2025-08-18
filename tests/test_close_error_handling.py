"""
Comprehensive tests for close() error handling and exists() flush parameter in LmdbObjectStore.

Tests various error handling scenarios including:
- Close method error handling with strict and non-strict modes
- Flush error handling during close operations
- exists() method with explicit flush parameter control
- __contains__ behavior with flush parameter overrides
- Environment sync error handling during close
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import lmdb
import pytest

from lmdb_object_store import LmdbObjectStore


def make_path(tmp_path: Path, subdir: bool = True) -> Path:
    """Create a database path for testing."""
    if subdir:
        db_path = tmp_path / "db"
        db_path.mkdir(exist_ok=True)
        return db_path
    else:
        return tmp_path / "db.lmdb"


def test_close_strict_mode_raises_on_flush_error(tmp_path):
    """Test that close(strict=True) raises when flush fails."""
    path = make_path(tmp_path, subdir=False)

    db = LmdbObjectStore(str(path), subdir=False, batch_size=10, key_encoding="utf-8")

    # Add some data to the buffer
    db.put("key1", "value1")

    # Mock _flush to raise an error
    with patch.object(db, "_flush", side_effect=lmdb.MapFullError("Test error")):
        # strict=True should raise
        with pytest.raises(RuntimeError, match="Final flush failed during close"):
            db.close(strict=True)

    # Verify the database is still closed despite the error
    assert db._is_closed


def test_close_non_strict_mode_logs_error(tmp_path, caplog):
    """Test that close(strict=False) logs error but doesn't raise."""
    path = make_path(tmp_path, subdir=False)

    db = LmdbObjectStore(str(path), subdir=False, batch_size=10, key_encoding="utf-8")

    # Add some data to the buffer
    db.put("key1", "value1")

    # Mock _flush to raise an error
    with patch.object(db, "_flush", side_effect=Exception("Test flush error")):
        # strict=False (default) should not raise
        db.close(strict=False)

    # Check that error was logged
    assert "Error during final flush on close" in caplog.text
    assert "Test flush error" in caplog.text

    # Verify the database is closed
    assert db._is_closed


def test_exists_with_flush_parameter(tmp_path):
    """Test exists() method with explicit flush parameter."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),
        subdir=False,
        batch_size=10,
        key_encoding="utf-8",
        autoflush_on_read=True,  # Default behavior
    ) as db:
        # Add data to buffer
        db.put("key1", "value1")

        # Test with flush=True (explicit flush)
        exists_with_flush = db.exists("key1", flush=True)
        assert exists_with_flush is True

        # Add another key to buffer
        db.put("key2", "value2")

        # Test with flush=False (no flush)
        exists_no_flush = db.exists("key2", flush=False)
        assert exists_no_flush is True  # Should still find it in buffer

        # Test with flush=None (use default autoflush_on_read)
        db.put("key3", "value3")
        exists_default = db.exists("key3")  # Should use autoflush_on_read=True
        assert exists_default is True


def test_contains_uses_flush_false(tmp_path):
    """Test that __contains__ always uses flush=False."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),
        subdir=False,
        batch_size=10,
        key_encoding="utf-8",
        autoflush_on_read=True,  # Even with this True, __contains__ should not flush
    ) as db:
        # Add data to buffer
        db.put("key1", "value1")
        db.put("key2", "value2")

        # Mock _flush to verify it's not called
        flush_mock = MagicMock()
        with patch.object(db, "_flush", flush_mock):
            # Use __contains__ (in operator)
            assert "key1" in db
            assert "key2" in db
            assert "key3" not in db

            # _flush should not have been called
            flush_mock.assert_not_called()


def test_close_with_env_sync_error(tmp_path, caplog):
    """Test that env.sync() errors are logged but don't prevent closing."""
    path = make_path(tmp_path, subdir=False)

    # Create a real database
    db = LmdbObjectStore(str(path), subdir=False, batch_size=10, key_encoding="utf-8")

    # We can't mock env.sync directly because it's read-only in LMDB
    # Instead, we'll test that if env.sync fails, close() still completes
    # We'll simulate this by closing the env first, then calling close again
    db.env.close()
    db._is_closed = False  # Reset flag to test close() behavior

    # Now calling close should handle the error gracefully
    db.close(strict=False)

    # The database should be marked as closed even if sync failed
    assert db._is_closed


def test_exists_flush_behavior_with_autoflush_false(tmp_path):
    """Test exists() with autoflush_on_read=False."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),
        subdir=False,
        batch_size=10,
        key_encoding="utf-8",
        autoflush_on_read=False,  # Disable auto-flush
    ) as db:
        # Write and flush one key
        db.put("persisted", "value")
        db.flush()

        # Add to buffer without flushing
        db.put("buffered", "value")

        # With flush=None (default), should use autoflush_on_read=False
        assert db.exists("buffered")  # Found in buffer
        assert db.exists("persisted")  # Found in DB

        # Verify buffered item is not in DB yet
        with db.env.begin() as txn:
            assert txn.get(db._norm_key("buffered")) is None

        # Check a non-existent key with flush=True - this will trigger flush
        # since the key is not in buffer
        assert not db.exists("nonexistent", flush=True)

        # After the flush triggered by exists(), buffer should be empty
        assert len(db.write_buffer) == 0

        # Now buffered item should be in DB
        with db.env.begin() as txn:
            raw_value = txn.get(db._norm_key("buffered"))
            assert raw_value is not None  # Should be the pickled "value"
