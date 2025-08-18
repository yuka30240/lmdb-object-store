"""
Comprehensive tests for multiple map resize retry behavior in LmdbObjectStore.

Tests various map resizing scenarios including:
- Multiple resize attempts during single flush operations
- Map size limit enforcement with max_map_size constraints
- Resize calculation logic with doubling and +64MB strategies
- Retry behavior when initial resize attempts fail
- Error handling when resize limits are exceeded
"""

from pathlib import Path

import pytest

from lmdb_object_store import LmdbObjectStore


def make_path(tmp_path: Path, subdir: bool = True) -> Path:
    """Create a database path for testing."""
    if subdir:
        db_path = tmp_path / "db"
        db_path.mkdir(exist_ok=True)
        return db_path
    else:
        # Return a non-existent file path
        return tmp_path / "db.lmdb"


def test_multiple_resize_retries(tmp_path):
    """Test that _flush() can handle multiple resize attempts when needed."""
    path = make_path(tmp_path, subdir=False)

    # Start with very small map size (512KB)
    with LmdbObjectStore(
        str(path),  # Convert Path to str
        subdir=False,
        key_encoding="utf-8",
        map_size=512 * 1024,  # 512KB initial
        max_map_size=20 * 1024 * 1024,  # 20MB max
        batch_size=10,
    ) as db:
        # Create data that will require multiple resizes
        # Each resize doubles or adds 64MB (whichever is larger)
        # From 512KB -> 64.5MB would require at least 2 resizes:
        # 512KB -> max(1MB, 64.5MB) = 64.5MB (but capped at 20MB)
        # So: 512KB -> 1MB -> 2MB -> 4MB -> 8MB -> 16MB -> 20MB (capped)

        # Add multiple large objects that together exceed initial size
        large_value = b"x" * (500 * 1024)  # 500KB per object

        # Add 8 objects = 4MB total, which requires multiple resizes from 512KB
        for i in range(8):
            db.put(f"large_{i}", large_value)

        # This flush should trigger multiple resize operations
        db.flush()

        # Verify all data was written successfully
        for i in range(8):
            assert db.get(f"large_{i}") == large_value


def test_multiple_resize_hits_limit(tmp_path):
    """Test that multiple resizes respect max_map_size limit."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        str(path),  # Convert Path to str
        subdir=False,
        key_encoding="utf-8",
        map_size=512 * 1024,  # 512KB initial
        max_map_size=1 * 1024 * 1024,  # 1MB max (very restrictive)
        batch_size=10,
    ) as db:
        # Try to write data that would require more space than max allows
        # This should trigger resize from 512KB -> 1MB, then fail
        large_value = b"x" * (600 * 1024)  # 600KB per object

        # Add 3 objects = 1.8MB total, exceeds max_map_size
        for i in range(3):
            db.put(f"large_{i}", large_value)

        # Should raise MapFullError after hitting max_map_size
        import lmdb

        with pytest.raises(lmdb.MapFullError):
            db.flush()


def test_resize_calculation_logic(tmp_path):
    """Test that resize calculation uses max(double, +64MB) logic."""
    path = make_path(tmp_path, subdir=False)

    # Test with initial size where doubling is less than +64MB
    with LmdbObjectStore(
        str(path),  # Convert Path to str
        subdir=False,
        key_encoding="utf-8",
        map_size=10 * 1024 * 1024,  # 10MB initial
        batch_size=5,
    ) as db:
        # Create data larger than 10MB to trigger resize
        # Next size should be max(20MB, 74MB) = 74MB
        large_value = b"x" * (15 * 1024 * 1024)  # 15MB
        db.put("large", large_value)
        db.flush()

        # Verify data was written (resize succeeded)
        assert db.get("large") == large_value

        # Check that map was resized appropriately
        info = db.env.info()
        # Should be at least 74MB (10MB + 64MB)
        assert info["map_size"] >= 74 * 1024 * 1024
