"""
Comprehensive tests for concurrent operations in LmdbObjectStore.

Tests various multi-threading scenarios including:
- Concurrent reads and writes
- Reader-writer lock behavior
- Close coordination with active operations
- Race conditions and thread safety
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def test_concurrent_reads(tmp_path):
    """Test multiple concurrent read operations."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # Populate data
        for i in range(100):
            db[f"key_{i:03d}"] = f"value_{i}"
        db.flush()

        # Concurrent reads
        results = {}
        errors = []

        def reader(key_num):
            try:
                key = f"key_{key_num:03d}"
                value = db.get(key)
                return key_num, value
            except Exception as e:
                errors.append((key_num, e))
                return key_num, None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(reader, i) for i in range(100)]
            for future in as_completed(futures):
                key_num, value = future.result()
                results[key_num] = value

        # Verify all reads succeeded
        assert len(errors) == 0
        assert len(results) == 100
        for i in range(100):
            assert results[i] == f"value_{i}"


def test_concurrent_writes(tmp_path):
    """Test multiple concurrent write operations."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, batch_size=10, key_encoding="utf-8") as db:
        errors = []

        def writer(thread_id, count):
            try:
                for i in range(count):
                    db[f"thread_{thread_id}_key_{i}"] = {
                        "thread": thread_id,
                        "index": i,
                        "data": f"value_{thread_id}_{i}",
                    }
            except Exception as e:
                errors.append((thread_id, e))

        # Launch multiple writer threads
        threads = []
        num_threads = 10
        writes_per_thread = 20

        for i in range(num_threads):
            t = threading.Thread(target=writer, args=(i, writes_per_thread))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Ensure no errors
        assert len(errors) == 0

        # Force flush
        db.flush()

        # Verify all data was written
        for thread_id in range(num_threads):
            for i in range(writes_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = db.get(key)
                assert value["thread"] == thread_id
                assert value["index"] == i


def test_concurrent_read_write_mix(tmp_path):
    """Test mixed concurrent reads and writes."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path,
        batch_size=5,
        key_encoding="utf-8",
        autoflush_on_read=True,  # This will cause more contention
    ) as db:
        # Pre-populate some data
        for i in range(50):
            db[f"existing_{i}"] = i
        db.flush()

        results = {"reads": [], "writes": [], "errors": []}

        def reader_worker():
            try:
                for _ in range(30):
                    key_num = random.randint(0, 49)
                    value = db.get(f"existing_{key_num}")
                    if value == key_num:
                        results["reads"].append(("existing", key_num))

                    # Also try to read newly written keys
                    new_key_num = random.randint(0, 49)
                    new_value = db.get(f"new_{new_key_num}")
                    if new_value is not None:
                        results["reads"].append(("new", new_key_num))

                    time.sleep(0.001)  # Small delay
            except Exception as e:
                results["errors"].append(("reader", e))

        def writer_worker(worker_id):
            try:
                for i in range(20):
                    db[f"new_{worker_id}_{i}"] = f"value_{worker_id}_{i}"
                    results["writes"].append((worker_id, i))
                    time.sleep(0.002)  # Small delay
            except Exception as e:
                results["errors"].append(("writer", worker_id, e))

        # Launch mixed workers
        threads = []

        # Readers
        for _ in range(5):
            t = threading.Thread(target=reader_worker)
            threads.append(t)
            t.start()

        # Writers
        for i in range(3):
            t = threading.Thread(target=writer_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Verify no errors
        assert len(results["errors"]) == 0

        # Verify writes completed
        assert len(results["writes"]) == 3 * 20  # 3 writers, 20 writes each


def test_close_with_multiple_active_readers(tmp_path):
    """Test close() waiting for multiple active readers."""
    path = make_path(tmp_path, subdir=False)

    db = LmdbObjectStore(path, key_encoding="utf-8")
    db["key"] = "value"
    db.flush()

    close_count = 0
    close_lock = threading.Lock()

    def closer():
        nonlocal close_count
        db.close()
        with close_lock:
            close_count += 1

    # Multiple threads trying to close simultaneously
    threads = []
    for _i in range(5):
        t = threading.Thread(target=closer)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # All threads should complete without error
    assert close_count == 5

    # DB should be closed
    with pytest.raises(lmdb.Error):
        db.get("key")


def test_concurrent_flush_operations(tmp_path):
    """Test concurrent operations during flush."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path, batch_size=20, key_encoding="utf-8", autoflush_on_read=False
    ) as db:
        results = {"flush_count": 0, "errors": []}
        flush_lock = threading.Lock()

        def writer_with_flush(worker_id):
            try:
                for i in range(10):
                    db[f"w_{worker_id}_{i}"] = i

                    # Occasionally flush
                    if i % 3 == 0:
                        with flush_lock:
                            results["flush_count"] += 1
                        db.flush()

                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(("writer", worker_id, e))

        def reader(worker_id):
            try:
                for _ in range(20):
                    key = f"w_{random.randint(0, 2)}_{random.randint(0, 9)}"
                    db.get(key)
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(("reader", worker_id, e))

        threads = []

        # Writers that flush
        for i in range(3):
            t = threading.Thread(target=writer_with_flush, args=(i,))
            threads.append(t)
            t.start()

        # Readers
        for i in range(2):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify no errors despite concurrent flushes
        assert len(results["errors"]) == 0
        assert results["flush_count"] > 0


def test_reader_writer_lock_fairness(tmp_path):
    """Test that readers don't starve writers and vice versa."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path,
        batch_size=100,  # Large batch to avoid auto-flush
        key_encoding="utf-8",
        autoflush_on_read=True,
    ) as db:
        # Pre-populate
        for i in range(10):
            db[f"key_{i}"] = i
        db.flush()

        operation_log = []
        log_lock = threading.Lock()

        def continuous_reader(reader_id, duration):
            end_time = time.time() + duration
            read_count = 0

            while time.time() < end_time:
                key = f"key_{random.randint(0, 9)}"
                db.get(key)
                read_count += 1

                with log_lock:
                    operation_log.append(("read", reader_id, time.time()))

                time.sleep(0.0001)  # Very short sleep

            return read_count

        def continuous_writer(writer_id, duration):
            end_time = time.time() + duration
            write_count = 0

            while time.time() < end_time:
                db[f"new_{writer_id}_{write_count}"] = write_count
                write_count += 1

                with log_lock:
                    operation_log.append(("write", writer_id, time.time()))

                time.sleep(0.0005)  # Slightly longer sleep

            return write_count

        # Run readers and writers concurrently
        duration = 0.5  # seconds

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []

            # 3 readers
            for i in range(3):
                futures.append(executor.submit(continuous_reader, i, duration))

            # 2 writers
            for i in range(2):
                futures.append(executor.submit(continuous_writer, i, duration))

            # Wait for completion
            [f.result() for f in futures]

        # Analyze operation log to ensure both reads and writes occurred
        read_ops = [op for op in operation_log if op[0] == "read"]
        write_ops = [op for op in operation_log if op[0] == "write"]

        # Both should have substantial operations
        assert len(read_ops) > 20
        assert len(write_ops) > 10

        # Check interleaving - writes shouldn't be completely blocked
        # Look for writes that happened between reads
        interleaved_writes = 0
        for i, (op_type, _, _timestamp) in enumerate(operation_log):
            # Check if there are reads before and after this write
            if (
                op_type == "write"
                and i > 0
                and i < len(operation_log) - 1
                and operation_log[i - 1][0] == "read"
                and operation_log[i + 1][0] == "read"
            ):
                interleaved_writes += 1

        assert interleaved_writes > 0  # Writers weren't starved


def test_concurrent_get_many(tmp_path):
    """Test that multiple threads can safely call get_many() concurrently."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(path, key_encoding="utf-8") as db:
        # Populate data
        for i in range(200):
            db[f"key_{i:03d}"] = {"value": i, "data": f"data_{i}"}
        db.flush()

        errors = []

        def get_many_worker(worker_id):
            try:
                # Each worker gets a different subset
                start = worker_id * 20
                keys = [f"key_{i:03d}" for i in range(start, start + 30)]
                keys.append("missing_key")  # Add some missing keys

                found, not_found = db.get_many(keys)

                # Verify results
                # Keys 0-199 exist, so check how many are in our range
                expected_found = 0
                for i in range(start, start + 30):
                    if i < 200:  # We created 200 keys
                        expected_found += 1

                assert len(found) == expected_found
                assert "missing_key" in not_found

                return worker_id, len(found)
            except Exception as e:
                errors.append((worker_id, e))
                return worker_id, -1

        # Run concurrent get_many
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_many_worker, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(errors) == 0
        # Verify each worker found the expected number of keys
        for i in range(10):
            worker_id, found_count = results[i]
            assert worker_id == i
            # Workers 0-6 should find 30 keys, worker 7-9 should find less
            start = i * 20
            expected = min(30, 200 - start)
            assert found_count == expected


def test_stress_test_mixed_operations(tmp_path):
    """Stress test with many threads doing mixed operations."""
    path = make_path(tmp_path, subdir=False)

    with LmdbObjectStore(
        path,
        batch_size=50,
        key_encoding="utf-8",
        map_size=10 * 1024 * 1024,  # 10MB
    ) as db:
        operation_counts = {
            "puts": 0,
            "gets": 0,
            "deletes": 0,
            "get_many": 0,
            "exists": 0,
            "errors": [],
        }
        count_lock = threading.Lock()

        def stress_worker(worker_id, iterations):
            try:
                for i in range(iterations):
                    op = random.choice(["put", "get", "delete", "get_many", "exists"])

                    if op == "put":
                        key = f"key_{worker_id}_{i}"
                        db[key] = {"worker": worker_id, "iter": i, "data": "x" * 100}
                        with count_lock:
                            operation_counts["puts"] += 1

                    elif op == "get":
                        key = f"key_{random.randint(0, 20)}_{random.randint(0, 100)}"
                        db.get(key)
                        with count_lock:
                            operation_counts["gets"] += 1

                    elif op == "delete":
                        key = f"key_{random.randint(0, 20)}_{random.randint(0, 100)}"
                        if db.exists(key):
                            db.delete(key)
                        with count_lock:
                            operation_counts["deletes"] += 1

                    elif op == "get_many":
                        keys = [
                            f"key_{random.randint(0, 20)}_{random.randint(0, 100)}"
                            for _ in range(5)
                        ]
                        found, not_found = db.get_many(keys)
                        with count_lock:
                            operation_counts["get_many"] += 1

                    elif op == "exists":
                        key = f"key_{random.randint(0, 20)}_{random.randint(0, 100)}"
                        db.exists(key)
                        with count_lock:
                            operation_counts["exists"] += 1

                    # Occasionally flush
                    if random.random() < 0.05:
                        db.flush()

            except Exception as e:
                with count_lock:
                    operation_counts["errors"].append((worker_id, op, str(e)))

        # Run stress test
        num_workers = 20
        iterations_per_worker = 100

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(stress_worker, i, iterations_per_worker)
                for i in range(num_workers)
            ]
            for f in futures:
                f.result()

        # Verify no errors
        assert len(operation_counts["errors"]) == 0

        # Verify operations were performed
        total_ops = sum(
            operation_counts[k]
            for k in ["puts", "gets", "deletes", "get_many", "exists"]
        )
        assert total_ops == num_workers * iterations_per_worker


def test_close_waits_for_active_readers(tmp_path):
    """Test that closing the database waits for active readers."""
    path = make_path(tmp_path, subdir=False)
    with LmdbObjectStore(path, subdir=False, key_encoding="utf-8") as db:
        db["a"] = 123
        db.flush()

        barrier = threading.Barrier(2)
        read_started = threading.Event()
        results: list[int] = []

        def reader():
            barrier.wait()
            # Start reading and immediately signal that read has begun
            val = db.get("a")
            read_started.set()  # Notify that get() has been called
            time.sleep(0.05)  # induce a little delay inside read window
            results.append(val)

        t = threading.Thread(target=reader)
        t.start()
        barrier.wait()  # ensure reader started
        read_started.wait()  # wait for reader to actually start reading
        # try close while reader is active. Should block until reader returns.
        db.close()
        t.join()

        assert results == [123]
        # After close, subsequent operations should fail
        with pytest.raises(lmdb.Error):
            db.get("a")
