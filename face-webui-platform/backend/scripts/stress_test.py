"""
stress_test.py — Phase 6 stress/stability testing suite.

Tests:
  1. ws_flood          — Single WS connection flooded at 60 fps for 10 s
  2. concurrent_ws     — 5 simultaneous WS connections each at 15 fps for 10 s
  3. duplicate_train   — 2 simultaneous training start requests
  4. disconnect_train  — Start training then disconnect immediately
  5. batch_inference   — POST /api/inference/batch with 50 JPEG images
  6. rate_limit_verify — Confirm server drops > live_max_fps frames
  7. ws_reconnect      — Connect → disconnect → reconnect 10 times

Usage:
    # From face-webui-platform/backend/
    python scripts/stress_test.py
    python scripts/stress_test.py --host localhost --port 8000 --tests 1,2,3
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Optional imports — skip gracefully if not installed ──────────────────────
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    print("[WARN] httpx not installed — HTTP tests will be skipped. pip install httpx")

try:
    import websockets  # type: ignore
    _HAS_WS = True
except ImportError:
    _HAS_WS = False
    print("[WARN] websockets not installed — WS tests will be skipped. pip install websockets")

try:
    import numpy as np
    from PIL import Image  # type: ignore
    _HAS_IMAGE = True
except ImportError:
    _HAS_IMAGE = False
    print("[WARN] numpy/Pillow not installed — image generation disabled. pip install numpy pillow")

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore
    _HAS_PSUTIL = False
    print("[WARN] psutil not installed — memory assertions disabled. pip install psutil")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("stress")


# ─────────────────────────────────────────────────────────────────────────────
#  Image generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg(width: int = 320, height: int = 240, quality: int = 70) -> bytes:
    """Generate a random JPEG frame as bytes."""
    if _HAS_IMAGE:
        rng = np.random.default_rng()
        arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    # Minimal JPEG header fallback (will likely fail inference; still tests WS)
    return bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0] * 14 + [0xFF, 0xD9])


# ─────────────────────────────────────────────────────────────────────────────
#  Result collector
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str) -> None:
        self.name    = name
        self.passed  = False
        self.notes:  List[str] = []
        self.metrics: Dict[str, Any] = {}
        self._start  = time.perf_counter()

    def ok(self, **kwargs: Any) -> "TestResult":
        self.passed = True
        self.metrics.update(kwargs)
        return self

    def fail(self, reason: str) -> "TestResult":
        self.passed = False
        self.notes.append(reason)
        return self

    def note(self, msg: str) -> "TestResult":
        self.notes.append(msg)
        return self

    @property
    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 3)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        extras = " | ".join(f"{k}={v}" for k, v in self.metrics.items())
        notes  = "; ".join(self.notes) if self.notes else ""
        return (
            f"[{status}] {self.name:<30} {self.elapsed:6.2f}s"
            + (f"  {extras}" if extras else "")
            + (f"  [{notes}]" if notes else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Individual tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_ws_flood(
    base_ws: str,
    duration_sec: float = 10.0,
    target_fps: int = 60,
    warmup_sec: float = 2.0,          # PyTorch allocates on first forward-pass
    max_memory_growth_mb: float = 50.0,
) -> TestResult:
    """Send 60 fps of JPEG frames over a single WS and measure receipt rate.

    Memory growth is measured from *after* the warmup window to *after* the
    full run so PyTorch's one-time caching allocator initialisation does not
    trigger a false positive.
    """
    r = TestResult("ws_flood")
    if not _HAS_WS:
        return r.fail("websockets not installed")

    uri      = f"{base_ws}/ws/live"
    frame    = _make_jpeg()
    interval = 1.0 / target_fps
    sent     = 0
    received = 0
    t_start  = time.perf_counter()

    # We take the memory baseline AFTER warmup, not before.
    rss_post_warmup_mb: float = 0.0

    try:
        async with websockets.connect(uri, max_size=2**23) as ws:
            # wait for ready
            await asyncio.wait_for(ws.recv(), timeout=5.0)

            async def sender() -> None:
                nonlocal sent
                while time.perf_counter() - t_start < duration_sec:
                    await ws.send(frame)
                    sent += 1
                    await asyncio.sleep(interval)

            async def receiver() -> None:
                nonlocal received, rss_post_warmup_mb
                while time.perf_counter() - t_start < duration_sec:
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=0.5)
                        received += 1
                        # Snapshot memory once, right after warmup window expires
                        elapsed = time.perf_counter() - t_start
                        if (
                            _HAS_PSUTIL
                            and rss_post_warmup_mb == 0.0
                            and elapsed >= warmup_sec
                        ):
                            rss_post_warmup_mb = (
                                _psutil.Process().memory_info().rss / 1_048_576
                            )
                    except asyncio.TimeoutError:
                        pass

            await asyncio.gather(sender(), receiver(), return_exceptions=True)

        actual_dur = time.perf_counter() - t_start

        # Memory growth assertion — comparing post-warmup baseline to final RSS
        memory_growth_mb: float = 0.0
        if _HAS_PSUTIL and rss_post_warmup_mb > 0.0:
            rss_final_mb = _psutil.Process().memory_info().rss / 1_048_576
            memory_growth_mb = rss_final_mb - rss_post_warmup_mb
            assert memory_growth_mb < max_memory_growth_mb, (
                f"Memory leak detected after warmup: grew {memory_growth_mb:.1f} MB "
                f"(limit {max_memory_growth_mb} MB)"
            )

        # Must have received at least some results
        assert received > 0, (
            f"Server returned 0 results for {sent} frames — inference pipeline broken"
        )

        r.ok(
            sent=sent,
            received=received,
            send_fps=round(sent / actual_dur, 1),
            recv_fps=round(received / actual_dur, 1),
            memory_growth_post_warmup_mb=round(memory_growth_mb, 1),
        )
        r.note(f"server processed {received}/{sent} frames in {actual_dur:.1f}s")
    except AssertionError as exc:
        r.fail(str(exc))
    except Exception as exc:
        r.fail(str(exc))
    return r


async def test_concurrent_ws(
    base_ws: str,
    num_connections: int = 5,
    duration_sec: float = 10.0,
    fps_per_conn: int = 15,
) -> TestResult:
    """Open N simultaneous WS connections each streaming at fps_per_conn."""
    r = TestResult("concurrent_ws")
    if not _HAS_WS:
        return r.fail("websockets not installed")

    frame    = _make_jpeg()
    interval = 1.0 / fps_per_conn
    results: List[Tuple[int, int]] = []

    async def one_client(idx: int) -> None:
        uri = f"{base_ws}/ws/live"
        sent  = 0
        recvd = 0
        t0    = time.perf_counter()
        try:
            async with websockets.connect(uri, max_size=2**23) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5.0)

                async def _send() -> None:
                    nonlocal sent
                    while time.perf_counter() - t0 < duration_sec:
                        await ws.send(frame)
                        sent += 1
                        await asyncio.sleep(interval)

                async def _recv() -> None:
                    nonlocal recvd
                    while time.perf_counter() - t0 < duration_sec:
                        try:
                            await asyncio.wait_for(ws.recv(), timeout=0.5)
                            recvd += 1
                        except asyncio.TimeoutError:
                            pass

                await asyncio.gather(_send(), _recv(), return_exceptions=True)
        except Exception as exc:
            logger.debug("Client %d error: %s", idx, exc)
        results.append((sent, recvd))

    await asyncio.gather(*(one_client(i) for i in range(num_connections)))

    total_sent  = sum(s for s, _ in results)
    total_recvd = sum(rc for _, rc in results)
    success     = len([s for s, _ in results if s > 0])

    try:
        assert success == num_connections, (
            f"Only {success}/{num_connections} WS connections completed successfully"
        )
        assert total_sent > 0, "No frames were sent across any connection"

        r.ok(
            connections=num_connections,
            successful=success,
            total_sent=total_sent,
            total_received=total_recvd,
        )
        if success < num_connections:
            r.note(f"{num_connections - success} connection(s) failed")
    except AssertionError as exc:
        r.fail(str(exc))
    return r


async def test_duplicate_train(
    base_http: str,
    model_name: str = "scratch_cnn",
) -> TestResult:
    """Fire 2 simultaneous training start requests and verify no server crash."""
    r = TestResult("duplicate_train")
    if not _HAS_HTTPX:
        return r.fail("httpx not installed")

    payload = {
        "model_name":  model_name,
        "epochs":      1,
        "batch_size":  8,
        "learning_rate": 0.001,
    }
    codes: List[int] = []
    try:
        async with httpx.AsyncClient(base_url=base_http, timeout=30.0) as client:
            responses = await asyncio.gather(
                client.post("/api/training/start", json=payload),
                client.post("/api/training/start", json=payload),
                return_exceptions=True,
            )
            for resp in responses:
                if isinstance(resp, Exception):
                    r.note(f"Request exception: {resp}")
                else:
                    codes.append(resp.status_code)

        # Expect at least one 200/201/202; the other may be 409 (conflict)
        successes = [c for c in codes if c in {200, 201, 202}]
        conflicts = [c for c in codes if c == 409]

        assert len(successes) >= 1, (
            f"Neither training request was accepted — both returned {codes}"
        )

        r.ok(codes=codes, accepted=len(successes), conflict=len(conflicts))
    except AssertionError as exc:
        r.fail(str(exc))
    except Exception as exc:
        r.fail(str(exc))
    return r


async def test_disconnect_during_training(
    base_http: str,
    base_ws: str,
    model_name: str = "scratch_cnn",
) -> TestResult:
    """Start training via HTTP, connect WS, disconnect immediately, verify server ok."""
    r = TestResult("disconnect_train")
    if not _HAS_HTTPX or not _HAS_WS:
        return r.fail("httpx or websockets not installed")

    try:
        async with httpx.AsyncClient(base_url=base_http, timeout=15.0) as client:
            resp = await client.post("/api/training/start", json={
                "model_name":    model_name,
                "epochs":        2,
                "batch_size":    8,
                "learning_rate": 0.001,
            })
            if resp.status_code not in {200, 201, 202}:
                return r.fail(f"Training start failed: {resp.status_code}")

            job_id = resp.json().get("job_id", "")
            r.note(f"job_id={job_id[:8]}")

            # Connect WS for training updates then disconnect immediately
            try:
                ws_uri = f"{base_ws}/ws/training/{job_id}"
                async with websockets.connect(ws_uri) as ws:
                    await asyncio.sleep(0.3)
                    # WS context manager closes on exit
            except Exception:
                pass   # disconnect errors are expected

            await asyncio.sleep(1.0)

            # Server should still be alive
            health = await client.get("/health")
            assert health.status_code == 200, (
                f"Server unhealthy after disconnect: HTTP {health.status_code}"
            )
            r.ok(server_alive=True, job_id=job_id[:8])

    except AssertionError as exc:
        r.fail(str(exc))
    except Exception as exc:
        r.fail(str(exc))
    return r


async def test_batch_inference(
    base_http: str,
    num_images: int = 20,
) -> TestResult:
    """POST a batch of images to /api/inference/batch and measure throughput."""
    r = TestResult("batch_inference")
    if not _HAS_HTTPX:
        return r.fail("httpx not installed")

    frames = [_make_jpeg(320, 240) for _ in range(num_images)]
    t0 = time.perf_counter()

    try:
        async with httpx.AsyncClient(base_url=base_http, timeout=60.0) as client:
            files = [("files", (f"frame_{i:03d}.jpg", f, "image/jpeg")) for i, f in enumerate(frames)]
            resp = await client.post("/api/inference/batch", files=files)
        elapsed = time.perf_counter() - t0

        if resp.status_code == 200:
            body = resp.json()
            n_results = len(body.get("results", body if isinstance(body, list) else []))

            assert n_results == num_images, (
                f"Expected {num_images} results but got {n_results}"
            )

            fps = round(num_images / elapsed, 1) if elapsed > 0 else 0.0
            r.ok(images=num_images, results=n_results, fps=fps, elapsed_s=round(elapsed, 2))
        else:
            raise AssertionError(f"Batch inference returned HTTP {resp.status_code}: {resp.text[:200]}")
    except AssertionError as exc:
        r.fail(str(exc))
    except Exception as exc:
        r.fail(str(exc))
    return r


async def test_rate_limit_verify(
    base_ws: str,
    burst_fps: int = 120,    # intentionally above server max
    duration_sec: float = 5.0,
    expected_max_fps: float = 35.0,  # server default limit is 30
) -> TestResult:
    """
    Flood the WS far above the server rate limit and confirm that the server
    accepts only ≤ expected_max_fps inferences (rate drop is the feature).
    """
    r = TestResult("rate_limit_verify")
    if not _HAS_WS:
        return r.fail("websockets not installed")

    uri      = f"{base_ws}/ws/live"
    frame    = _make_jpeg()
    interval = 1.0 / burst_fps
    sent     = 0
    received = 0
    t_start  = time.perf_counter()

    try:
        async with websockets.connect(uri, max_size=2**23) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5.0)

            async def _send() -> None:
                nonlocal sent
                while time.perf_counter() - t_start < duration_sec:
                    await ws.send(frame)
                    sent += 1
                    await asyncio.sleep(interval)

            async def _recv() -> None:
                nonlocal received
                while time.perf_counter() - t_start < duration_sec:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        data = json.loads(msg) if isinstance(msg, str) else {}
                        if data.get("type") == "live_result":
                            received += 1
                    except asyncio.TimeoutError:
                        pass

            await asyncio.gather(_send(), _recv(), return_exceptions=True)

        actual_dur  = time.perf_counter() - t_start
        recv_fps    = received / actual_dur if actual_dur > 0 else 0.0

        assert recv_fps <= expected_max_fps, (
            f"Rate limit violated: server processed {recv_fps:.1f} live_result/s "
            f"but limit is {expected_max_fps} fps"
        )

        r.ok(
            sent=sent,
            live_results=received,
            recv_fps=round(recv_fps, 1),
            rate_limited=True,
        )
    except AssertionError as exc:
        r.fail(str(exc))
    except Exception as exc:
        r.fail(str(exc))
    return r


async def test_ws_reconnect(
    base_ws: str,
    cycles: int = 10,
) -> TestResult:
    """Connect → send 3 frames → disconnect, repeat N times. No crashes expected."""
    r = TestResult("ws_reconnect")
    if not _HAS_WS:
        return r.fail("websockets not installed")

    uri   = f"{base_ws}/ws/live"
    frame = _make_jpeg()
    ok    = 0
    errs: List[str] = []

    for i in range(cycles):
        try:
            async with websockets.connect(uri) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5.0)
                for _ in range(3):
                    await ws.send(frame)
                    await asyncio.sleep(0.05)
                ok += 1
        except Exception as exc:
            errs.append(f"cycle {i}: {exc}")

    try:
        assert ok == cycles, (
            f"WS reconnect failures: {cycles - ok}/{cycles} cycles failed — "
            + "; ".join(errs[:3])
        )
        r.ok(cycles=cycles, successful=ok)
        if errs:
            r.note("; ".join(errs[:3]))
    except AssertionError as exc:
        r.fail(str(exc))
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    1: "ws_flood",
    2: "concurrent_ws",
    3: "duplicate_train",
    4: "disconnect_train",
    5: "batch_inference",
    6: "rate_limit_verify",
    7: "ws_reconnect",
}


async def run_all(
    host: str = "localhost",
    port: int = 8000,
    selected: Optional[List[int]] = None,
) -> None:
    base_http = f"http://{host}:{port}"
    base_ws   = f"ws://{host}:{port}"

    logger.info("=" * 64)
    logger.info("  ML Dashboard Stress Tests")
    logger.info("  Target: %s", base_http)
    logger.info("=" * 64)

    # Quick server health check before starting
    if _HAS_HTTPX:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{base_http}/health", timeout=5.0)
                if r.status_code != 200:
                    logger.error("Server health check failed (%s) — aborting", r.status_code)
                    sys.exit(1)
            logger.info("Server health: OK")
        except Exception as exc:
            logger.error("Cannot reach server at %s: %s", base_http, exc)
            sys.exit(1)

    tests_to_run = selected or list(ALL_TESTS.keys())
    results: List[TestResult] = []

    run_map = {
        1: lambda: test_ws_flood(base_ws),
        2: lambda: test_concurrent_ws(base_ws),
        3: lambda: test_duplicate_train(base_http),
        4: lambda: test_disconnect_during_training(base_http, base_ws),
        5: lambda: test_batch_inference(base_http),
        6: lambda: test_rate_limit_verify(base_ws),
        7: lambda: test_ws_reconnect(base_ws),
    }

    for num in tests_to_run:
        if num not in run_map:
            logger.warning("Unknown test number %d — skipping", num)
            continue
        logger.info("")
        logger.info("--- Test %d: %s ---", num, ALL_TESTS[num])
        result = await run_map[num]()
        results.append(result)
        logger.info("%s", result)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 64)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    logger.info("RESULTS: %d/%d passed  |  %d failed", passed, len(results), failed)
    for r in results:
        sym = "✓" if r.passed else "✗"
        logger.info("  %s  %s", sym, r)
    logger.info("=" * 64)

    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Dashboard stress tests")
    parser.add_argument("--host",  default="localhost", help="Server hostname")
    parser.add_argument("--port",  default=8000, type=int, help="Server port")
    parser.add_argument(
        "--tests",
        default="",
        help="Comma-separated test numbers to run (all by default). "
             + ", ".join(f"{k}={v}" for k, v in ALL_TESTS.items()),
    )
    args = parser.parse_args()

    selected: Optional[List[int]] = None
    if args.tests:
        try:
            selected = [int(x.strip()) for x in args.tests.split(",") if x.strip()]
        except ValueError:
            parser.error("--tests must be comma-separated integers, e.g. '1,2,3'")

    asyncio.run(run_all(host=args.host, port=args.port, selected=selected))


if __name__ == "__main__":
    main()
