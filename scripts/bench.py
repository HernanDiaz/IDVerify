"""Concurrency benchmark for a running DocVerify API.

Submits N images at a target concurrency, waits for each analysis to finish,
and reports end-to-end latency percentiles, throughput and the server-side
inference time.

Usage:
    python scripts/bench.py
    BASE_URL=http://localhost:8000 N=60 CONCURRENCY=8 python scripts/bench.py
"""

from __future__ import annotations

import asyncio
import io
import os
import statistics
import time

import httpx
from PIL import Image

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
N = int(os.environ.get("N", "60"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
POLL_INTERVAL = 0.1
TIMEOUT = 60.0


def _image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 320), (180, 170, 160)).save(buf, format="JPEG")
    return buf.getvalue()


async def _run_one(
    client: httpx.AsyncClient,
    image: bytes,
    sem: asyncio.Semaphore,
    out: list[tuple[float, float | None, str]],
) -> None:
    async with sem:
        start = time.perf_counter()
        resp = await client.post(
            "/analyses", files={"file": ("doc.jpg", image, "image/jpeg")}
        )
        resp.raise_for_status()
        analysis_id = resp.json()["id"]

        status, inference_ms = "pending", None
        deadline = time.perf_counter() + TIMEOUT
        while time.perf_counter() < deadline:
            body = (await client.get(f"/analyses/{analysis_id}")).json()
            status = body["status"]
            if status in ("done", "failed"):
                inference_ms = body.get("inference_ms")
                break
            await asyncio.sleep(POLL_INTERVAL)
        out.append(((time.perf_counter() - start) * 1000.0, inference_ms, status))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


async def main() -> None:
    image = _image_bytes()
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[tuple[float, float | None, str]] = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        start = time.perf_counter()
        await asyncio.gather(*(_run_one(client, image, sem, results) for _ in range(N)))
        wall = time.perf_counter() - start

    totals = [t for t, _, _ in results]
    infers = [i for _, i, _ in results if i is not None]
    done = sum(1 for _, _, s in results if s == "done")

    print(f"requests={N} concurrency={CONCURRENCY} done={done}")
    print(f"wall={wall:.2f}s throughput={N / wall:.1f} req/s")
    print(
        f"end-to-end ms   p50={_percentile(totals, 0.5):.0f} "
        f"p95={_percentile(totals, 0.95):.0f} max={max(totals):.0f}"
    )
    if infers:
        print(
            f"inference ms    mean={statistics.mean(infers):.0f} "
            f"p95={_percentile(infers, 0.95):.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
