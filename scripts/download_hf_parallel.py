#!/usr/bin/env python3
"""Download one Hugging Face file with concurrent HTTP Range requests.

This uses only Python's standard library plus curl.  It is intended for large
public checkpoint shards when a mirror serves byte ranges faster than the
regular Hub client's single-file downloader.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def remote_size(url: str) -> int:
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--head", url],
        check=True,
        capture_output=True,
        text=True,
    )
    sizes = [
        line.split(":", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.lower().startswith("content-length:")
    ]
    if not sizes:
        raise RuntimeError("The server did not provide Content-Length")
    return int(sizes[-1])


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download_part(url: str, destination: Path, start: int, end: int) -> int:
    expected_size = end - start + 1
    if destination.exists() and destination.stat().st_size == expected_size:
        return expected_size

    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "6",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--range",
            f"{start}-{end}",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    actual_size = temporary.stat().st_size
    if actual_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Range {start}-{end} has {actual_size} bytes; expected {expected_size}"
        )
    os.replace(temporary, destination)
    return actual_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Repository in namespace/name form")
    parser.add_argument("path", help="Repository-relative file path")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sha256", help="Expected SHA-256 checksum")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    endpoint = args.endpoint.rstrip("/")
    url = f"{endpoint}/{args.repo_id}/resolve/{args.revision}/{args.path}"
    size = remote_size(url)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.destination.with_name(args.destination.name + ".parts")
    parts_dir.mkdir(exist_ok=True)

    workers = min(args.workers, size)
    chunk_size = (size + workers - 1) // workers
    ranges = [
        (index, start, min(size - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, size, chunk_size))
    ]
    started = time.monotonic()
    print(f"Downloading {args.path}: {size:,} bytes in {len(ranges)} ranges", flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {
            executor.submit(
                download_part,
                url,
                parts_dir / f"{index:04d}.part",
                start,
                end,
            ): index
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"Completed {completed}/{len(ranges)} ranges ({completed * chunk_size / elapsed / 1_000_000:.1f} MB/s)", flush=True)

    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    with temporary.open("wb") as combined:
        for index, _, _ in ranges:
            with (parts_dir / f"{index:04d}.part").open("rb") as part:
                shutil.copyfileobj(part, combined, 16 * 1024 * 1024)
    if temporary.stat().st_size != size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Combined file has an unexpected size")
    if args.sha256:
        actual_digest = digest(temporary)
        if actual_digest.lower() != args.sha256.lower():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch: got {actual_digest}, expected {args.sha256}")
    os.replace(temporary, args.destination)
    shutil.rmtree(parts_dir)
    print(f"Saved and verified {args.destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
