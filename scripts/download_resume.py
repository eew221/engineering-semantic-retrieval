from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retries", type=int, default=50)
    parser.add_argument("--chunk-mb", type=int, default=1)
    parser.add_argument("--progress-step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mb * 1024 * 1024

    for attempt in range(args.retries):
        existing = out.stat().st_size if out.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        print(f"Attempt {attempt + 1}, resume from {existing / 1024 / 1024:.1f} MB")
        sys.stdout.flush()
        resp = requests.get(args.url, headers=headers, stream=True, timeout=60)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code}")

        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            grand_total = int(content_range.split("/")[-1])
        else:
            grand_total = int(resp.headers.get("content-length", 0)) + existing

        mode = "ab" if existing else "wb"
        try:
            with open(out, mode) as f:
                written = existing
                last_pct = int(written * 100 / grand_total) if grand_total else -1
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if grand_total:
                        pct = int(written * 100 / grand_total)
                        if pct != last_pct and pct % args.progress_step == 0:
                            print(f"{pct}% {written / 1024 / 1024:.1f}MB/{grand_total / 1024 / 1024:.1f}MB")
                            sys.stdout.flush()
                            last_pct = pct
            if grand_total and out.stat().st_size >= grand_total:
                print("DONE", out, out.stat().st_size)
                return
        except Exception as e:
            print("RETRY", type(e).__name__, e)
            sys.stdout.flush()
            time.sleep(5)

    raise RuntimeError("Failed to complete download after retries")


if __name__ == "__main__":
    main()
