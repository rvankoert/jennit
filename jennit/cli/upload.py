"""jennit-upload — batch upload client for the jennit API.

Walks a directory (or takes a single file), uploads every supported
image with concurrent HTTP requests (folder structure preserved via
the ?path= parameter), then signals batch completion.
"""
import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import requests

from ..analysis import IMAGE_EXTENSIONS


def collect_files(input_dir: str) -> list[tuple[Path, str]]:
    """Return [(absolute_path, relative_posix_path)] for supported images."""
    root = Path(input_dir)
    if root.is_file():
        return [(root, root.name)]
    if not root.is_dir():
        raise SystemExit(f"not a file or directory: {input_dir}")
    out: list[tuple[Path, str]] = []
    exts = tuple(e.lower() for e in IMAGE_EXTENSIONS)
    for dirpath, _, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        for fn in filenames:
            if fn.lower().endswith(exts):
                p = Path(dirpath) / fn
                rel = f"{rel_dir}/{fn}" if rel_dir else fn
                out.append((p, rel))
    return out


def upload_one(base: str, headers: dict, batch_id: str,
               path: Path, rel: str, max_bytes: int) -> dict:
    if path.stat().st_size > max_bytes:
        return {"rel": rel, "ok": False,
                "error": "skipped: larger than --max-bytes"}
    folder = str(Path(rel).parent)
    if folder == ".":
        folder = ""
    with open(path, "rb") as fh:
        r = requests.post(
            f"{base}/api/v1/batches/{batch_id}/files",
            params={"path": folder} if folder else None,
            headers=headers,
            files={"file": (path.name, fh)},
            timeout=600,
        )
    if r.status_code >= 400:
        return {"rel": rel, "ok": False,
                "error": f"HTTP {r.status_code} {r.text[:200]}"}
    return {"rel": rel, "ok": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a batch of scanned images to a jennit server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="base URL of the jennit API "
                             "(default: http://127.0.0.1:8000)")
    parser.add_argument("--token",
                        default=os.environ.get("JENNIT_API_TOKEN", ""),
                        help="bearer token (or set JENNIT_API_TOKEN)")
    parser.add_argument("input",
                        help="image file, or directory of images "
                             "(walked recursively, folder structure kept)")
    parser.add_argument("--name", default=None,
                        help="batch name (default: <input name>-<timestamp>)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent uploads (default: 8)")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 ** 3,
                        help="skip files larger than this "
                             "(default: 2 GiB)")
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        print("no supported image files found")
        sys.exit(1)
    print(f"{len(files)} file(s) to upload to {args.url}")

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    base = args.url.rstrip("/")

    name = args.name or (
        f"{Path(args.input).name}-{time.strftime('%Y%m%d-%H%M%S')}")
    r = requests.post(f"{base}/api/v1/batches",
                      data={"name": name}, headers=headers, timeout=30)
    r.raise_for_status()
    batch_id = r.json()["id"]
    print(f"batch created: {batch_id}")

    errors = 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as ex:
        futs = [ex.submit(upload_one, base, headers, batch_id, p, rel,
                          args.max_bytes) for p, rel in files]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res["ok"]:
                print(f"[{i}/{len(files)}] {res['rel']}")
            else:
                errors += 1
                print(f"[{i}/{len(files)}] FAIL {res['rel']}: "
                      f"{res['error']}")

    r = requests.post(f"{base}/api/v1/batches/{batch_id}/complete",
                      headers=headers, timeout=30)
    r.raise_for_status()
    status = r.json()["status"]
    print(f"complete signaled: status={status}, errors={errors}")
    print(f"progress: GET {base}/api/v1/batches/{batch_id} "
          f"(or open the web UI)")
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
