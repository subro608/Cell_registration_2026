#!/usr/bin/env python3
"""Download the three dataset files (~GBs total) from Dropbox into data/.

Downloads:
  - data/slice3_to_invivoLANDMARKS.json     (115 paired landmark coordinates)
  - data/zstack.tif                         (in-vivo GCaMP volume)
  - data/Sparrow_3_po_488_4x-registered.tif (ex-vivo registered volume)

Idempotent: existing files are skipped (use --force to re-download).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

# Dropbox URLs. Append `&dl=1` to force a direct download (vs the file
# previewer landing page).
FILES = [
    {
        "name": "slice3_to_invivoLANDMARKS.json",
        "url":  "https://www.dropbox.com/scl/fi/ou6q1b2czrruynnm5kc6v/slice3_to_invivoLANDMARKS.json?rlkey=qr1cymvkm5tkcnk35uq9165rn&dl=1",
    },
    {
        "name": "zstack.tif",
        "url":  "https://www.dropbox.com/scl/fi/opfa4wetmw4w0tngr15qv/zstack.tif?rlkey=ddbopk6tumvi5phz9744xffhm&st=s1pn5ff4&dl=1",
    },
    {
        "name": "Sparrow_3_po_488_4x-registered.tif",
        "url":  "https://www.dropbox.com/scl/fi/sgviwbb64qypjrjw0xtyg/Sparrow_3_po_488_4x-registered.tif?rlkey=wds9m9a1lx7rs9ebax3ukc6df&dl=1",
    },
]


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, dest: Path) -> None:
    """Download with a progress indicator using urllib."""
    last_pct = [-1]

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = int(downloaded * 100 / total_size)
            if pct != last_pct[0] and (pct % 5 == 0 or pct == 100):
                sys.stdout.write(
                    f"\r  {pct:3d}%  ({fmt_bytes(min(downloaded, total_size))} / {fmt_bytes(total_size)})"
                )
                sys.stdout.flush()
                last_pct[0] = pct
        else:
            sys.stdout.write(f"\r  {fmt_bytes(downloaded)} downloaded")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the dataset from Dropbox.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for entry in FILES:
        dest = DATA_DIR / entry["name"]
        if dest.exists() and not args.force:
            print(f"[skip] {dest.relative_to(PROJECT_ROOT)} already exists "
                  f"({fmt_bytes(dest.stat().st_size)})")
            continue
        print(f"[get ] {dest.relative_to(PROJECT_ROOT)}")
        try:
            download(entry["url"], dest)
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            print(f"  FAILED: {exc}")
            print(f"  Manual download: {entry['url']}")
            sys.exit(1)
        print(f"  -> {fmt_bytes(dest.stat().st_size)}")

    print("\nAll files present. Set CELLINVARIANCE_DATA_DIR if you need a different location.")


if __name__ == "__main__":
    main()
