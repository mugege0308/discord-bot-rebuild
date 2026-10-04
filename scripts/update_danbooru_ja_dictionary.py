"""Safely update the local PYU224 Japanese Danbooru dictionary.

This is a manual command.  Runtime search never downloads from GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.danbooru_ja_dictionary import (
    DEFAULT_DICTIONARY_PATH,
    DEFAULT_META_PATH,
    UPSTREAM_CSV_URL,
    UPSTREAM_META_URL,
    parse_dictionary_file,
)


MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
MIN_ROW_COUNT = 10_000


def _download(url: str, destination: Path, max_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "local-gallery-dictionary-updater"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("upstream file exceeds size limit")
            output.write(chunk)


def _validate_meta(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("generated_at"):
        raise ValueError("upstream metadata is invalid")
    return value


def update_dictionary(
    *,
    csv_url: str = UPSTREAM_CSV_URL,
    meta_url: str = UPSTREAM_META_URL,
    dictionary_path: Path = DEFAULT_DICTIONARY_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> dict[str, object]:
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="danbooru-ja-", dir=dictionary_path.parent) as temporary:
        temporary_dir = Path(temporary)
        csv_temp = temporary_dir / "danbooru-ja.csv"
        meta_temp = temporary_dir / "meta.json"
        _download(csv_url, csv_temp, MAX_CSV_BYTES)
        entries = parse_dictionary_file(csv_temp)
        if len(entries) < MIN_ROW_COUNT:
            raise ValueError(f"upstream dictionary has too few rows: {len(entries)}")
        _download(meta_url, meta_temp, MAX_META_BYTES)
        metadata = _validate_meta(meta_temp)
        metadata.update(
            {
                "upstream_repository": "https://github.com/PYU224/tagdb-updater",
                "upstream_csv": csv_url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "row_count_validated": len(entries),
            }
        )
        meta_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(csv_temp, dictionary_path)
        os.replace(meta_temp, meta_path)
        return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-url", default=UPSTREAM_CSV_URL)
    parser.add_argument("--meta-url", default=UPSTREAM_META_URL)
    args = parser.parse_args()
    metadata = update_dictionary(csv_url=args.csv_url, meta_url=args.meta_url)
    print(json.dumps({"row_count": metadata["row_count_validated"], "generated_at": metadata.get("generated_at")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
