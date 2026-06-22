#!/usr/bin/env python3
"""Remove all racket-related keys from .pt label files.

Searches for a "label" folder under the given root path, loads every .pt file
inside it, drops any keys starting with "racket_", and writes the file back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_ROOT = Path(
    "/path/to/recordings"
)
LABEL_DIR_NAME = "labels"
RACKET_PREFIX = ("player", "racket", "human")


def find_label_dirs(root: Path) -> list[Path]:
    return [p for p in root.rglob(LABEL_DIR_NAME) if p.is_dir()]


def strip_racket_keys(pt_path: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Load a .pt file, remove racket_* keys, save in place. Returns (changed, removed_keys)."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        return False, []

    removed = [
        k
        for k in list(data.keys())
        if isinstance(k, str) and k.startswith(RACKET_PREFIX)
    ]
    if not removed:
        return False, []

    for k in removed:
        del data[k]

    if not dry_run:
        torch.save(data, pt_path)
    return True, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root path to search (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes without writing files.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(
            f"ERROR: root path does not exist or is not a directory: {root}",
            file=sys.stderr,
        )
        return 1

    label_dirs = find_label_dirs(root)
    if not label_dirs:
        print(f"No '{LABEL_DIR_NAME}' folders found under {root}")
        return 0

    print(f"Found {len(label_dirs)} '{LABEL_DIR_NAME}' folder(s) under {root}")

    total_files = 0
    total_changed = 0
    total_skipped = 0
    total_errors = 0

    for label_dir in label_dirs:
        pt_files = sorted(label_dir.glob("*.pt"))
        if not pt_files:
            continue
        print(f"\n[{label_dir}] processing {len(pt_files)} .pt file(s)")
        for pt_path in pt_files:
            total_files += 1
            try:
                changed, removed = strip_racket_keys(pt_path, dry_run=args.dry_run)
            except Exception as e:  # noqa: BLE001
                total_errors += 1
                print(f"  ERROR {pt_path.name}: {e}", file=sys.stderr)
                continue
            if changed:
                total_changed += 1
                action = "would remove" if args.dry_run else "removed"
                print(f"  {pt_path.name}: {action} {removed}")
            else:
                total_skipped += 1

    print(
        f"\nDone. files={total_files} changed={total_changed} "
        f"skipped={total_skipped} errors={total_errors} dry_run={args.dry_run}"
    )
    return 0 if total_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
