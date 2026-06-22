#!/usr/bin/env python3
"""Remove all non-ball-related topics from ROS2 bag files.

Searches recursively for 'rosbag' folders containing .db3 files under the
given root path, filters out any topics not related to the ball (e.g. racket,
player, image), and writes the filtered bag back in place.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import rosbag2_py
except ImportError:
    sys.exit(
        "rosbag2_py not found. Make sure ROS2 is sourced:\n"
        "  source /opt/ros/<distro>/setup.bash"
    )

DEFAULT_ROOT = Path(
    "/path/to/recordings"
)

ROSBAG_DIR_NAME = "rosbag"
BALL_KEYWORDS = ["ball", "jpeg"]
EXCLUDE_KEYWORDS = ["player", "racket", "human", "image", "gcs"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_ball_topic(topic: str) -> bool:
    lower = topic.lower()
    if any(kw in lower for kw in EXCLUDE_KEYWORDS):
        return False
    return any(kw in lower for kw in BALL_KEYWORDS)


def find_db3_files(root: Path) -> list[Path]:
    """Return all .db3 files that live inside a directory named 'rosbag'."""
    return sorted(p for p in root.rglob("*.db3") if p.parent.name == ROSBAG_DIR_NAME)


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------


def strip_non_ball_topics(
    db3_path: Path, dry_run: bool = False
) -> tuple[bool, list[str]]:
    """
    Read a .db3 bag file, remove non-ball topics, write back in place.

    Returns (changed, removed_topics).
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(db3_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    all_topics: list[rosbag2_py.TopicMetadata] = reader.get_all_topics_and_types()
    ball_topics = [t for t in all_topics if is_ball_topic(t.name)]
    removed_names = [t.name for t in all_topics if not is_ball_topic(t.name)]

    if not removed_names:
        return False, []

    if dry_run:
        return True, removed_names

    ball_names = {t.name for t in ball_topics}

    # Write filtered messages to a sibling .db3 temp file, then swap.
    tmp_path = db3_path.with_name(db3_path.stem + "__tmp_filtered.db3")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(tmp_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    for tmeta in ball_topics:
        writer.create_topic(tmeta)

    reader.set_filter(rosbag2_py.StorageFilter(topics=list(ball_names)))
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        writer.write(topic, data, timestamp)

    del writer  # flush / close
    del reader

    # Swap: remove original file, rename temp into place
    db3_path.unlink()
    tmp_path.rename(db3_path)

    return True, removed_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root path to search recursively for .db3 files (default: {DEFAULT_ROOT})",
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

    db3_files = find_db3_files(root)
    if not db3_files:
        print(f"No .db3 files found inside '{ROSBAG_DIR_NAME}' folders under {root}")
        return 0

    print(f"Found {len(db3_files)} .db3 file(s) under {root}")

    total_files = 0
    total_changed = 0
    total_skipped = 0
    total_errors = 0

    for db3_path in db3_files:
        total_files += 1
        print(f"\n[{db3_path}]")
        try:
            changed, removed = strip_non_ball_topics(db3_path, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001
            total_errors += 1
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        if changed:
            total_changed += 1
            action = "would remove" if args.dry_run else "removed"
            print(f"  {action} {len(removed)} topic(s):")
            for t in removed:
                print(f"    {t}")
        else:
            total_skipped += 1
            print("  no non-ball topics found, skipped")

    print(
        f"\nDone. files={total_files} changed={total_changed} "
        f"skipped={total_skipped} errors={total_errors} dry_run={args.dry_run}"
    )
    return 0 if total_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
