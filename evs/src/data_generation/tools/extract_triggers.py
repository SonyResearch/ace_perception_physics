# pylint: disable=broad-except, subprocess-run-check
# TODO(asude): clean pylint

"""
@brief Extract APS, EVS time synchronization triggers.

@file extract_triggers.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(
    "/media/chaydina/T71/asude_overhead_evs_recordings/Univ_2024-05-29_recording_upload"
)
EVS_DIR_NAME = "evs"
TRIGGER_FILE_NAME = "triggers.txt"
# Path to the trigger-extraction tool inside the source tree (NOT the install/ share dir).
DEFAULT_TRIGGER_TOOL = Path(
    "/home/EU/chaydina/ws/src/project_ace_evs_ball/src/evs/tools/evs_trigger_to_txt"
)


def find_evs_dirs(root: Path) -> list[Path]:
    """Recursively find all directories named EVS_DIR_NAME under root."""
    return [p for p in root.rglob(EVS_DIR_NAME) if p.is_dir()]


def extract_triggers_for_dir(evs_dir: Path, trigger_execution_path: Path) -> str:
    """Extract triggers from all .raw files in a single evs directory.

    Returns one of: "ok", "skipped", "no_raw", "error".
    """
    target_trigger_file = evs_dir / TRIGGER_FILE_NAME

    # if target_trigger_file.exists():
    #     print(f"  [{evs_dir}] {TRIGGER_FILE_NAME} already exists, skipping.")
    #     return "skipped"

    raw_files = sorted(evs_dir.glob("*.raw"))
    if not raw_files:
        print(f"  [{evs_dir}] no .raw files found, skipping.")
        return "no_raw"

    print(f"  [{evs_dir}] processing {len(raw_files)} .raw file(s)")
    any_error = False
    for raw_file in raw_files:
        try:
            subprocess.run(
                [
                    "python3",
                    str(trigger_execution_path),
                    str(raw_file),
                    str(target_trigger_file),
                ]
            )
        except Exception as error:
            any_error = True
            print(f"    ERROR {raw_file.name}: {error}", file=sys.stderr)
            continue

        # Only one of the ~4 cameras carries the external triggers. Once a
        # camera produces a non-empty trigger file, stop so a subsequent
        # (trigger-less) camera doesn't overwrite it with an empty file.
        if target_trigger_file.exists() and target_trigger_file.stat().st_size > 0:
            print(f"    triggers found in {raw_file.name}, stopping.")
            break

    if not target_trigger_file.exists():
        print(
            f"    ERROR: expected {target_trigger_file} was not produced.",
            file=sys.stderr,
        )
        return "error"

    return "error" if any_error else "ok"

def extract_triggers(args):
    """Recursively extract triggers for every evs/ folder under args.root_dir."""
    print("-----------------STEP 1-----------------------")
    print(f": Create trigger time stamps under {args.root_dir}")

    root = Path(args.root_dir)
    if not root.is_dir():
        print(
            f"ERROR: root path does not exist or is not a directory: {root}",
            file=sys.stderr,
        )
        return 1

    trigger_execution_path = Path(args.trigger_tool)
    if not trigger_execution_path.is_file():
        print(
            f"ERROR: trigger tool not found at: {trigger_execution_path}",
            file=sys.stderr,
        )
        return 1

    evs_dirs = find_evs_dirs(root)
    if not evs_dirs:
        print(f"No '{EVS_DIR_NAME}' folders found under {root}")
        return 0

    print(f"Found {len(evs_dirs)} '{EVS_DIR_NAME}' folder(s) under {root}")

    counts = {"ok": 0, "skipped": 0, "no_raw": 0, "error": 0}
    for evs_dir in evs_dirs:
        result = extract_triggers_for_dir(evs_dir, trigger_execution_path)
        counts[result] += 1

    print(
        f"\nDone. ok={counts['ok']} skipped={counts['skipped']} "
        f"no_raw={counts['no_raw']} errors={counts['error']}"
    )
    print("-----------------Done with STEP 1-----------------------\n")
    return 0 if counts["error"] == 0 else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root path to search recursively for '{EVS_DIR_NAME}' folders "
        f"(default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--trigger_tool",
        type=Path,
        default=DEFAULT_TRIGGER_TOOL,
        help=f"Path to the evs_trigger_to_txt script (default: {DEFAULT_TRIGGER_TOOL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(extract_triggers(_parse_args()))
