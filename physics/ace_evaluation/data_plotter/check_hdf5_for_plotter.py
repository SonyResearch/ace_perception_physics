#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""Check HDF5 files in a folder for data required by the data_plotter.

Reports missing groups, datasets, and attributes at the match/game/rally
level so you can quickly see why certain rallies show up as
gcs_source=unknown or fail to load.

Usage:
    python check_hdf5_for_plotter.py /path/to/folder          # recursive
    python check_hdf5_for_plotter.py /path/to/file.h5         # single file
    python check_hdf5_for_plotter.py /path/to/folder --json   # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ── What the data_plotter loading code expects ────────────────────────────
# (see data_classes.py  _load_rally_static / _load_match_static)

# Required for a rally to load at all (guard condition)
RALLY_REQUIRED_GROUPS = [
    "ground_truth_200",
    "ground_truth_200/ball",
    "ground_truth_200/racket1",
]

# Datasets inside ground_truth_200/ball  (required — no try/except)
BALL_REQUIRED_DATASETS = {
    "ground_truth_200/ball/time":             {"shape_cols": None, "label": "ball time"},
    "ground_truth_200/ball/position":         {"shape_cols": 3,    "label": "ball position (Nx3)"},
    "ground_truth_200/ball/linear_velocity":  {"shape_cols": 3,    "label": "ball linear velocity (Nx3)"},
    "ground_truth_200/ball/angular_velocity": {"shape_cols": 3,    "label": "ball angular velocity (Nx3)"},
}

# Optional ball datasets
BALL_OPTIONAL_DATASETS = {
    "ground_truth_200/ball/confidences": {"min_cols": 13, "label": "ball confidences (needs cols 0-2,10-12)"},
}

# Datasets inside ground_truth_200/racket1  (required — no try/except)
RACKET_REQUIRED_DATASETS = {
    "ground_truth_200/racket1/time":             {"shape_cols": None, "label": "racket time"},
    "ground_truth_200/racket1/position":         {"shape_cols": 3,    "label": "racket position (Nx3)"},
    "ground_truth_200/racket1/linear_velocity":  {"shape_cols": 3,    "label": "racket linear velocity (Nx3)"},
    "ground_truth_200/racket1/orientation":       {"shape_cols": 4,    "label": "racket orientation quaternion (Nx4)"},
}

# Spin sensor groups (checked in priority order)
GCS_SENSOR_KEYS = ["gcs_offline", "gcs_filtered", "gcs"]
GCS_REQUIRED_DATASETS = {
    "sequence_number": {"shape_cols": None, "label": "spin sequence numbers"},
    "spins":           {"shape_cols": 3,    "label": "spin values (Nx3)"},
    "covariance":      {"shape_cols": None, "label": "spin covariance"},
}

# APS triangulation  (required — loaded via get_ball_trajectory)
APS_GROUP = "sensors/aps_ball_triangulation"
APS_REQUIRED_DATASETS = {
    "sensors/aps_ball_triangulation/sequence_number": {"shape_cols": None, "label": "APS sequence numbers"},
    "sensors/aps_ball_triangulation/positions":       {"shape_cols": 3,    "label": "APS ball positions (Nx3)"},
}

# Labels  (required — used to build events/shots)
LABEL_REQUIRED_DATASETS = {
    "labels/timestamps": {"label": "event timestamps"},
    "labels/type":       {"label": "event type strings"},
}
EXPECTED_LABEL_TYPES = {"shot_p1", "shot_p2", "bounce_p1", "bounce_p2", "net", "start", "end"}

# Rally-level HDF5 attributes
RALLY_EXPECTED_ATTRS = ["player_won_point"]

# Match-level (file root) attributes
MATCH_EXPECTED_ATTRS = ["experiment/name", "experiment/date"]

# ground_truth_curr — temporary / optional enrichment
GT_CURR_GROUP = "ground_truth_curr"
GT_CURR_BALL_DATASETS = {
    "ground_truth_curr/ball/time":             {"shape_cols": None, "label": "GT-curr ball time"},
    "ground_truth_curr/ball/position":         {"shape_cols": 3,    "label": "GT-curr ball position (Nx3)"},
    "ground_truth_curr/ball/linear_velocity":  {"shape_cols": 3,    "label": "GT-curr ball linear velocity (Nx3)"},
    "ground_truth_curr/ball/angular_velocity": {"shape_cols": 3,    "label": "GT-curr ball angular velocity (Nx3)"},
}
GT_CURR_OPTIONAL_DATASETS = {
    "ground_truth_curr/ball/confidences": {"min_cols": 13, "label": "GT-curr confidences (needs cols 0-2,10-12)"},
}

# Aerodynamics — optional enrichment
AERO_GROUP = "ground_truth_200/aerodynamics"
# Core aerodynamics datasets (the most commonly used ones)
AERO_CORE_DATASETS = ["t", "pos_optimized", "vel_optimized"]
# Extended aerodynamics variants
AERO_VARIANT_DATASETS = {
    "cd_optimized":        "optimized Cd",
    "cm_optimized":        "optimized Cm",
    "pos_0226":             "0226 position",
    "vel_0226":             "0226 velocity",
    "cd_0226":              "0226 Cd",
    "cm_0226":              "0226 Cm",
    "spin":                 "aerodynamic spin",
}


# ── Severity levels ───────────────────────────────────────────────────────
class Severity:
    ERROR = "ERROR"    # rally won't load at all
    WARNING = "WARN"   # loads but with degraded data
    INFO = "INFO"      # optional data missing


def _check_dataset_shape(
    group: h5py.Group,
    ds_path: str,
    expected_cols: int | None,
) -> str | None:
    """Check a dataset exists and has the expected shape.  Return issue text or None."""
    try:
        ds = group[ds_path]
    except KeyError:
        return None  # missing — caller handles this
    if not isinstance(ds, h5py.Dataset):
        return f"'{ds_path}' is a group, not a dataset"

    shape = ds.shape
    if len(shape) == 0:
        return f"'{ds_path}' is scalar (shape={shape}), expected array"
    if shape[0] == 0:
        return f"'{ds_path}' is empty (shape={shape})"
    if expected_cols is not None and len(shape) >= 2 and shape[1] < expected_cols:
        return f"'{ds_path}' has {shape[1]} columns, expected >= {expected_cols} (shape={shape})"
    if expected_cols is not None and len(shape) == 1:
        return f"'{ds_path}' is 1-D (shape={shape}), expected 2-D with {expected_cols} columns"
    return None


def _check_required_datasets(
    group: h5py.Group,
    spec: dict[str, dict[str, Any]],
    add_fn,
    severity: str = Severity.ERROR,
) -> bool:
    """Check that all datasets in *spec* exist in *group* with correct shapes.

    Returns True if every dataset is present and valid.
    """
    all_ok = True
    for ds_path, meta in spec.items():
        if ds_path not in group:
            add_fn(severity, f"Missing required dataset '{ds_path}' ({meta['label']})")
            all_ok = False
        else:
            shape_issue = _check_dataset_shape(group, ds_path, meta.get("shape_cols"))
            if shape_issue:
                add_fn(severity, f"Shape issue: {shape_issue}")
                all_ok = False
    return all_ok


# ── Per-section rally checkers ────────────────────────────────────────────
# Each takes (rally_group, presence, add_fn) and mutates *presence* in place.


def _check_ball_data(rally_group, presence, add_fn):
    """Check ground_truth_200/ball datasets and optional confidences."""
    if _check_required_datasets(rally_group, BALL_REQUIRED_DATASETS, add_fn):
        presence["gt200_ball"] = True
        try:
            presence["ball_rows"] = rally_group["ground_truth_200/ball/time"].shape[0]
        except Exception:
            pass

    for ds_path, meta in BALL_OPTIONAL_DATASETS.items():
        if ds_path not in rally_group:
            add_fn(Severity.INFO, f"Missing optional '{ds_path}' ({meta['label']})")
        else:
            ds = rally_group[ds_path]
            if isinstance(ds, h5py.Dataset):
                presence["has_confidences"] = True
                if len(ds.shape) >= 2 and ds.shape[1] < meta["min_cols"]:
                    add_fn(Severity.WARNING,
                           f"'{ds_path}' has {ds.shape[1]} columns, need >= {meta['min_cols']} "
                           f"for position + spin confidence indices")


def _check_racket_data(rally_group, presence, add_fn):
    """Check ground_truth_200/racket1 datasets."""
    if _check_required_datasets(rally_group, RACKET_REQUIRED_DATASETS, add_fn):
        presence["gt200_racket"] = True
        try:
            presence["racket_rows"] = rally_group["ground_truth_200/racket1/time"].shape[0]
        except Exception:
            pass


def _check_sensors_data(rally_group, presence, add_fn):
    """Check sensor data: spin (GCS priority cascade) and APS triangulation."""
    if "sensors" not in rally_group:
        add_fn(Severity.ERROR, "Missing 'sensors' group entirely — no spin or APS data")
        return

    sensors = rally_group["sensors"]

    # Spin sensors (gcs_offline > gcs_filtered > gcs)
    gcs_source = None
    gcs_issues_list: list[str] = []
    for gcs_key in GCS_SENSOR_KEYS:
        if gcs_key not in sensors:
            continue
        gcs_group = sensors[gcs_key]
        missing_ds = []
        shape_problems = []
        for ds_name, meta in GCS_REQUIRED_DATASETS.items():
            if ds_name not in gcs_group:
                missing_ds.append(ds_name)
            else:
                shape_issue = _check_dataset_shape(gcs_group, ds_name, meta["shape_cols"])
                if shape_issue:
                    shape_problems.append(shape_issue)

        if missing_ds:
            gcs_issues_list.append(f"sensors/{gcs_key}: missing datasets {missing_ds}")
        elif shape_problems:
            gcs_issues_list.append(
                f"sensors/{gcs_key}: shape issues — {'; '.join(shape_problems)}")
        else:
            gcs_source = gcs_key
            try:
                presence["spin_rows"] = gcs_group["sequence_number"].shape[0]
            except Exception:
                pass
            break

    presence["gcs_source"] = gcs_source if gcs_source else "none"

    if gcs_source is None:
        available = [k for k in GCS_SENSOR_KEYS if k in sensors]
        if not available:
            add_fn(Severity.WARNING,
                   "No spin sensor group (gcs_offline/gcs_filtered/gcs) — spin will be NaN")
        else:
            for gi in gcs_issues_list:
                add_fn(Severity.WARNING, gi)
    elif gcs_source != "gcs_offline":
        add_fn(Severity.WARNING,
               f"Using fallback spin source '{gcs_source}' instead of preferred 'gcs_offline'")

    # APS triangulation
    if APS_GROUP not in rally_group:
        add_fn(Severity.ERROR, f"Missing required group '{APS_GROUP}' — no raw ball positions")
    elif _check_required_datasets(rally_group, APS_REQUIRED_DATASETS, add_fn):
        presence["aps"] = True
        try:
            presence["aps_rows"] = rally_group[
                "sensors/aps_ball_triangulation/sequence_number"].shape[0]
        except Exception:
            pass


def _check_labels_data(rally_group, presence, add_fn):
    """Check label datasets and validate content."""
    if not _check_required_datasets(rally_group, LABEL_REQUIRED_DATASETS, add_fn):
        return

    presence["labels"] = True
    try:
        types_raw = rally_group["labels/type"][()]
        label_types = {t.decode("utf-8") if isinstance(t, bytes) else str(t)
                       for t in types_raw}
        presence["label_types"] = label_types

        if "start" not in label_types or "end" not in label_types:
            add_fn(Severity.WARNING,
                   "Labels missing 'start' and/or 'end' — rally segmentation will fail")
        shot_labels = {"shot_p1", "shot_p2"} & label_types
        if not shot_labels:
            add_fn(Severity.WARNING, "No shot labels (shot_p1/shot_p2) — no shots will be extracted")
        unexpected_labels = label_types - EXPECTED_LABEL_TYPES
        if unexpected_labels:
            add_fn(Severity.INFO, f"Unexpected label types: {sorted(unexpected_labels)}")

        timestamps = rally_group["labels/timestamps"][()]
        if len(timestamps) != len(types_raw):
            add_fn(Severity.ERROR,
                   f"Label mismatch: {len(timestamps)} timestamps vs {len(types_raw)} types")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) >= 0):
            add_fn(Severity.WARNING, "Label timestamps are not monotonically increasing")
    except Exception as e:
        add_fn(Severity.WARNING, f"Could not validate label content: {e}")


def _check_gt_curr_data(rally_group, presence, add_fn):
    """Check ground_truth_curr (optional enrichment)."""
    if GT_CURR_GROUP not in rally_group:
        return
    if "ball" not in rally_group.get(GT_CURR_GROUP, {}):
        return
    presence["gt_curr"] = True
    for ds_path, meta in GT_CURR_BALL_DATASETS.items():
        if ds_path not in rally_group:
            add_fn(Severity.INFO,
                   f"ground_truth_curr/ball present but missing '{ds_path}' ({meta['label']})")
    for ds_path, meta in GT_CURR_OPTIONAL_DATASETS.items():
        if ds_path in rally_group:
            ds = rally_group[ds_path]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) >= 2 and ds.shape[1] < meta["min_cols"]:
                add_fn(Severity.INFO,
                       f"'{ds_path}' has {ds.shape[1]} columns, need >= {meta['min_cols']}")


def _check_aero_data(rally_group, presence, add_fn):
    """Check aerodynamics data (optional enrichment)."""
    if AERO_GROUP not in rally_group:
        add_fn(Severity.INFO, f"No aerodynamics data ('{AERO_GROUP}')")
        return

    aero = rally_group[AERO_GROUP]
    missing_core = [d for d in AERO_CORE_DATASETS if d not in aero]
    if missing_core:
        add_fn(Severity.INFO,
               f"Aerodynamics group present but missing core datasets: {missing_core}")
    else:
        presence["aerodynamics"] = True
        for ds_name in ["pos_optimized", "vel_optimized"]:
            if ds_name in aero:
                ds = aero[ds_name]
                if isinstance(ds, h5py.Dataset) and len(ds.shape) == 1:
                    add_fn(Severity.WARNING,
                           f"Corrupted aerodynamics: '{ds_name}' is 1-D (shape={ds.shape}), "
                           f"expected 2-D — will be skipped")

    found_variants = []
    for ds_name, _label in AERO_VARIANT_DATASETS.items():
        if ds_name in aero:
            found_variants.append(ds_name)
            if ds_name.startswith(("pos_", "vel_")):
                ds = aero[ds_name]
                if isinstance(ds, h5py.Dataset) and len(ds.shape) == 1:
                    add_fn(Severity.WARNING,
                           f"Corrupted aerodynamics: '{ds_name}' is 1-D (shape={ds.shape})")
    presence["aero_variants"] = found_variants


def _check_consistency(presence, add_fn):
    """Cross-check data consistency across sections."""
    if not (presence["gt200_ball"] and presence["gt200_racket"]):
        return
    br = presence["ball_rows"]
    rr = presence["racket_rows"]
    if br > 0 and rr > 0:
        ratio = br / rr
        if ratio > 5 or ratio < 0.2:
            add_fn(Severity.WARNING,
                   f"Large ball/racket row count mismatch: {br} ball vs {rr} racket rows "
                   f"(ratio={ratio:.1f})")


# ── Main rally checker ────────────────────────────────────────────────────


def _check_rally(rally_group: h5py.Group, rally_path: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return (issues, data_presence) for a single rally group."""
    issues: list[dict[str, str]] = []
    presence: dict[str, Any] = {
        "gt200_ball": False,
        "gt200_racket": False,
        "gt_curr": False,
        "gcs_source": "unknown",
        "aps": False,
        "labels": False,
        "aerodynamics": False,
        "aero_variants": [],
        "label_types": set(),
        "ball_rows": 0,
        "racket_rows": 0,
        "aps_rows": 0,
        "spin_rows": 0,
        "has_confidences": False,
    }

    def _add(severity: str, message: str):
        issues.append({"severity": severity, "path": rally_path, "message": message})

    # Guard condition: ground_truth_200 + racket1
    guard_failed = False
    for grp in RALLY_REQUIRED_GROUPS:
        if grp not in rally_group:
            _add(Severity.ERROR, f"Missing required group '{grp}' — rally will NOT load")
            guard_failed = True

    if guard_failed:
        available_groups = []
        if "sensors" in rally_group:
            sensors = rally_group["sensors"]
            found_gcs = [k for k in GCS_SENSOR_KEYS if k in sensors]
            if found_gcs:
                available_groups.append(f"spin: {', '.join(found_gcs)}")
            if "aps_ball_triangulation" in sensors:
                available_groups.append("aps_ball_triangulation")
        if "labels" in rally_group:
            available_groups.append("labels")
        if GT_CURR_GROUP in rally_group:
            available_groups.append("ground_truth_curr")
        if AERO_GROUP in rally_group:
            available_groups.append("aerodynamics")
        if available_groups:
            _add(Severity.INFO,
                 f"Data present but unreachable: {', '.join(available_groups)}")
        return issues, presence

    _check_ball_data(rally_group, presence, _add)
    _check_racket_data(rally_group, presence, _add)
    _check_sensors_data(rally_group, presence, _add)
    _check_labels_data(rally_group, presence, _add)

    for attr in RALLY_EXPECTED_ATTRS:
        if attr not in rally_group.attrs:
            _add(Severity.INFO, f"Missing rally attribute '{attr}'")

    _check_gt_curr_data(rally_group, presence, _add)
    _check_aero_data(rally_group, presence, _add)
    _check_consistency(presence, _add)

    return issues, presence


def check_file(file_path: Path) -> dict[str, Any]:
    """Check a single HDF5 file and return a structured report."""
    report: dict[str, Any] = {
        "file": str(file_path),
        "file_ok": True,
        "match_attrs": {},
        "games": {},
        "issues": [],
        "summary": {
            "total_rallies": 0, "ok_rallies": 0,
            "error_rallies": 0, "warn_rallies": 0,
            "gcs_sources": defaultdict(int),
            "has_aps": 0, "missing_aps": 0,
            "has_labels": 0, "missing_labels": 0,
            "has_gt200_ball": 0, "missing_gt200_ball": 0,
            "has_gt200_racket": 0, "missing_gt200_racket": 0,
            "has_aerodynamics": 0, "missing_aerodynamics": 0,
            "has_gt_curr": 0,
            "has_confidences": 0,
        },
    }

    try:
        with h5py.File(file_path, "r") as f:
            # ── Match-level attributes ───────────────────────────────
            for attr in MATCH_EXPECTED_ATTRS:
                val = f.attrs.get(attr, None)
                if val is None:
                    report["issues"].append({
                        "severity": Severity.WARNING,
                        "path": str(file_path),
                        "message": f"Missing match-level attribute '{attr}'",
                    })
                else:
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    report["match_attrs"][attr] = val

            # Validate experiment/name format (used to extract player name)
            exp_name = report["match_attrs"].get("experiment/name", "")
            if exp_name and "vs_" not in exp_name:
                report["issues"].append({
                    "severity": Severity.WARNING,
                    "path": str(file_path),
                    "message": f"experiment/name='{exp_name}' has no 'vs_' — "
                               f"player name extraction will fall back to folder name",
                })

            # ── Iterate games / rallies ──────────────────────────────
            game_keys = sorted([k for k in f.keys() if k.startswith("game_")],
                               key=lambda k: int(k.split("_")[1]))

            if not game_keys:
                report["issues"].append({
                    "severity": Severity.ERROR,
                    "path": str(file_path),
                    "message": "No game_* groups found in file",
                })
                report["file_ok"] = False
                return report

            for game_key in game_keys:
                game_group = f[game_key]
                rally_keys = sorted(
                    [k for k in game_group.keys() if k.startswith("rally_")],
                    key=lambda k: int(k.split("_")[1]),
                )

                if not rally_keys:
                    report["issues"].append({
                        "severity": Severity.WARNING,
                        "path": f"{file_path}/{game_key}",
                        "message": f"No rally_* groups in {game_key}",
                    })

                game_report: dict[str, Any] = {"rallies": {}}

                for rally_key in rally_keys:
                    rally_path = f"{game_key}/{rally_key}"
                    rally_group = game_group[rally_key]
                    rally_issues, presence = _check_rally(rally_group, rally_path)

                    # Accumulate summary counters
                    s = report["summary"]
                    s["gcs_sources"][presence["gcs_source"]] += 1
                    s["total_rallies"] += 1
                    for key, pkey in [
                        ("has_aps", "aps"), ("has_labels", "labels"),
                        ("has_gt200_ball", "gt200_ball"),
                        ("has_gt200_racket", "gt200_racket"),
                        ("has_aerodynamics", "aerodynamics"),
                    ]:
                        if presence[pkey]:
                            s[key] += 1
                        else:
                            s[key.replace("has_", "missing_")] += 1
                    if presence["gt_curr"]:
                        s["has_gt_curr"] += 1
                    if presence["has_confidences"]:
                        s["has_confidences"] += 1

                    severities = {i["severity"] for i in rally_issues}
                    if Severity.ERROR in severities:
                        status = "ERROR"
                        s["error_rallies"] += 1
                        report["file_ok"] = False
                    elif Severity.WARNING in severities:
                        status = "WARN"
                        s["warn_rallies"] += 1
                    else:
                        status = "OK"
                        s["ok_rallies"] += 1

                    game_report["rallies"][rally_key] = {
                        "status": status,
                        "gcs_source": presence["gcs_source"],
                        "presence": {k: v for k, v in presence.items()
                                     if k not in ("label_types", "aero_variants")},
                        "issues": rally_issues,
                    }
                    report["issues"].extend(rally_issues)

                report["games"][game_key] = game_report

    except Exception as e:
        report["file_ok"] = False
        report["issues"].append({
            "severity": Severity.ERROR,
            "path": str(file_path),
            "message": f"Failed to open file: {type(e).__name__}: {e}",
        })

    return report


def find_h5_files(path: Path) -> list[Path]:
    """Find all .h5 files under a path (or return the path itself)."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.h5"))


# ── Pretty printing ──────────────────────────────────────────────────────

_SEVERITY_SYMBOLS = {
    Severity.ERROR: "❌",
    Severity.WARNING: "⚠️ ",
    Severity.INFO: "ℹ️ ",
}

_GCS_SYMBOLS = {
    "gcs_offline": "✅",
    "gcs_filtered": "⚠️ ",
    "gcs": "⚠️ ",
    "none": "❌",
    "unknown": "❌",
}


def _presence_bar(have: int, total: int, label: str, width: int = 20) -> str:
    """Return a compact bar like: ████████░░░░  16/20  label"""
    if total == 0:
        return f"{'░' * width}   0/0   {label}"
    filled = round(width * have / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * have / total
    sym = "✅" if have == total else ("⚠️ " if have > 0 else "❌")
    return f"{sym} {bar} {have:3d}/{total:<3d} ({pct:5.1f}%)  {label}"


def print_report(report: dict[str, Any], *, verbose: bool = False):
    """Print a human-readable report for one file."""
    s = report["summary"]
    total = s["total_rallies"]
    ok = s["ok_rallies"]
    err = s["error_rallies"]
    warn = s["warn_rallies"]

    file_status = "✅" if report["file_ok"] else "❌"
    print(f"\n{'='*80}")
    print(f"{file_status}  {report['file']}")
    if report["match_attrs"]:
        for k, v in report["match_attrs"].items():
            print(f"    {k}: {v}")
    print(f"    Rallies: {total} total, {ok} OK, {warn} warnings, {err} errors")

    # Data presence overview
    if total > 0:
        print(f"    ── Data presence across {total} rallies ──")
        print(f"      {_presence_bar(s['has_gt200_ball'], total, 'ground_truth_200/ball')}")
        print(f"      {_presence_bar(s['has_gt200_racket'], total, 'ground_truth_200/racket1')}")
        print(f"      {_presence_bar(s['has_aps'], total, 'sensors/aps_ball_triangulation')}")
        print(f"      {_presence_bar(s['has_labels'], total, 'labels')}")
        # GCS source breakdown
        gcs = s["gcs_sources"]
        gcs_ok = gcs.get("gcs_offline", 0) + gcs.get("gcs_filtered", 0) + gcs.get("gcs", 0)
        print(f"      {_presence_bar(gcs_ok, total, 'spin sensor (any)')}")
        if gcs:
            parts = []
            for src in ["gcs_offline", "gcs_filtered", "gcs", "none", "unknown"]:
                if src in gcs:
                    parts.append(f"{_GCS_SYMBOLS.get(src, '?')} {src}={gcs[src]}")
            print(f"        breakdown: {', '.join(parts)}")
        print(f"      {_presence_bar(s.get('has_confidences', 0), total, 'ball confidences')}")
        print(f"      {_presence_bar(s.get('has_gt_curr', 0), total, 'ground_truth_curr')}")
        print(f"      {_presence_bar(s['has_aerodynamics'], total, 'aerodynamics')}")

    # Per-game/rally details
    for game_key, game_data in report["games"].items():
        has_issues = any(
            r["status"] != "OK" for r in game_data["rallies"].values()
        )
        if not has_issues and not verbose:
            rally_count = len(game_data["rallies"])
            print(f"  {game_key}: all {rally_count} rallies OK")
            continue

        print(f"  {game_key}:")
        for rally_key, rally_data in game_data["rallies"].items():
            status = rally_data["status"]
            gcs_src = rally_data["gcs_source"]
            issues = rally_data["issues"]

            if status == "OK" and not verbose:
                continue

            status_sym = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}[status]
            gcs_sym = _GCS_SYMBOLS.get(gcs_src, "?")
            print(f"    {status_sym} {rally_key}  (gcs: {gcs_sym} {gcs_src})")

            for issue in issues:
                sym = _SEVERITY_SYMBOLS.get(issue["severity"], "?")
                print(f"        {sym} [{issue['severity']}] {issue['message']}")


def print_summary(all_reports: list[dict[str, Any]]):
    """Print an overall summary across all files."""
    total_files = len(all_reports)
    ok_files = sum(1 for r in all_reports if r["file_ok"])
    total_rallies = sum(r["summary"]["total_rallies"] for r in all_reports)
    ok_rallies = sum(r["summary"]["ok_rallies"] for r in all_reports)
    err_rallies = sum(r["summary"]["error_rallies"] for r in all_reports)
    warn_rallies = sum(r["summary"]["warn_rallies"] for r in all_reports)

    # Aggregate all counters
    def _sum_key(key):
        return sum(r["summary"].get(key, 0) for r in all_reports)

    gcs_totals: dict[str, int] = defaultdict(int)
    for r in all_reports:
        for src, cnt in r["summary"]["gcs_sources"].items():
            gcs_totals[src] += cnt

    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"  Files:   {total_files} checked, {ok_files} fully OK, {total_files - ok_files} with errors")
    print(f"  Rallies: {total_rallies} total, {ok_rallies} OK, {warn_rallies} warnings, {err_rallies} errors")

    if total_rallies > 0:
        print(f"\n  ── Data presence across all {total_rallies} rallies ──")
        print(f"    {_presence_bar(_sum_key('has_gt200_ball'), total_rallies, 'ground_truth_200/ball (required)')}")
        print(f"    {_presence_bar(_sum_key('has_gt200_racket'), total_rallies, 'ground_truth_200/racket1 (required)')}")
        print(f"    {_presence_bar(_sum_key('has_aps'), total_rallies, 'sensors/aps_ball_triangulation (required)')}")
        print(f"    {_presence_bar(_sum_key('has_labels'), total_rallies, 'labels (required)')}")

        gcs_ok = gcs_totals.get("gcs_offline", 0) + gcs_totals.get("gcs_filtered", 0) + gcs_totals.get("gcs", 0)
        print(f"    {_presence_bar(gcs_ok, total_rallies, 'spin sensor (any)')}")
        if gcs_totals:
            print("      GCS source breakdown:")
            for src in ["gcs_offline", "gcs_filtered", "gcs", "none", "unknown"]:
                if src in gcs_totals:
                    pct = 100 * gcs_totals[src] / total_rallies
                    sym = _GCS_SYMBOLS.get(src, "?")
                    print(f"        {sym} {src:15s}: {gcs_totals[src]:5d}  ({pct:5.1f}%)")

        print(f"    {_presence_bar(_sum_key('has_confidences'), total_rallies, 'ball confidences (optional)')}")
        print(f"    {_presence_bar(_sum_key('has_gt_curr'), total_rallies, 'ground_truth_curr (optional)')}")
        print(f"    {_presence_bar(_sum_key('has_aerodynamics'), total_rallies, 'aerodynamics (optional)')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check HDF5 files for data required by the data_plotter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", type=Path,
                        help="Path to an HDF5 file or a folder to scan recursively")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show details for all rallies including OK ones")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable text")
    parser.add_argument("--errors-only", "-e", action="store_true",
                        help="Only show rallies with errors (hide warnings/info)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: path '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)

    h5_files = find_h5_files(args.path)
    if not h5_files:
        print(f"No .h5 files found under '{args.path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(h5_files)} HDF5 file(s)...")

    all_reports = []
    for fpath in h5_files:
        report = check_file(fpath)
        all_reports.append(report)

    if args.json:
        # Convert defaultdicts to regular dicts for JSON serialization
        for r in all_reports:
            r["summary"]["gcs_sources"] = dict(r["summary"]["gcs_sources"])
        json.dump(all_reports, sys.stdout, indent=2, default=str)
        print()
    else:
        skipped_ok = 0
        for report in all_reports:
            # Skip fully-OK files unless --verbose is set
            has_errors_or_warnings = (
                report["summary"]["error_rallies"] > 0
                or report["summary"]["warn_rallies"] > 0
                or not report["file_ok"]
            )
            if not has_errors_or_warnings and not args.verbose:
                skipped_ok += 1
                continue

            if args.errors_only:
                # Filter to only ERROR-level issues
                for game_data in report["games"].values():
                    for rally_data in game_data["rallies"].values():
                        rally_data["issues"] = [
                            i for i in rally_data["issues"]
                            if i["severity"] == Severity.ERROR
                        ]
            print_report(report, verbose=args.verbose)

        if skipped_ok > 0:
            print(f"\n({skipped_ok} file(s) with all data OK — not shown, use --verbose to see)")

        print_summary(all_reports)


if __name__ == "__main__":
    main()
