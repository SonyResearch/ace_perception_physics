#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.

import argparse
import pathlib
import sys
import h5py
import numpy as np


def compare_elements(x1, x2, atol, rtol):
    """
    Compare two arbitrary elements (which can be NumPy arrays, floats, strings, etc.)
    with tolerance checks for numeric types.
    """
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        if x1.shape != x2.shape:
            return False
        # If they are numeric arrays
        if np.issubdtype(x1.dtype, np.number) and np.issubdtype(x2.dtype, np.number):
            nan1 = np.isnan(x1)
            nan2 = np.isnan(x2)
            if not np.array_equal(nan1, nan2):
                return False
            inf1 = np.isinf(x1)
            inf2 = np.isinf(x2)
            if not np.array_equal(inf1, inf2):
                return False
            valid_mask = ~nan1 & ~inf1
            if not np.any(valid_mask):
                return True
            v1 = x1[valid_mask]
            v2 = x2[valid_mask]
            abs_diff = np.abs(v1 - v2)
            tol_limit = atol + rtol * np.abs(v2)
            if np.any(abs_diff > tol_limit):
                return False
            return True
        else:
            return np.array_equal(x1, x2)
    else:
        try:
            is_arr1 = isinstance(x1, np.ndarray)
            is_arr2 = isinstance(x2, np.ndarray)
            if is_arr1 != is_arr2:
                return False
            if is_arr1:
                return compare_elements(x1, x2, atol, rtol)
            # Handle float scalars
            if isinstance(x1, (float, np.floating)) and isinstance(x2, (float, np.floating)):
                if np.isnan(x1) and np.isnan(x2):
                    return True
                if np.isinf(x1) and np.isinf(x2):
                    return (x1 > 0) == (x2 > 0)
                return np.abs(x1 - x2) <= atol + rtol * np.abs(x2)
            return bool(x1 == x2)
        except Exception:
            return False


def compare_datasets(ds1, ds2, atol, rtol):
    """
    Compare two h5py.Dataset objects.
    Returns:
        bool: True if they match within tolerances, False otherwise.
        str: Description of comparison status or details of the difference.
    """
    if ds1.shape != ds2.shape:
        return False, f"Shape mismatch: {ds1.shape} vs {ds2.shape}"

    val1 = ds1[...]
    val2 = ds2[...]

    # Check numeric types
    is_num1 = np.issubdtype(ds1.dtype, np.number)
    is_num2 = np.issubdtype(ds2.dtype, np.number)

    if is_num1 != is_num2:
        return False, f"Type class mismatch: {ds1.dtype} vs {ds2.dtype}"

    # Handle object arrays (non-numeric dtype)
    if ds1.dtype == object or ds2.dtype == object:
        for idx, x1 in np.ndenumerate(val1):
            x2 = val2[idx]
            if not compare_elements(x1, x2, atol, rtol):
                return False, f"Object array values differ at index {idx}: val1={x1}, val2={x2}"
        return True, "Identical object array"

    if not is_num1:
        # Non-numeric comparison (strings, object arrays, etc.) for other dtypes
        if not np.array_equal(val1, val2):
            return False, "Non-numeric values differ"
        return True, "Identical"

    if val1.size == 0:
        return True, "Empty"

    # NaN comparison
    nan1 = np.isnan(val1)
    nan2 = np.isnan(val2)
    if not np.array_equal(nan1, nan2):
        return False, "NaN masks differ"

    # Inf comparison
    inf1 = np.isinf(val1)
    inf2 = np.isinf(val2)
    if not np.array_equal(inf1, inf2):
        return False, "Infinity masks differ"

    # Values to compare (excluding NaNs and Infs)
    valid_mask = ~nan1 & ~inf1
    if not np.any(valid_mask):
        return True, "All NaN/Inf (Identical)"

    v1 = val1[valid_mask]
    v2 = val2[valid_mask]

    abs_diff = np.abs(v1 - v2)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Tolerance: |v1 - v2| <= atol + rtol * |v2|
    tol_limit = atol + rtol * np.abs(v2)
    outside_tol = abs_diff > tol_limit
    num_outside = np.sum(outside_tol)

    if num_outside > 0:
        indices = np.where(outside_tol)[0]
        ex_idx = indices[:3]
        ex_v1 = v1[ex_idx]
        ex_v2 = v2[ex_idx]
        ex_diff = abs_diff[ex_idx]
        details = (
            f"Differs. Max abs diff: {max_abs_diff:.6e}, Mean abs diff: {mean_abs_diff:.6e}. "
            f"{num_outside}/{len(v1)} elements exceed tolerance (atol={atol}, rtol={rtol}). "
            f"Examples: v1={ex_v1}, v2={ex_v2}, diff={ex_diff}"
        )
        return False, details

    return True, f"Identical (max abs diff: {max_abs_diff:.6e})"


def compare_h5_files(file_path1, file_path2, atol, rtol, verbose):
    """
    Compare two HDF5 files recursively.
    """
    errors = []
    identical_datasets = 0
    mismatched_datasets = 0

    try:
        with h5py.File(file_path1, "r") as f1, h5py.File(file_path2, "r") as f2:
            # Gather all items recursively
            items1 = {}
            items2 = {}

            f1.visititems(lambda name, obj: items1.update({name: obj}))
            f2.visititems(lambda name, obj: items2.update({name: obj}))

            keys1 = set(items1.keys())
            keys2 = set(items2.keys())

            # Filter out keys related to purged or C++ dependent models that are decoupled in standalone branch
            purged_patterns = (
                "_Nakashima_cpp",
                "_Parametric_7p",
                "_cpp_p7p",
                "_cpp_tangential",
                "/simulated_trajectories/",
            )
            keys1 = {k for k in keys1 if not any(pat in k for pat in purged_patterns)}
            keys2 = {k for k in keys2 if not any(pat in k for pat in purged_patterns)}

            # Structure checks
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1

            if missing_in_2:
                errors.append(f"Paths in File 1 but missing in File 2: {sorted(list(missing_in_2))}")
            if missing_in_1:
                errors.append(f"Paths in File 2 but missing in File 1: {sorted(list(missing_in_1))}")

            # Compare common keys
            common_keys = keys1 & keys2
            for key in sorted(common_keys):
                obj1 = items1[key]
                obj2 = items2[key]

                # Check if types match (Group vs Dataset)
                type1 = type(obj1).__name__
                type2 = type(obj2).__name__
                if type1 != type2:
                    errors.append(f"Type mismatch at '{key}': {type1} vs {type2}")
                    continue

                if isinstance(obj1, h5py.Dataset):
                    success, detail = compare_datasets(obj1, obj2, atol, rtol)
                    if success:
                        identical_datasets += 1
                        if verbose:
                            print(f"  [OK]  {key}: {detail}")
                    else:
                        mismatched_datasets += 1
                        errors.append(f"Value mismatch at '{key}': {detail}")

    except Exception as e:
        errors.append(f"Failed to open or read files: {e}")

    return len(errors) == 0, errors, identical_datasets, mismatched_datasets


def main():
    parser = argparse.ArgumentParser(description="Compare updated HDF5 files recursively.")
    parser.add_argument(
        "--dir1",
        type=pathlib.Path,
        default=pathlib.Path("../Data/test_updatehdf5/ace_eval_full"),
        help="First directory containing HDF5 files",
    )
    parser.add_argument(
        "--dir2",
        type=pathlib.Path,
        default=pathlib.Path("../Data/test_updatehdf5/ace_eval_standalone"),
        help="Second directory containing HDF5 files",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for float comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for float comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details of identical datasets",
    )
    args = parser.parse_args()

    if not args.dir1.is_dir():
        print(f"Error: --dir1 '{args.dir1}' does not exist or is not a directory.")
        sys.exit(1)
    if not args.dir2.is_dir():
        print(f"Error: --dir2 '{args.dir2}' does not exist or is not a directory.")
        sys.exit(1)

    print(f"Comparing HDF5 files in:")
    print(f"  Dir 1 (Full):       {args.dir1.resolve()}")
    print(f"  Dir 2 (Standalone): {args.dir2.resolve()}")
    print(f"Tolerances: atol={args.atol}, rtol={args.rtol}\n")

    # Find all h5 files in dir1
    files1 = sorted(list(args.dir1.rglob("*.h5")))
    if not files1:
        print("No HDF5 files (*.h5) found in Dir 1.")
        sys.exit(0)

    matched_count = 0
    mismatched_files = 0
    missing_files = []
    all_passed = True

    for f1_path in files1:
        rel_path = f1_path.relative_to(args.dir1)
        f2_path = args.dir2 / rel_path

        print(f"Comparing {rel_path}...")

        if not f2_path.is_file():
            print(f"  [ERROR] File missing in Dir 2")
            missing_files.append(rel_path)
            all_passed = False
            continue

        matched_count += 1
        success, errors, num_ok, num_fail = compare_h5_files(f1_path, f2_path, args.atol, args.rtol, args.verbose)

        if success:
            print(f"  [PASS] All {num_ok} datasets identical.")
        else:
            print(f"  [FAIL] {num_fail} datasets differed, {num_ok} identical.")
            for err in errors:
                print(f"    - {err}")
            mismatched_files += 1
            all_passed = False

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total files in Dir 1:          {len(files1)}")
    print(f"Compared successfully:         {matched_count}")
    print(f"Identical files:               {matched_count - mismatched_files}")
    print(f"Files with differences:        {mismatched_files}")
    print(f"Files missing in Dir 2:        {len(missing_files)}")

    if missing_files:
        print("\nMissing files in Dir 2:")
        for mf in missing_files:
            print(f"  - {mf}")

    if not all_passed:
        sys.exit(1)
    else:
        print("\nAll compared files are completely identical within tolerance!")
        sys.exit(0)


if __name__ == "__main__":
    main()
