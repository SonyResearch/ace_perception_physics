# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Drag Coefficient Fitting Analysis

This notebook/script analyzes the drag coefficient data and fits a smooth model.
"""

import numpy as np
import matplotlib.pyplot as plt

from ace_evaluation.utilities.data_classes import MatchCollection
from data_plotter.data_processor import build_flight_segment_cache


def load_data_for_fitting(match_collection_path: str, min_confidence: float = 0.5, min_duration: float = 0.15):
    """
    Load and filter drag coefficient data for fitting.

    Args:
        match_collection_path: Path to match collection HDF5 file or directory
        min_confidence: Minimum confidence threshold (default 0.5)
        min_duration: Minimum duration threshold in seconds (default 0.15)

    Returns:
        Dictionary with filtered data arrays
    """
    # Load match collection
    print(f"Loading match collection from {match_collection_path}...")
    mc = MatchCollection(match_collection_path)

    # Build cache
    print("Building flight segment cache...")
    cache = build_flight_segment_cache(mc)
    print(f"Total segments: {cache.n_segments}")

    # Extract data with filters
    radius_ball = 0.02  # 2cm radius

    # Filter mask: has aero data + valid cd_opt + confidence + duration
    mask = (
        cache.has_aero &
        ~np.isnan(cache.cd_opt) &
        ~np.isnan(cache.v_eff_drag) &
        (cache.min_confidence >= min_confidence) &
        (cache.durations >= min_duration)
    )

    print(f"Segments with aero data: {np.sum(cache.has_aero)}")
    print(f"Segments with valid cd_opt: {np.sum(~np.isnan(cache.cd_opt))}")
    print(f"Segments with confidence >= {min_confidence}: {np.sum(cache.min_confidence >= min_confidence)}")
    print(f"Segments with duration >= {min_duration}s: {np.sum(cache.durations >= min_duration)}")
    print(f"Segments passing all filters: {np.sum(mask)}")

    # Extract filtered data
    v_eff_drag = cache.v_eff_drag[mask]
    cd_opt = cache.cd_opt[mask]

    # Calculate spin magnitude and spin ratio from first point data
    wx = cache.wx_first[mask]
    wy = cache.wy_first[mask]
    wz = cache.wz_first[mask]
    spin_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    spin_ratio = (radius_ball * spin_mag) / np.maximum(v_eff_drag, 1e-9)

    # Also get velocity magnitude for comparison
    vx = cache.vx_opt_first[mask]
    vy = cache.vy_opt_first[mask]
    vz = cache.vz_opt_first[mask]
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Additional metadata
    confidence = cache.min_confidence[mask]
    duration = cache.durations[mask]
    player_type = cache.player_types[mask]  # 1=robot, 2=player

    return {
        'v_eff_drag': v_eff_drag,
        'vel_mag': vel_mag,
        'spin_mag': spin_mag,
        'spin_ratio': spin_ratio,
        'cd_opt': cd_opt,
        'confidence': confidence,
        'duration': duration,
        'player_type': player_type,
    }


def analyze_data_distribution(data: dict):
    """Analyze and visualize the data distribution."""
    v = data['v_eff_drag']
    S = data['spin_ratio']
    cd = data['cd_opt']
    spin_mag = data['spin_mag']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Histogram of v_eff_drag
    ax = axes[0, 0]
    ax.hist(v, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('v_eff_drag (m/s)')
    ax.set_ylabel('Count')
    ax.set_title(f'Velocity Distribution (n={len(v)})')
    ax.axvline(13, color='r', linestyle='--', label='v=13 m/s (sparse above)')
    ax.legend()

    # 2. Histogram of spin ratio
    ax = axes[0, 1]
    ax.hist(S, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Spin Ratio S = r·ω/v')
    ax.set_ylabel('Count')
    ax.set_title('Spin Ratio Distribution')

    # 3. Histogram of CD
    ax = axes[0, 2]
    ax.hist(cd, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('C_D')
    ax.set_ylabel('Count')
    ax.set_title('Drag Coefficient Distribution')

    # 4. Scatter: v vs S colored by CD
    ax = axes[1, 0]
    sc = ax.scatter(v, S, c=cd, cmap='viridis', alpha=0.5, s=5)
    ax.set_xlabel('v_eff_drag (m/s)')
    ax.set_ylabel('Spin Ratio S')
    ax.set_title('Data Coverage (colored by C_D)')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 2)
    plt.colorbar(sc, ax=ax, label='C_D')

    # 5. Scatter: v vs CD colored by S
    ax = axes[1, 1]
    sc = ax.scatter(v, cd, c=S, cmap='plasma', alpha=0.5, s=5)
    ax.set_xlabel('v_eff_drag (m/s)')
    ax.set_ylabel('C_D')
    ax.set_title('CD vs Velocity (colored by S)')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    plt.colorbar(sc, ax=ax, label='S')

    # 6. Scatter: S vs CD colored by v
    ax = axes[1, 2]
    sc = ax.scatter(S, cd, c=v, cmap='coolwarm', alpha=0.5, s=5)
    ax.set_xlabel('Spin Ratio S')
    ax.set_ylabel('C_D')
    ax.set_title('CD vs Spin Ratio (colored by v)')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    plt.colorbar(sc, ax=ax, label='v (m/s)')

    plt.tight_layout()
    return fig


def analyze_velocity_slices(data: dict, v_slices: list = None):
    """Analyze CD vs S for different velocity slices."""
    if v_slices is None:
        v_slices = [
            (0, 5, 'v < 5'),
            (5, 8, '5 ≤ v < 8'),
            (8, 11, '8 ≤ v < 11'),
            (11, 14, '11 ≤ v < 14'),
            (14, 18, '14 ≤ v < 18'),
            (18, 30, 'v ≥ 18'),
        ]

    v = data['v_eff_drag']
    S = data['spin_ratio']
    cd = data['cd_opt']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (v_min, v_max, label) in enumerate(v_slices):
        ax = axes[i]
        mask = (v >= v_min) & (v < v_max)
        v_slice = v[mask]
        S_slice = S[mask]
        cd_slice = cd[mask]

        if len(S_slice) > 0:
            ax.scatter(S_slice, cd_slice, c=v_slice, cmap='coolwarm', alpha=0.5, s=10,
                      vmin=v_min, vmax=v_max)

            # Compute binned statistics
            S_bins = np.linspace(0, 1.5, 16)
            S_centers = 0.5 * (S_bins[:-1] + S_bins[1:])
            cd_means = []
            cd_stds = []
            for j in range(len(S_bins) - 1):
                bin_mask = (S_slice >= S_bins[j]) & (S_slice < S_bins[j+1])
                if np.sum(bin_mask) > 5:
                    cd_means.append(np.mean(cd_slice[bin_mask]))
                    cd_stds.append(np.std(cd_slice[bin_mask]))
                else:
                    cd_means.append(np.nan)
                    cd_stds.append(np.nan)

            cd_means = np.array(cd_means)
            cd_stds = np.array(cd_stds)
            valid = ~np.isnan(cd_means)
            ax.errorbar(S_centers[valid], cd_means[valid], yerr=cd_stds[valid],
                       fmt='k-o', linewidth=2, markersize=4, label='Mean ± std')

        ax.set_xlabel('Spin Ratio S')
        ax.set_ylabel('C_D')
        ax.set_title(f'{label} (n={np.sum(mask)})')
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0.2, 0.8)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage - update path to your data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to match collection')
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--min_duration', type=float, default=0.15)
    parser.add_argument('--output_dir', type=str, default='./drag_analysis')
    args = parser.parse_args()

    # Load data
    data = load_data_for_fitting(args.data_path, args.min_confidence, args.min_duration)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Analyze distribution
    fig1 = analyze_data_distribution(data)
    fig1.savefig(output_dir / 'data_distribution.png', dpi=150)
    print(f"Saved data_distribution.png")

    # Analyze velocity slices
    fig2 = analyze_velocity_slices(data)
    fig2.savefig(output_dir / 'velocity_slices.png', dpi=150)
    print(f"Saved velocity_slices.png")

    plt.show()
