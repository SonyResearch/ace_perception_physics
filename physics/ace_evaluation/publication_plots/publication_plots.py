# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Publication-ready Plotly figures for aerodynamics and contact models.

Usage:
    python publication_plots.py /path/to/hdf5/data/folder [--output-dir plots/publication]
    python publication_plots.py --json /path/to/manifest.json [--output-dir plots/publication]

Generates:
    1. Drag coefficient vs spin ratio (v = 8–16 m/s), with V1 model lines
    2. Drag coefficient vs spin ratio (v ≤ 8 m/s), no model lines
    3. Magnus coefficient vs spin (v = 8–16 m/s), with V2 model lines
    4. Aerodynamics fitness box plot (Optimal, Residual0805, Nakashima)
    5. Table contact box plot (NakashimaITTF, NakashimaPaper, Residual0805)
    6. Racket contact box plot (Nakashima, Nakashima refined, ONNX alex refined) grouped by contact offset
"""

import csv
import pathlib
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project imports ──────────────────────────────────────────────────────────
import sys
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from ace_evaluation.utilities.data_classes import MatchCollection
from data_plotter.data_processor import DataProcessor, build_flight_segment_cache
from data_plotter.plot_widgets.aero_estimates import (
    get_drag_estimate_v1,
    get_magnus_estimate_v2,
)

# ── Publication theme ────────────────────────────────────────────────────────
FONT_FAMILY = "Computer Modern, Times New Roman, serif"
FONT_SIZE = 20
AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
COLORBAR_FONT_SIZE = 16
LINE_WIDTH = 3.5
LINE_BORDER_WIDTH = 5.5
MARKER_SIZE = 5
PLOT_WIDTH = 900
PLOT_HEIGHT = 600
SUBPLOT_HEIGHT = 500

# Turbo-like discrete palette for velocity coloring
VELOCITY_COLORSCALE = "Turbo"

# Box plot model colors
MODEL_COLORS = {
    #"OPT": "rgba(66, 133, 244, 0.8)",
    #"0226": "rgba(255, 165, 0, 0.8)",
    "Optimal": "rgba(52, 168, 83, 0.8)",
    #"NakashimaITTF": "rgba(52, 168, 83, 0.8)",
    #"NakashimaPaper": "rgba(255, 165, 0, 0.8)",
    #"Residual0805": "rgba(200, 50, 200, 0.8)",
    "Nakashima et al.": "rgba(0, 114, 178, 0.8)",
    "Nakashima et al. (refined)": "rgba(0, 178, 114, 0.8)",
    #"ONNX alex (refined)": "rgba(0, 158, 115, 0.8)",
    "Proposed": "rgba(200, 50, 200, 0.8)",
    "Dürr et al.": "rgba(255, 127, 14, 0.8)",
}


def _pub_layout(**overrides):
    """Return a base layout dict for publication figures."""
    layout = dict(
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=70, r=30, t=50, b=60),
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    layout.update(overrides)
    return layout


def _pub_axis(title, **overrides):
    """Return axis styling dict."""
    ax = dict(
        title=dict(text=title, font=dict(size=AXIS_FONT_SIZE)),
        tickfont=dict(size=TICK_FONT_SIZE),
        showgrid=True,
        gridcolor="rgba(200,200,200,0.5)",
        gridwidth=1,
        zeroline=False,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
    )
    ax.update(overrides)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_folder: pathlib.Path):
    """Load MatchCollection + DataProcessor from HDF5 folder."""
    dp = DataProcessor()
    mc = dp.load_data(data_folder)
    return dp, mc


def load_data_from_json(json_path: pathlib.Path):
    """Load MatchCollection + DataProcessor from a JSON manifest file."""
    dp = DataProcessor()
    mc = dp.load_data_from_json(str(json_path))
    return dp, mc


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1: Drag C_D vs spin ratio (v = 8–16 m/s) with model lines
# ═══════════════════════════════════════════════════════════════════════════════

def plot_drag_vs_spin_ratio(dp, mc, vel_min=8, vel_max=16, show_model=True,
                            title_suffix=""):
    """Scatter of C_D vs spin ratio, colored by velocity, with optional model lines."""
    data = dp.extract_drag_coefficient_data(mc, player_type=None)
    (vel_mag, v_eff_drag, spin_mag, spin_ratio, cd_opt,
     flight_segments, metadata_list, *_, confidence_list, rmse_opt_list,
     inconsistent_list, drag_impulse_proxy_list) = data

    vel = np.asarray(v_eff_drag)
    sr = np.asarray(spin_ratio)
    cd = np.asarray(cd_opt)
    conf = np.asarray(confidence_list)
    inconsistent = np.asarray(inconsistent_list, dtype=bool)

    # Filtering
    radius_ball = 0.02
    sr_veff = (radius_ball * np.asarray(spin_mag)) / np.maximum(vel, 1e-9)

    mask = (
        (vel >= vel_min) & (vel <= vel_max) &
        (conf >= 0.5) & (~inconsistent) &
        ~np.isnan(vel) & ~np.isnan(sr_veff) & ~np.isnan(cd)
    )
    vel_f, sr_f, cd_f = vel[mask], sr_veff[mask], cd[mask]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=sr_f, y=cd_f,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=vel_f,
            colorscale=VELOCITY_COLORSCALE,
            cmin=vel_min, cmax=vel_max,
            colorbar=dict(
                title=dict(text="v<sub>eff</sub> (m/s)", font=dict(size=COLORBAR_FONT_SIZE)),
                tickfont=dict(size=TICK_FONT_SIZE),
                thickness=15, len=0.6,
            ),
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Data",
        showlegend=False,
    ))

    # Model lines
    if show_model:
        sr_theory = np.linspace(0, 2, 300)
        import matplotlib.cm as mpl_cm
        cmap = mpl_cm.get_cmap("turbo")
        for v_val in np.arange(vel_min, vel_max + 1, 2):
            cd_theory = get_drag_estimate_v1(v_val, sr_theory)
            norm_v = (v_val - vel_min) / max(vel_max - vel_min, 1)
            rgba = cmap(norm_v)
            color = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
            # Black border line (wider, behind)
            fig.add_trace(go.Scatter(
                x=sr_theory, y=cd_theory,
                mode="lines",
                line=dict(color="black", width=LINE_BORDER_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=sr_theory, y=cd_theory,
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH),
                name=f"V1 model v={v_val:.0f}",
                showlegend=False,
            ))

    fig.update_layout(
        **_pub_layout(),
        xaxis=_pub_axis("Spin Ratio (ωr / v)", range=[0, 2]),
        yaxis=_pub_axis("C<sub>D</sub>", range=[0.3, 0.65]),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1b: Drag C_D vs spin ratio at discrete velocity bands
# ═══════════════════════════════════════════════════════════════════════════════

def plot_drag_discrete_velocity_bands(dp, mc,
                                      ref_velocities=(2.5, 7.5, 12.5, 17.5),
                                      half_width=0.5):
    """Scatter of C_D vs spin ratio at discrete velocity bands with V1 model lines.

    Data is shown only for v_eff within ±half_width of each reference velocity.
    Model lines are drawn at the exact reference velocities.
    """
    data = dp.extract_drag_coefficient_data(mc, player_type=None)
    (vel_mag, v_eff_drag, spin_mag, spin_ratio, cd_opt,
     flight_segments, metadata_list, *_, confidence_list, rmse_opt_list,
     inconsistent_list, drag_impulse_proxy_list) = data

    vel = np.asarray(v_eff_drag)
    cd = np.asarray(cd_opt)
    conf = np.asarray(confidence_list)
    inconsistent = np.asarray(inconsistent_list, dtype=bool)

    radius_ball = 0.02
    sr_veff = (radius_ball * np.asarray(spin_mag)) / np.maximum(vel, 1e-9)

    # Base quality mask (no velocity filter yet)
    base_mask = (
        (conf >= 0.5) & (~inconsistent) &
        ~np.isnan(vel) & ~np.isnan(sr_veff) & ~np.isnan(cd)
    )

    # Build per-band velocity mask
    band_mask = np.zeros(len(vel), dtype=bool)
    for v_ref in ref_velocities:
        band_mask |= (vel >= v_ref - half_width) & (vel <= v_ref + half_width)
    combined_mask = base_mask & band_mask

    vel_f = vel[combined_mask]
    sr_f = sr_veff[combined_mask]
    cd_f = cd[combined_mask]

    vel_all_min = min(ref_velocities) - half_width
    vel_all_max = max(ref_velocities) + half_width

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=sr_f, y=cd_f,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=vel_f,
            colorscale=VELOCITY_COLORSCALE,
            cmin=vel_all_min, cmax=vel_all_max,
            colorbar=dict(
                title=dict(text="v<sub>eff</sub> (m/s)",
                           font=dict(size=COLORBAR_FONT_SIZE)),
                tickfont=dict(size=TICK_FONT_SIZE),
                tickvals=list(ref_velocities),
                thickness=15, len=0.6,
            ),
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Data",
        showlegend=False,
    ))

    # Model lines at reference velocities only
    sr_theory = np.linspace(0, 2, 300)
    import matplotlib.cm as mpl_cm
    cmap = mpl_cm.get_cmap("turbo")
    for v_val in ref_velocities:
        cd_theory = get_drag_estimate_v1(v_val, sr_theory)
        norm_v = (v_val - vel_all_min) / max(vel_all_max - vel_all_min, 1)
        rgba = cmap(norm_v)
        color = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
        # Black border line (wider, behind)
        fig.add_trace(go.Scatter(
            x=sr_theory, y=cd_theory,
            mode="lines",
            line=dict(color="black", width=LINE_BORDER_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sr_theory, y=cd_theory,
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
            name=f"V1 model v={v_val:.1f}",
            showlegend=False,
        ))

    fig.update_layout(
        **_pub_layout(),
        xaxis=_pub_axis("Spin Ratio (ωr / v)", range=[0, 2]),
        yaxis=_pub_axis("C<sub>D</sub>", range=[0.3, 0.65]),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2: Drag C_D vs spin ratio (v ≤ 8 m/s), no model
# ═══════════════════════════════════════════════════════════════════════════════

def plot_drag_low_velocity(dp, mc):
    """Scatter of C_D vs spin ratio for v ≤ 8 m/s, no model lines."""
    fig = plot_drag_vs_spin_ratio(dp, mc, vel_min=0, vel_max=8, show_model=False,
                                  title_suffix="(low velocity)")
    fig.update_yaxes(range=[0.3, 0.8])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Magnus C_M vs spin (v = 8–16 m/s) with model lines
# ═══════════════════════════════════════════════════════════════════════════════

def plot_magnus_vs_spin(dp, mc, vel_min=8, vel_max=16):
    """Scatter of C_M vs spin magnitude, colored by velocity, with V2 model lines."""
    data = dp.extract_magnus_coefficient_data(mc, player_type=None)
    (vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp, cm_opt,
     flight_segments, metadata_list, *_, confidence_list, rmse_opt_list,
     inconsistent_list, magnus_impulse_proxy_list) = data

    vel = np.asarray(v_eff_magnus)
    spin = np.asarray(spin_mag)
    cm = np.asarray(cm_opt)
    conf = np.asarray(confidence_list)
    inconsistent = np.asarray(inconsistent_list, dtype=bool)

    mask = (
        (vel >= vel_min) & (vel <= vel_max) &
        (conf >= 0.5) & (~inconsistent) &
        ~np.isnan(vel) & ~np.isnan(spin) & ~np.isnan(cm)
    )
    vel_f, spin_f, cm_f = vel[mask], spin[mask], cm[mask]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=spin_f, y=cm_f,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=vel_f,
            colorscale=VELOCITY_COLORSCALE,
            cmin=vel_min, cmax=vel_max,
            colorbar=dict(
                title=dict(text="v<sub>eff</sub> (m/s)", font=dict(size=COLORBAR_FONT_SIZE)),
                tickfont=dict(size=TICK_FONT_SIZE),
                thickness=15, len=0.6,
            ),
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Data",
        showlegend=False,
    ))

    # Model lines (V2)
    spin_theory = np.linspace(0, 900, 300)
    import matplotlib.cm as mpl_cm
    cmap = mpl_cm.get_cmap("turbo")
    for v_val in np.arange(vel_min, vel_max + 1, 2):
        cm_theory = get_magnus_estimate_v2(v_val, spin_theory)
        norm_v = (v_val - vel_min) / max(vel_max - vel_min, 1)
        rgba = cmap(norm_v)
        color = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
        # Black border line (wider, behind)
        fig.add_trace(go.Scatter(
            x=spin_theory, y=cm_theory,
            mode="lines",
            line=dict(color="black", width=LINE_BORDER_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=spin_theory, y=cm_theory,
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
            name=f"V2 model v={v_val:.0f}",
            showlegend=False,
        ))

    fig.update_layout(
        **_pub_layout(),
        xaxis=_pub_axis("Spin magnitude ω (rad/s)", range=[0, 900]),
        yaxis=_pub_axis("C<sub>M</sub>", range=[0, 0.35]),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3b: Magnus C_M vs spin at discrete velocity bands
# ═══════════════════════════════════════════════════════════════════════════════

def plot_magnus_discrete_velocity_bands(dp, mc,
                                        ref_velocities=(2, 3.5, 7.5, 10.5, 13.5, 17.0),
                                        half_width=0.5):
    """Scatter of C_M vs spin magnitude at discrete velocity bands with V2 model lines.

    Data is shown only for v_eff within ±half_width of each reference velocity.
    Model lines are drawn at the exact reference velocities.
    """
    data = dp.extract_magnus_coefficient_data(mc, player_type=None)
    (vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp, cm_opt,
     flight_segments, metadata_list, *_, confidence_list, rmse_opt_list,
     inconsistent_list, magnus_impulse_proxy_list) = data

    vel = np.asarray(v_eff_magnus)
    spin = np.asarray(spin_mag)
    cm = np.asarray(cm_opt)
    conf = np.asarray(confidence_list)
    inconsistent = np.asarray(inconsistent_list, dtype=bool)

    # Base quality mask (no velocity filter yet)
    base_mask = (
        (conf >= 0.5) & (~inconsistent) &
        ~np.isnan(vel) & ~np.isnan(spin) & ~np.isnan(cm)
    )

    # Build per-band velocity mask
    band_mask = np.zeros(len(vel), dtype=bool)
    for v_ref in ref_velocities:
        band_mask |= (vel >= v_ref - half_width) & (vel <= v_ref + half_width)
    combined_mask = base_mask & band_mask

    vel_f = vel[combined_mask]
    spin_f = spin[combined_mask]
    cm_f = cm[combined_mask]

    vel_all_min = min(ref_velocities) - half_width
    vel_all_max = max(ref_velocities) + half_width

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=spin_f, y=cm_f,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=vel_f,
            colorscale=VELOCITY_COLORSCALE,
            cmin=vel_all_min, cmax=vel_all_max,
            colorbar=dict(
                title=dict(text="v<sub>eff</sub> (m/s)",
                           font=dict(size=COLORBAR_FONT_SIZE)),
                tickfont=dict(size=TICK_FONT_SIZE),
                tickvals=list(ref_velocities),
                thickness=15, len=0.6,
            ),
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Data",
        showlegend=False,
    ))

    # Model lines at reference velocities only
    spin_theory = np.linspace(0, 900, 300)
    import matplotlib.cm as mpl_cm
    cmap = mpl_cm.get_cmap("turbo")
    for v_val in ref_velocities:
        cm_theory = get_magnus_estimate_v2(v_val, spin_theory)
        norm_v = (v_val - vel_all_min) / max(vel_all_max - vel_all_min, 1)
        rgba = cmap(norm_v)
        color = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
        # Black border line (wider, behind)
        fig.add_trace(go.Scatter(
            x=spin_theory, y=cm_theory,
            mode="lines",
            line=dict(color="black", width=LINE_BORDER_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=spin_theory, y=cm_theory,
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
            name=f"V2 model v={v_val:.1f}",
            showlegend=False,
        ))

    fig.update_layout(
        **_pub_layout(),
        xaxis=_pub_axis("Spin magnitude ω (rad/s)", range=[0, 900]),
        yaxis=_pub_axis("C<sub>M</sub>", range=[0, 0.35]),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4: Aerodynamics position RMSE box plot (OPT, 0226, Nakashima)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_aero_boxplot(dp, mc):
    """
    Aerodynamics box plot: 3D position RMSE per flight segment.
    Models: OPT, 0226, Nakashima (compared to APS observations).
    Values in mm.
    """
    aero_data = dp.extract_aero_summary_error_data(mc)
    error_dict = aero_data[0]
    confidence = aero_data[1]
    durations = aero_data[2]

    # Quality filters: spin confidence >= 0.5 and OPT RMSE < 20mm
    conf_mask = np.asarray(confidence) >= 0.5
    opt_rmse = error_dict.get("opt")
    rmse_mask = np.ones(len(conf_mask), dtype=bool)
    if opt_rmse is not None:
        rmse_mm_all = np.where(np.isfinite(opt_rmse), opt_rmse * 1000.0, np.nan)
        rmse_mask = np.isfinite(rmse_mm_all) & (rmse_mm_all < 20.0)
    quality_mask = conf_mask & rmse_mask

    models = [
        #("Optimal", "opt"),
        ("Nakashima et al.", "nakashima"),
        ("Dürr et al.", "0426"),
        ("Proposed", "0226"),
    ]

    fig = go.Figure()

    for m_label, m_key in models:
        rmse_vals = error_dict.get(m_key)
        if rmse_vals is None:
            continue
        # Apply quality mask then filter out NaN
        filtered = rmse_vals[quality_mask]
        valid = np.isfinite(filtered)
        rmse_mm = filtered[valid] * 1000.0  # m → mm
        rmse_mm = _clip_iqr(rmse_mm)

        fig.add_trace(go.Violin(
            y=rmse_mm,
            name=m_label,
            marker_color=MODEL_COLORS.get(m_label, "gray"),
            line_color=MODEL_COLORS.get(m_label, "gray"),
            meanline_visible=True,
            points=False,
        ))

    fig.update_layout(
        **_pub_layout(
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH,
        ),
        yaxis=_pub_axis("Position RMSE (mm)"),
        xaxis=_pub_axis(""),
    )

    # ── Terminal statistics ──────────────────────────────────────────────────
    print("\n  [Aero violin] Improvement of Proposed over baselines (lower RMSE = better):")
    # Collect per-model clipped RMSE arrays
    clipped = {}
    for m_label, m_key in models:
        rmse_vals = error_dict.get(m_key)
        if rmse_vals is None:
            continue
        filtered = rmse_vals[quality_mask]
        valid = np.isfinite(filtered)
        rmse_mm = filtered[valid] * 1000.0
        clipped[m_label] = _clip_iqr(rmse_mm)

    if "Proposed" in clipped:
        cur = clipped["Proposed"]
        cur_med = np.median(cur)
        cur_p75 = np.percentile(cur, 75)
        for baseline in ["Nakashima et al.", "Dürr et al."]:
            if baseline not in clipped:
                continue
            bl = clipped[baseline]
            bl_med = np.median(bl)
            bl_p75 = np.percentile(bl, 75)
            imp_med = (bl_med - cur_med) / bl_med * 100
            imp_p75 = (bl_p75 - cur_p75) / bl_p75 * 100
            print(f"    vs {baseline}: median {imp_med:+.1f}%, 75th pctl {imp_p75:+.1f}%")
            print(f"      (Proposed median={cur_med:.2f}mm, 75th={cur_p75:.2f}mm | "
                  f"{baseline} median={bl_med:.2f}mm, 75th={bl_p75:.2f}mm)")

    return fig


def _clip_iqr(arr):
    """Remove outliers beyond 1.5*IQR fences."""
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]


def _build_mask(data, conf_min=0.0, conf_max=1.0, rmse_max=None, fitness_post_max=None):
    """Build a boolean mask matching the data_plotter's filtering logic.

    Missing confidence defaults to 1.0 (passes all filters).
    When rmse_max is set, NaN RMSE entries are excluded.
    When fitness_post_max is set, points with fitness_post > max are excluded
    (NaN fitness_post passes, matching data_plotter behaviour).
    """
    n = len(next(iter(data.values())))
    mask = np.ones(n, dtype=bool)

    # Confidence filter: missing confidence → treated as 1.0 (passes)
    confidence = np.asarray(data.get("confidence", np.ones(n)))
    if conf_min > 0:
        mask &= confidence >= conf_min
    if conf_max < 1:
        mask &= confidence <= conf_max

    # RMSE filter: when active, NaN entries are excluded (matching data_plotter)
    if rmse_max is not None:
        max_rmse = np.asarray(data.get("max_rmse_opt", np.full(n, np.nan)))
        rmse_valid = ~np.isnan(max_rmse)
        mask &= rmse_valid & (max_rmse <= rmse_max)

    # Fitness-post filter: NaN passes (no data → keep), matching data_plotter
    if fitness_post_max is not None and fitness_post_max > 0:
        fp = np.asarray(data.get("fitness_post", np.full(n, np.nan)))
        has_fp = ~np.isnan(fp)
        mask &= (~has_fp) | (fp <= fitness_post_max)

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 5: Table contact box plot (NakashimaITTF, NakashimaPaper, Residual0805)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_table_contact_boxplot(dp, mc):
    """
    Table contact violin plot: 2 rows (velocity / spin), components as x-axis groups.
    Models: Nakashima, Proposed (Residual0805).

    Model predictions are read from HDF5 (computed by table_contacts.py).
    """
    contact_data = dp.extract_table_contact_data(mc, player_type=None)

    # Diagnostic: check which model keys have valid (non-NaN) data
    for _sfx in ["paper", "0426", "res0805"]:
        _key = f"vx_post_{_sfx}"
        if _key in contact_data:
            _arr = np.asarray(contact_data[_key])
            _n_valid = np.sum(np.isfinite(_arr))
            print(f"  [TCM-plot] {_key}: {_n_valid}/{len(_arr)} valid")
        else:
            print(f"  [TCM-plot] {_key}: NOT FOUND in contact_data")

    models = [
        #("NakashimaITTF", "ittf"),
        ("Nakashima et al.", "paper"),
        ("Dürr et al.", "0426"),
        ("Proposed", "res0805"),
    ]

    # Confidence + RMSE filter
    mask = _build_mask(contact_data, conf_min=0.5, rmse_max=0.020)

    vel_comps = [("vx", "v<sub>x</sub>"), ("vy", "v<sub>y</sub>"), ("vz", "v<sub>z</sub>")]
    spin_comps = [("wx", "ω<sub>x</sub>"), ("wy", "ω<sub>y</sub>"), ("wz", "ω<sub>z</sub>")]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Velocity error (m/s)", "Spin error (rad/s)"],
        vertical_spacing=0.18,
    )

    n_models = len(models)
    # Per-model x offsets: centre models around each category position
    model_offsets = np.linspace(-0.2, 0.2, n_models)
    violin_width = 0.35 / n_models  # width per violin

    for row, (comp_group, unit) in enumerate(
        [(vel_comps, "m/s"), (spin_comps, "rad/s")], start=1
    ):
        is_spin = (row == 2)
        tick_positions = list(range(len(comp_group)))
        tick_labels = [clabel for _, clabel in comp_group]

        for comp_idx, (comp, clabel) in enumerate(comp_group):
            obs_key = f"{comp}_post"

            for m_idx, (m_label, m_suffix) in enumerate(models):
                model_key = f"{comp}_post_{m_suffix}"
                if model_key not in contact_data or obs_key not in contact_data:
                    continue
                err = (np.asarray(contact_data[model_key]) - np.asarray(contact_data[obs_key]))[mask]
                err = err[np.isfinite(err)]
                if len(err) == 0:
                    continue
                err = _clip_iqr(err)

                x_pos = comp_idx + model_offsets[m_idx]
                fig.add_trace(go.Violin(
                    y=err,
                    x=np.full(len(err), x_pos),
                    name=m_label,
                    marker_color=MODEL_COLORS.get(m_label, "gray"),
                    line_color=MODEL_COLORS.get(m_label, "gray"),
                    meanline_visible=True,
                    points=False,
                    scalemode="width" if is_spin else "count",
                    width=violin_width,
                    showlegend=(comp_idx == 0 and row == 1),
                    legendgroup=m_label,
                ), row=row, col=1)

        fig.update_yaxes(title_text=f"Error ({unit})", row=row, col=1,
                         **{k: v for k, v in _pub_axis("").items()
                            if k not in ("title",)})
        fig.update_xaxes(tickvals=tick_positions, ticktext=tick_labels,
                         row=row, col=1,
                         **{k: v for k, v in _pub_axis("").items()
                            if k not in ("title",)})

    fig.update_layout(
        **_pub_layout(
            height=SUBPLOT_HEIGHT * 2,
            width=PLOT_WIDTH,
        ),
    )

    # ── Terminal statistics ──────────────────────────────────────────────────
    print("\n  [Table contact] Per-component improvement of Proposed over baselines (absolute error):")
    all_comps = vel_comps + spin_comps
    for comp, clabel in all_comps:
        obs_key = f"{comp}_post"
        if obs_key not in contact_data:
            continue
        # Collect per-model absolute errors
        comp_errors = {}
        for m_label, m_suffix in models:
            model_key = f"{comp}_post_{m_suffix}"
            if model_key not in contact_data:
                continue
            err = np.abs((np.asarray(contact_data[model_key]) - np.asarray(contact_data[obs_key]))[mask])
            err = err[np.isfinite(err)]
            if len(err) == 0:
                continue
            comp_errors[m_label] = _clip_iqr(err)

        if "Proposed" not in comp_errors:
            continue
        cur = comp_errors["Proposed"]
        cur_stats = (np.percentile(cur, 25), np.median(cur), np.percentile(cur, 75))
        print(f"    {comp} — Proposed: 25th={cur_stats[0]:.4f}, median={cur_stats[1]:.4f}, 75th={cur_stats[2]:.4f}")
        for baseline in ["Nakashima et al.", "Dürr et al."]:
            if baseline not in comp_errors:
                continue
            bl = comp_errors[baseline]
            bl_stats = (np.percentile(bl, 25), np.median(bl), np.percentile(bl, 75))
            imp = tuple((bl_stats[i] - cur_stats[i]) / bl_stats[i] * 100 for i in range(3))
            print(f"      vs {baseline}: 25th {imp[0]:+.1f}%, median {imp[1]:+.1f}%, 75th {imp[2]:+.1f}%")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 6: Racket contact box plot (v_mag, w_mag) grouped by contact offset
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_magnitude(data, mask, components, suffix=None):
    """Compute magnitude = sqrt(sum(comp²)) for a set of components."""
    sq_sum = np.zeros(mask.sum())
    for comp in components:
        key = f"{comp}_post_{suffix}" if suffix else f"{comp}_post"
        if key in data:
            sq_sum += np.asarray(data[key])[mask] ** 2
    return np.sqrt(sq_sum)


def export_rcm_csv(dp, mc, output_path: pathlib.Path):
    """Export per-contact RCM error data to CSV for comparison with data_plotter."""
    rcm_data = dp.extract_rcm_data(mc)
    base_mask = _build_mask(rcm_data, conf_min=0.5, rmse_max=0.020,
                            fitness_post_max=0.1)

    dy = np.asarray(rcm_data.get("dy_pre", np.zeros(len(base_mask))))
    dz = np.asarray(rcm_data.get("dz_pre", np.zeros(len(base_mask))))
    offset_cm = np.sqrt(dy ** 2 + dz ** 2) * 100.0

    models = [
        ("Nakashima et al.", "default"),
        ("Nakashima et al. (refined)", "nakashima_refined"),
        ("Dürr et al.", "onnx_0426"),
        ("Proposed", "onnx_alex_refined"),
    ]
    vel_comps = ["vx", "vy", "vz"]
    spin_comps = ["wx", "wy", "wz"]

    indices = np.where(base_mask)[0]
    rows = []
    for idx_pos, idx in enumerate(indices):
        row = {"index": int(idx), "offset_cm": float(offset_cm[idx])}
        # Observed post values
        for comp in vel_comps + spin_comps:
            key = f"{comp}_post"
            row[f"obs_{comp}"] = float(rcm_data[key][idx]) if key in rcm_data else float("nan")
        # Pre values (for reference)
        for comp in vel_comps + spin_comps:
            key = f"{comp}_pre"
            row[f"pre_{comp}"] = float(rcm_data[key][idx]) if key in rcm_data else float("nan")
        # Model post values
        for m_label, m_suffix in models:
            for comp in vel_comps + spin_comps:
                key = f"{comp}_post_{m_suffix}"
                col_name = f"{m_label}_{comp}"
                row[col_name] = float(rcm_data[key][idx]) if key in rcm_data else float("nan")
        # Confidence / RMSE / fitness for debugging
        row["confidence"] = float(rcm_data.get("confidence", [float("nan")] * len(base_mask))[idx])
        row["max_rmse_opt"] = float(rcm_data.get("max_rmse_opt", [float("nan")] * len(base_mask))[idx])
        row["fitness_post"] = float(rcm_data.get("fitness_post", [float("nan")] * len(base_mask))[idx])
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → CSV ({len(rows)} contacts): {output_path}")


def plot_rcm_boxplot(dp, mc):
    """
    Racket contact box plot: velocity and spin magnitudes,
    Nakashima vs Nakashima (refined) vs ONNX alex (refined), grouped by contact offset, signed error.
    """
    rcm_data = dp.extract_rcm_data(mc)

    # Filters (conf≥0.5, RMSE ≤ 20mm, fitness_post ≤ 0.1)
    base_mask = _build_mask(rcm_data, conf_min=0.5, rmse_max=0.020,
                            fitness_post_max=0.1)

    # Contact offset bands
    dy = np.asarray(rcm_data.get("dy_pre", np.zeros(len(base_mask))))
    dz = np.asarray(rcm_data.get("dz_pre", np.zeros(len(base_mask))))
    offset_cm = np.sqrt(dy ** 2 + dz ** 2) * 100.0  # m → cm
    #offset_bands = [(0, 3), (3, 5), (5, 8), (8, 100)]
    #band_labels = ["0–3 cm", "3–5 cm", "5–8 cm", "8+ cm"]
    offset_bands = [(0, 5), (5, 100)]
    band_labels = ["0–5 cm", "5+ cm"]

    models = [
        ("Nakashima et al.", "default"),
        ("Nakashima et al. (refined)", "nakashima_refined"),
        ("Dürr et al.", "onnx_0426"),
        ("Proposed", "onnx_alex_refined"),
    ]

    vel_comps = ["vx", "vy", "vz"]
    spin_comps = ["wx", "wy", "wz"]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Velocity magnitude error (m/s)", "Spin magnitude error (rad/s)"],
        vertical_spacing=0.18,
    )

    n_models = len(models)
    model_offsets = np.linspace(-0.25, 0.25, n_models)
    violin_width = 0.45 / n_models  # width per violin

    for row, (comps, unit) in enumerate([(vel_comps, "m/s"), (spin_comps, "rad/s")], start=1):
        is_spin = (row == 2)
        tick_positions = list(range(len(band_labels)))

        for band_idx, ((lo, hi), blabel) in enumerate(zip(offset_bands, band_labels)):
            band_mask = base_mask & (offset_cm >= lo) & (offset_cm < hi) & np.isfinite(offset_cm)
            if band_mask.sum() == 0:
                continue

            obs_mag = _compute_magnitude(rcm_data, band_mask, comps, suffix=None)

            for m_idx, (m_label, m_suffix) in enumerate(models):
                model_mag = _compute_magnitude(rcm_data, band_mask, comps, suffix=m_suffix)
                err = model_mag - obs_mag  # signed error
                err = _clip_iqr(err)

                x_pos = band_idx + model_offsets[m_idx]
                fig.add_trace(go.Violin(
                    y=err,
                    x=np.full(len(err), x_pos),
                    name=m_label,
                    marker_color=MODEL_COLORS.get(m_label, "gray"),
                    line_color=MODEL_COLORS.get(m_label, "gray"),
                    meanline_visible=True,
                    points=False,
                    scalemode="width" if is_spin else "count",
                    width=violin_width,
                    showlegend=(band_idx == 0 and row == 1),
                    legendgroup=m_label,
                ), row=row, col=1)

        fig.update_yaxes(title_text=f"Error ({unit})", row=row, col=1,
                         **{k: v for k, v in _pub_axis("").items()
                            if k not in ("title",)})
        fig.update_xaxes(tickvals=tick_positions, ticktext=band_labels,
                         row=row, col=1,
                         **{k: v for k, v in _pub_axis("").items()
                            if k not in ("title",)})

    fig.update_layout(
        **_pub_layout(
            height=SUBPLOT_HEIGHT * 2,
            width=PLOT_WIDTH,
        ),
    )

    # ── Terminal statistics ──────────────────────────────────────────────────
    print("\n  [RCM violin] Magnitude error statistics (all offset bands combined):")
    # Collect magnitude errors per model across all bands
    combined_mask = base_mask & np.isfinite(offset_cm)
    for row_idx, (comps, unit, kind) in enumerate(
        [(vel_comps, "m/s", "Velocity"), (spin_comps, "rad/s", "Spin")]
    ):
        obs_mag = _compute_magnitude(rcm_data, combined_mask, comps, suffix=None)
        model_abs_err = {}
        for m_label, m_suffix in models:
            model_mag = _compute_magnitude(rcm_data, combined_mask, comps, suffix=m_suffix)
            err = np.abs(model_mag - obs_mag)
            err = _clip_iqr(err)
            model_abs_err[m_label] = err

        print(f"\n    {kind} magnitude ({unit}):")
        if "Proposed" in model_abs_err:
            cur = model_abs_err["Proposed"]
            cur_stats = (np.percentile(cur, 25), np.median(cur), np.percentile(cur, 75))
            print(f"      Proposed: 25th={cur_stats[0]:.4f}, median={cur_stats[1]:.4f}, 75th={cur_stats[2]:.4f}")
            for baseline in ["Nakashima et al.", "Nakashima et al. (refined)", "Dürr et al."]:
                if baseline not in model_abs_err:
                    continue
                bl = model_abs_err[baseline]
                bl_stats = (np.percentile(bl, 25), np.median(bl), np.percentile(bl, 75))
                imp = tuple((bl_stats[i] - cur_stats[i]) / bl_stats[i] * 100 for i in range(3))
                print(f"      vs {baseline}: 25th {imp[0]:+.1f}%, median {imp[1]:+.1f}%, 75th {imp[2]:+.1f}%")

        # Nakashima refined vs Nakashima
        if "Nakashima et al. (refined)" in model_abs_err and "Nakashima et al." in model_abs_err:
            ref = model_abs_err["Nakashima et al. (refined)"]
            nak = model_abs_err["Nakashima et al."]
            ref_stats = (np.percentile(ref, 25), np.median(ref), np.percentile(ref, 75))
            nak_stats = (np.percentile(nak, 25), np.median(nak), np.percentile(nak, 75))
            imp = tuple((nak_stats[i] - ref_stats[i]) / nak_stats[i] * 100 for i in range(3))
            print(f"      Nakashima (refined) vs Nakashima: 25th {imp[0]:+.1f}%, median {imp[1]:+.1f}%, 75th {imp[2]:+.1f}%")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots")
    parser.add_argument("data_folder", nargs="?", type=str, default=None,
                        help="Path to HDF5 data folder")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to JSON manifest file (alternative to data_folder)")
    parser.add_argument("--output-dir", type=str, default="plots/publication",
                        help="Output directory for plots")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "svg", "png", "html"],
                        help="Output format")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive plots in browser")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export box plot data to CSV for comparison")
    args = parser.parse_args()

    if args.json is None and args.data_folder is None:
        parser.error("either data_folder or --json is required")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.json:
        json_path = pathlib.Path(args.json)
        print(f"Loading data from JSON manifest {json_path}...")
        dp, mc = load_data_from_json(json_path)
    else:
        data_folder = pathlib.Path(args.data_folder)
        print(f"Loading data from {data_folder}...")
        dp, mc = load_data(data_folder)
    print(f"Loaded {len(mc.matches)} matches.")

    plots = [
        ("drag_cd_vs_spin_8_16", lambda: plot_drag_vs_spin_ratio(dp, mc, 8, 16)),
        ("drag_cd_discrete_bands", lambda: plot_drag_discrete_velocity_bands(dp, mc)),
        ("drag_cd_vs_spin_low_vel", lambda: plot_drag_low_velocity(dp, mc)),
        ("magnus_cm_vs_spin_8_16", lambda: plot_magnus_vs_spin(dp, mc, 8, 16)),
        ("magnus_cm_discrete_bands", lambda: plot_magnus_discrete_velocity_bands(dp, mc)),
        ("aero_position_rmse_boxplot", lambda: plot_aero_boxplot(dp, mc)),
        ("table_contact_boxplot", lambda: plot_table_contact_boxplot(dp, mc)),
        ("rcm_contact_offset_boxplot", lambda: plot_rcm_boxplot(dp, mc)),
    ]

    for name, plot_fn in plots:
        print(f"Generating {name}...")
        fig = plot_fn()

        out_path = output_dir / f"{name}.{args.format}"
        if args.format == "html":
            fig.write_html(str(out_path))
        else:
            fig.write_image(str(out_path), scale=3)
        print(f"  → {out_path}")

        if args.interactive:
            fig.show()

    if args.export_csv:
        print("Exporting CSV data...")
        export_rcm_csv(dp, mc, output_dir / "rcm_boxplot_data.csv")

    print("Done.")


if __name__ == "__main__":
    main()
