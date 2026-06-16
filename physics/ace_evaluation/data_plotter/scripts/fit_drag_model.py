# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Drag Coefficient Model Fitting

Fits a smooth 2D model for C_D(v_eff_drag, S) using radial basis functions
with regularization for smooth extrapolation.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Any, Dict, Tuple, Optional
import pickle


class DragCoefficientModel:
    """
    Smooth 2D model for drag coefficient C_D(v, S).

    Uses Radial Basis Function interpolation with:
    - Data binning to reduce noise and computation
    - Regularization for smooth behavior
    - Constrained extrapolation for sparse regions

    Domain:
    - v (v_eff_drag): [0, 30] m/s
    - S (spin_ratio = r*ω/v): [0, 2]
    """

    def __init__(self,
                 kernel: str = 'thin_plate_spline',
                 smoothing: float = 0.1,
                 degree: int = 1):
        """
        Args:
            kernel: RBF kernel type ('thin_plate_spline', 'cubic', 'quintic', 'multiquadric')
            smoothing: Smoothing parameter (higher = smoother, less fit to data)
            degree: Degree of polynomial added to RBF (0, 1, or 2)
        """
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self.rbf: Optional[RBFInterpolator] = None
        self.v_scale = 30.0  # Normalization scale for velocity
        self.S_scale = 2.0   # Normalization scale for spin ratio

        # Domain bounds
        self.v_min, self.v_max = 0.0, 30.0
        self.S_min, self.S_max = 0.0, 2.0

        # Store fit statistics
        self.fit_stats: Dict[str, Any] = {}

    def _preprocess_data(self, v: np.ndarray, S: np.ndarray, cd: np.ndarray,
                        n_v_bins: int = 30, n_S_bins: int = 20,
                        min_points_per_bin: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data by binning to reduce noise and computational load.

        Returns:
            v_binned, S_binned, cd_binned, weights (inverse variance weights)
        """
        # Create bins
        v_edges = np.linspace(self.v_min, self.v_max, n_v_bins + 1)
        S_edges = np.linspace(self.S_min, self.S_max, n_S_bins + 1)

        v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
        S_centers = 0.5 * (S_edges[:-1] + S_edges[1:])

        # Digitize data
        v_bin_idx = np.digitize(v, v_edges) - 1
        S_bin_idx = np.digitize(S, S_edges) - 1

        # Clamp to valid range
        v_bin_idx = np.clip(v_bin_idx, 0, n_v_bins - 1)
        S_bin_idx = np.clip(S_bin_idx, 0, n_S_bins - 1)

        # Compute statistics per bin
        v_binned: list[float] = []
        S_binned: list[float] = []
        cd_binned: list[float] = []
        weights: list[float] = []

        for i in range(n_v_bins):
            for j in range(n_S_bins):
                mask = (v_bin_idx == i) & (S_bin_idx == j)
                n_points: int = int(np.sum(mask))

                if n_points >= min_points_per_bin:
                    cd_vals = cd[mask]
                    mean_cd = float(np.mean(cd_vals))
                    std_cd = float(np.std(cd_vals))

                    # Use bin center
                    v_binned.append(float(v_centers[i]))
                    S_binned.append(float(S_centers[j]))
                    cd_binned.append(mean_cd)

                    # Weight by inverse variance (clamped to avoid division issues)
                    std_cd = max(std_cd, 0.01)
                    weights.append(float(n_points) / (std_cd ** 2))

        v_binned_arr = np.array(v_binned)
        S_binned_arr = np.array(S_binned)
        cd_binned_arr = np.array(cd_binned)
        weights_arr = np.array(weights)

        # Normalize weights
        weights_arr = weights_arr / np.mean(weights_arr)

        return v_binned_arr, S_binned_arr, cd_binned_arr, weights_arr

    def _add_boundary_constraints(self, v: np.ndarray, S: np.ndarray, cd: np.ndarray,
                                  weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Add synthetic boundary points to constrain extrapolation behavior.

        Physics-based constraints:
        - At v→0: CD should be constant at 0.55 for all S
        - At S=0 (no spin): CD varies with velocity
        - At very high S: CD tends toward constant behavior
        """
        # Estimate baseline CD at S=0 from data (for higher velocities)
        mask_low_S = S < 0.1
        if np.sum(mask_low_S) > 10:
            cd_baseline: float = float(np.median(cd[mask_low_S]))
        else:
            cd_baseline = 0.50

        # Add boundary points
        v_boundary: list[float] = []
        S_boundary: list[float] = []
        cd_boundary: list[float] = []
        w_boundary = []

        # v→0 constraint: CD = 0.55 constant for all S values up to v=4
        # Add strong constraints at low velocity
        cd_at_zero_velocity = 0.55
        for S_val in np.linspace(0, 2.0, 21):
            # At v=0
            v_boundary.append(0.0)
            S_boundary.append(S_val)
            cd_boundary.append(cd_at_zero_velocity)
            w_boundary.append(1.0)  # Strong weight for this constraint

            # At v=1
            v_boundary.append(1.0)
            S_boundary.append(S_val)
            cd_boundary.append(cd_at_zero_velocity)
            w_boundary.append(1.0)

            # At v=2
            v_boundary.append(2.0)
            S_boundary.append(S_val)
            cd_boundary.append(cd_at_zero_velocity)
            w_boundary.append(1.0)

            # At v=3
            v_boundary.append(3.0)
            S_boundary.append(S_val)
            cd_boundary.append(cd_at_zero_velocity)
            w_boundary.append(1.0)

            # At v=4 (end of constant region)
            v_boundary.append(4.0)
            S_boundary.append(S_val)
            cd_boundary.append(cd_at_zero_velocity)
            w_boundary.append(0.8)  # Slightly lower to allow transition

        # S=0 line at higher velocities (no spin - use data-derived baseline)
        for v_val in np.linspace(5, 30, 10):
            v_boundary.append(v_val)
            S_boundary.append(0.0)
            cd_boundary.append(cd_baseline)
            w_boundary.append(0.3)  # Lower weight for constraints

        # High S boundary (constant behavior)
        for v_val in np.linspace(5, 25, 10):
            # Estimate from data at similar v with high S
            mask = (v > v_val - 2) & (v < v_val + 2) & (S > 1.0)
            if np.sum(mask) > 3:
                cd_high_S: float = float(np.median(cd[mask]))
            else:
                cd_high_S = 0.50

            v_boundary.append(v_val)
            S_boundary.append(2.0)
            cd_boundary.append(cd_high_S)
            w_boundary.append(0.2)

        # Combine with original data
        v_all = np.concatenate([v, np.array(v_boundary)])
        S_all = np.concatenate([S, np.array(S_boundary)])
        cd_all = np.concatenate([cd, np.array(cd_boundary)])
        weights_all = np.concatenate([weights, np.array(w_boundary)])

        return v_all, S_all, cd_all, weights_all

    def fit(self, v: np.ndarray, S: np.ndarray, cd: np.ndarray,
            add_boundaries: bool = True,
            n_v_bins: int = 30, n_S_bins: int = 20) -> 'DragCoefficientModel':
        """
        Fit the model to data.

        Args:
            v: velocity (v_eff_drag) array
            S: spin ratio array
            cd: drag coefficient array
            add_boundaries: Whether to add boundary constraints
            n_v_bins: Number of velocity bins for preprocessing
            n_S_bins: Number of spin ratio bins for preprocessing

        Returns:
            self for chaining
        """
        # Filter valid data
        valid = np.isfinite(v) & np.isfinite(S) & np.isfinite(cd)
        valid &= (v >= self.v_min) & (v <= self.v_max)
        valid &= (S >= self.S_min) & (S <= self.S_max)
        valid &= (cd > 0) & (cd < 2)  # Physical bounds on CD

        v_clean = v[valid]
        S_clean = S[valid]
        cd_clean = cd[valid]

        print(f"Valid data points: {len(v_clean)} / {len(v)}")

        # Preprocess by binning
        v_binned, S_binned, cd_binned, weights = self._preprocess_data(
            v_clean, S_clean, cd_clean, n_v_bins, n_S_bins
        )
        print(f"Binned data points: {len(v_binned)}")

        # Add boundary constraints if requested
        if add_boundaries:
            v_binned, S_binned, cd_binned, weights = self._add_boundary_constraints(
                v_binned, S_binned, cd_binned, weights
            )
            print(f"Data points with boundaries: {len(v_binned)}")

        # Normalize coordinates
        v_norm = v_binned / self.v_scale
        S_norm = S_binned / self.S_scale

        # Stack into coordinate array
        coords = np.column_stack([v_norm, S_norm])

        # Fit RBF interpolator
        self.rbf = RBFInterpolator(
            coords, cd_binned,
            kernel=self.kernel,
            smoothing=self.smoothing,
            degree=self.degree
        )

        # Store fit statistics
        cd_pred = self.rbf(coords)
        residuals = cd_binned - cd_pred
        self.fit_stats = {
            'n_raw_points': len(v),
            'n_valid_points': len(v_clean),
            'n_binned_points': len(v_binned),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'r2': 1 - np.var(residuals) / np.var(cd_binned),
        }

        print(f"Fit statistics: RMSE={self.fit_stats['rmse']:.4f}, R²={self.fit_stats['r2']:.4f}")

        return self

    def __call__(self, v: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Evaluate the model at given (v, S) points.

        Args:
            v: velocity (v_eff_drag) - scalar or array
            S: spin ratio - scalar or array

        Returns:
            cd: drag coefficient - same shape as input
        """
        if self.rbf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        v_arr = np.atleast_1d(np.asarray(v, dtype=float))
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))

        # Handle broadcasting
        v_arr, S_arr = np.broadcast_arrays(v_arr, S_arr)
        original_shape = v_arr.shape

        # Flatten for evaluation
        v_flat = v_arr.ravel()
        S_flat = S_arr.ravel()

        # Clamp to valid domain
        v_clamped = np.clip(v_flat, self.v_min, self.v_max)
        S_clamped = np.clip(S_flat, self.S_min, self.S_max)

        # Normalize
        v_norm = v_clamped / self.v_scale
        S_norm = S_clamped / self.S_scale

        # Evaluate
        coords = np.column_stack([v_norm, S_norm])
        cd = self.rbf(coords)

        # Clamp output to physical range
        cd = np.clip(cd, 0.2, 1.0)

        # Reshape to original shape
        cd = cd.reshape(original_shape)

        # Return scalar if input was scalar
        if cd.size == 1 and np.ndim(v) == 0 and np.ndim(S) == 0:
            return np.array(float(cd.ravel()[0]))

        return cd

    def save(self, filepath: str):
        """Save the fitted model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kernel': self.kernel,
                'smoothing': self.smoothing,
                'degree': self.degree,
                'v_scale': self.v_scale,
                'S_scale': self.S_scale,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'S_min': self.S_min,
                'S_max': self.S_max,
                'fit_stats': self.fit_stats,
                'rbf': self.rbf,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'DragCoefficientModel':
        """Load a fitted model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            kernel=data['kernel'],
            smoothing=data['smoothing'],
            degree=data['degree']
        )
        model.v_scale = data['v_scale']
        model.S_scale = data['S_scale']
        model.v_min = data['v_min']
        model.v_max = data['v_max']
        model.S_min = data['S_min']
        model.S_max = data['S_max']
        model.fit_stats = data['fit_stats']
        model.rbf = data['rbf']

        return model

    def export_as_python(self, filepath: str,
                         n_v_points: int = 31,
                         n_S_points: int = 21,
                         function_name: str = 'get_drag_estimate_fitted') -> str:
        """
        Export the fitted model as a human-readable Python function.

        The exported function uses bilinear interpolation over a precomputed
        lookup table, making it fast, portable, and easy to understand/modify.

        Args:
            filepath: Path to save the Python file
            n_v_points: Number of velocity grid points (default 31 for 1 m/s resolution)
            n_S_points: Number of spin ratio grid points (default 21 for 0.1 resolution)
            function_name: Name of the exported function

        Returns:
            The generated Python code as a string
        """
        if self.rbf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Create evaluation grid
        v_grid = np.linspace(self.v_min, self.v_max, n_v_points)
        S_grid = np.linspace(self.S_min, self.S_max, n_S_points)

        # Evaluate model on grid
        V, S = np.meshgrid(v_grid, S_grid, indexing='ij')
        CD_grid = self(V, S)

        # Format arrays as Python code
        def format_array_1d(arr, name, indent=4):
            """Format a 1D array as readable Python code."""
            indent_str = ' ' * indent
            indent_inner = ' ' * (indent + 4)
            lines = [f"{indent_str}{name} = np.array(["]
            # Split into rows of ~8 values
            values_per_row = 8
            for i in range(0, len(arr), values_per_row):
                chunk = arr[i:i+values_per_row]
                row = ", ".join(f"{v:.6f}" for v in chunk)
                if i + values_per_row < len(arr):
                    row += ","
                lines.append(f"{indent_inner}{row}")
            lines.append(indent_str + "])")
            return "\n".join(lines)

        def format_array_2d(arr, name, indent=4):
            """Format a 2D array as readable Python code."""
            indent_str = ' ' * indent
            indent_inner = ' ' * (indent + 4)
            lines = [f"{indent_str}{name} = np.array(["]
            for i, row in enumerate(arr):
                row_str = "[" + ", ".join(f"{v:.6f}" for v in row) + "]"
                if i < len(arr) - 1:
                    row_str += ","
                lines.append(f"{indent_inner}{row_str}")
            lines.append(indent_str + "])")
            return "\n".join(lines)

        # Pre-compute statistics for string formatting
        n_raw = self.fit_stats.get('n_raw_points', 'N/A')
        n_valid = self.fit_stats.get('n_valid_points', 'N/A')
        rmse = self.fit_stats.get('rmse', None)
        r2 = self.fit_stats.get('r2', None)
        rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
        r2_str = f"{r2:.6f}" if r2 is not None else "N/A"
        v_spacing = (self.v_max - self.v_min) / (n_v_points - 1)
        S_spacing = (self.S_max - self.S_min) / (n_S_points - 1)

        # Build code using list of lines to avoid f-string escaping issues
        lines = []
        lines.append('# SPDX-License-Identifier: BSD-3-Clause')
        lines.append('# Copyright (c) 2024-2026, Sony AI Inc.')
        lines.append('# fmt: off')
        lines.append('# pylint: skip-file')
        lines.append('"""')
        lines.append('Fitted Drag Coefficient Model')
        lines.append('')
        lines.append('Auto-generated from DragCoefficientModel.export_as_python()')
        lines.append('')
        lines.append('Model Statistics:')
        lines.append(f'- Raw data points: {n_raw}')
        lines.append(f'- Valid data points: {n_valid}')
        lines.append(f'- RMSE: {rmse_str}')
        lines.append(f'- R²: {r2_str}')
        lines.append('')
        lines.append('Domain:')
        lines.append(f'- v_eff_drag: [{self.v_min}, {self.v_max}] m/s')
        lines.append(f'- spin_ratio S: [{self.S_min}, {self.S_max}]')
        lines.append('')
        lines.append('Grid Resolution:')
        lines.append(f'- v: {n_v_points} points ({v_spacing:.2f} m/s spacing)')
        lines.append(f'- S: {n_S_points} points ({S_spacing:.3f} spacing)')
        lines.append('"""')
        lines.append('')
        lines.append('import numpy as np')
        lines.append('')
        lines.append('')
        lines.append('# Lookup table grid points')
        lines.append(format_array_1d(v_grid, '_V_GRID', indent=0))
        lines.append('')
        lines.append(format_array_1d(S_grid, '_S_GRID', indent=0))
        lines.append('')
        lines.append(f'# Precomputed C_D values: shape ({n_v_points}, {n_S_points})')
        lines.append('# _CD_TABLE[i, j] = C_D(v=_V_GRID[i], S=_S_GRID[j])')
        lines.append(format_array_2d(CD_grid, '_CD_TABLE', indent=0))
        lines.append('')
        lines.append('')
        lines.append(f'def {function_name}(v, S):')
        lines.append('    """')
        lines.append('    Estimate drag coefficient C_D as a function of velocity and spin ratio.')
        lines.append('    ')
        lines.append('    Uses bilinear interpolation over a precomputed lookup table.')
        lines.append('    Values outside the domain are clamped to the boundary.')
        lines.append('    ')
        lines.append('    Args:')
        lines.append('        v: velocity (v_eff_drag) in m/s (scalar or array)')
        lines.append('        S: spin ratio S = r*w/v (scalar or array)')
        lines.append('        ')
        lines.append('    Returns:')
        lines.append('        C_D: drag coefficient (scalar or array matching input shape)')
        lines.append('    """')
        lines.append('    # Handle input types')
        lines.append('    v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))')
        lines.append('    S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))')
        lines.append('    ')
        lines.append('    # Track if input was scalar')
        lines.append('    scalar_input = (np.ndim(v) == 0) and (np.ndim(S) == 0)')
        lines.append('    ')
        lines.append('    # Broadcast arrays')
        lines.append('    v_arr, S_arr = np.broadcast_arrays(v_arr, S_arr)')
        lines.append('    original_shape = v_arr.shape')
        lines.append('    v_flat = v_arr.ravel()')
        lines.append('    S_flat = S_arr.ravel()')
        lines.append('    ')
        lines.append('    # Clamp to valid domain')
        lines.append('    v_clamped = np.clip(v_flat, _V_GRID[0], _V_GRID[-1])')
        lines.append('    S_clamped = np.clip(S_flat, _S_GRID[0], _S_GRID[-1])')
        lines.append('    ')
        lines.append('    # Find grid indices and interpolation weights')
        lines.append('    # For v')
        lines.append('    v_idx_float = (v_clamped - _V_GRID[0]) / (_V_GRID[-1] - _V_GRID[0]) * (len(_V_GRID) - 1)')
        lines.append('    v_idx_lo = np.clip(np.floor(v_idx_float).astype(int), 0, len(_V_GRID) - 2)')
        lines.append('    v_idx_hi = v_idx_lo + 1')
        lines.append('    v_frac = v_idx_float - v_idx_lo')
        lines.append('    ')
        lines.append('    # For S')
        lines.append('    S_idx_float = (S_clamped - _S_GRID[0]) / (_S_GRID[-1] - _S_GRID[0]) * (len(_S_GRID) - 1)')
        lines.append('    S_idx_lo = np.clip(np.floor(S_idx_float).astype(int), 0, len(_S_GRID) - 2)')
        lines.append('    S_idx_hi = S_idx_lo + 1')
        lines.append('    S_frac = S_idx_float - S_idx_lo')
        lines.append('    ')
        lines.append('    # Bilinear interpolation')
        lines.append('    # C_D = (1-tv)*(1-ts)*C00 + tv*(1-ts)*C10 + (1-tv)*ts*C01 + tv*ts*C11')
        lines.append('    C00 = _CD_TABLE[v_idx_lo, S_idx_lo]')
        lines.append('    C10 = _CD_TABLE[v_idx_hi, S_idx_lo]')
        lines.append('    C01 = _CD_TABLE[v_idx_lo, S_idx_hi]')
        lines.append('    C11 = _CD_TABLE[v_idx_hi, S_idx_hi]')
        lines.append('    ')
        lines.append('    cd = ((1 - v_frac) * (1 - S_frac) * C00 +')
        lines.append('          v_frac * (1 - S_frac) * C10 +')
        lines.append('          (1 - v_frac) * S_frac * C01 +')
        lines.append('          v_frac * S_frac * C11)')
        lines.append('    ')
        lines.append('    # Reshape to original shape')
        lines.append('    cd = cd.reshape(original_shape)')
        lines.append('    ')
        lines.append('    # Return scalar if input was scalar')
        lines.append('    if scalar_input:')
        lines.append('        return float(cd.ravel()[0])')
        lines.append('    return cd')
        lines.append('')
        lines.append('')
        lines.append('# Convenience alias')
        lines.append(f'get_drag_estimate = {function_name}')
        lines.append('')

        code = '\n'.join(lines)

        # Write to file
        with open(filepath, 'w') as f:
            f.write(code)

        print(f"Exported model to {filepath}")
        print(f"  Grid: {n_v_points} x {n_S_points} = {n_v_points * n_S_points} lookup values")
        print(f"  Function name: {function_name}")

        return code


def fit_drag_model_from_data(data: dict,
                             smoothing: float = 0.1,
                             kernel: str = 'thin_plate_spline') -> DragCoefficientModel:
    """
    Convenience function to fit a drag model from extracted data dictionary.

    Args:
        data: Dictionary with 'v_eff_drag', 'spin_ratio', 'cd_opt' arrays
        smoothing: RBF smoothing parameter
        kernel: RBF kernel type

    Returns:
        Fitted DragCoefficientModel
    """
    model = DragCoefficientModel(kernel=kernel, smoothing=smoothing)
    model.fit(data['v_eff_drag'], data['spin_ratio'], data['cd_opt'])
    return model


def visualize_model(model: DragCoefficientModel, data: dict = None):
    """
    Create visualization of the fitted model.

    Args:
        model: Fitted DragCoefficientModel
        data: Optional data dictionary for comparison scatter plot
    """
    import matplotlib.pyplot as plt

    # Create evaluation grid
    v_grid = np.linspace(1, 30, 60)
    S_grid = np.linspace(0, 1.5, 50)
    V, S = np.meshgrid(v_grid, S_grid)
    CD = model(V, S)

    fig = plt.figure(figsize=(16, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(V, S, CD, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('v_eff_drag (m/s)')
    ax1.set_ylabel('Spin Ratio S')
    ax1.set_zlabel('C_D')
    ax1.set_title('Fitted C_D(v, S) Surface')
    ax1.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='C_D')

    # Contour plot
    ax2 = fig.add_subplot(132)
    cs = ax2.contourf(V, S, CD, levels=20, cmap='viridis')
    ax2.set_xlabel('v_eff_drag (m/s)')
    ax2.set_ylabel('Spin Ratio S')
    ax2.set_title('C_D Contour Plot')
    fig.colorbar(cs, ax=ax2, label='C_D')

    # Add data points if provided
    if data is not None:
        v_data = data['v_eff_drag']
        S_data = data['spin_ratio']
        mask = (S_data >= 0) & (S_data <= 1.5) & (v_data >= 1) & (v_data <= 30)
        ax2.scatter(v_data[mask], S_data[mask], c='red', s=1, alpha=0.2, label='Data')

    # Slices at different velocities
    ax3 = fig.add_subplot(133)
    v_slices = [5, 10, 15, 20, 25]
    S_line = np.linspace(0, 1.5, 100)

    for v_val in v_slices:
        cd_line = model(np.full_like(S_line, v_val), S_line)
        ax3.plot(S_line, cd_line, label=f'v={v_val} m/s', linewidth=2)

    ax3.set_xlabel('Spin Ratio S')
    ax3.set_ylabel('C_D')
    ax3.set_title('C_D vs S at Different Velocities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.5)
    ax3.set_ylim(0.3, 0.7)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from data_plotter.scripts.analyze_drag_data import load_data_for_fitting

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--min_duration', type=float, default=0.15)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='drag_model.pkl')
    args = parser.parse_args()

    # Load data
    data = load_data_for_fitting(args.data_path, args.min_confidence, args.min_duration)

    # Fit model
    model = fit_drag_model_from_data(data, smoothing=args.smoothing)

    # Save model (pickle)
    model.save(args.output)
    print(f"Saved model to {args.output}")

    # Export as human-readable Python
    py_output = args.output.replace('.pkl', '_function.py')
    model.export_as_python(py_output)

    # Visualize
    import matplotlib.pyplot as plt
    fig = visualize_model(model, data)
    fig.savefig('drag_model_visualization.png', dpi=150)
    print("Saved visualization to drag_model_visualization.png")
    plt.show()
