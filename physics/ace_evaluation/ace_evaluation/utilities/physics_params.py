# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Default physics parameters for table-tennis aerodynamics and contact models.

This module is self-contained (no proprietary dependencies) and provides the
canonical ``get_params()`` factory that was previously in ``utilities.py``.
"""

from typing import Any, Dict


def get_params(
    drag_coeff: float = 0.55,
    magnus_coeff: float = -0.001,
    magnus_coeff_linear: float = 0.1,
    magnus_coeff_max: float = 0.4,
    coeff_restitution_table: float = 0.98,
    coeff_restitution_table_zvelocity_scale: float = 0.02,
    coeff_dynamic_friction_table: float = 0.25,
    coeff_elasticity_factor_racket: float = 1.928e-3,
    use_piecewise_restitution_racket: bool = True,
    m1_piecewise_racket: float = -0.555,
    b1_piecewise_racket: float = 1.241,
    v_t_piecewise_racket: float = -3.031,
    use_compact_residual_model: bool = True,
    compact_residual_model_name: str = "residual_model_compact.exponential.dim8.pre1002_wo0428.RCM-118.onnx",
    test_new_model: bool = True,
) -> Dict[str, Any]:
    """Return a physics-parameters dict with sensible defaults."""
    return {
        "air_density": 1.204,
        "ball_density": 81.2,
        "ball_radius": 0.02,
        "gravity": -9.81,
        "drag_coeff": drag_coeff,
        "drag_coeff_linear": 0.0,
        "magnus_coeff": magnus_coeff,
        "magnus_coeff_linear": magnus_coeff_linear,
        "magnus_coeff_max": magnus_coeff_max,
        "coeff_restitution_table": coeff_restitution_table,
        "coeff_restitution_table_zvelocity_scale": coeff_restitution_table_zvelocity_scale,
        "coeff_dynamic_friction_table": coeff_dynamic_friction_table,
        "coeff_elasticity_factor_racket": coeff_elasticity_factor_racket,
        "use_piecewise_restitution_racket": use_piecewise_restitution_racket,
        "m1_piecewise_racket": m1_piecewise_racket,
        "b1_piecewise_racket": b1_piecewise_racket,
        "v_t_piecewise_racket": v_t_piecewise_racket,
        "use_compact_residual_model": use_compact_residual_model,
        "compact_residual_model_name": compact_residual_model_name,
        "initial_state_cov": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0],
        "free_flight_process_noise_cov_rate": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "rebound_process_noise_cov_rate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "obs_noise_cov": [0.0001, 0.0001, 0.0001, 1.0, 1.0, 1.0],
        "enable_rk4": True,
        "coeff_restitution_racket": 0.74,
        "coeff_restitution_ground": 0.5,
        "coeff_friction_table": 1.0,
        "coeff_friction_racket": 1.0,
        "coeff_restitution_net": 0.05,
        "test_new_model": test_new_model,
    }
