# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.

import os
from pathlib import Path
from typing import Any, Tuple, Dict

import numpy as np
import torch
import yaml
from omegaconf import DictConfig

# input_config = {
#     "bv":{"input_dim": 3, "layer_dim": 8},   # ball relative velocity (racket frame)
#     "bs":{"input_dim": 3, "layer_dim": 8},   # ball spin (racket frame)
#     "dist":{"input_dim": 2, "layer_dim": 2}, # dyz (racket frame)
#     "rv":{"input_dim": 3, "layer_dim": 8},  # racket velocity (racket frame)
#     "rs":{"input_dim": 3, "layer_dim": 8},  # racket spin (racket frame)
#     "rp":{"input_dim": 3, "layer_dim": 3},  # racket position (global)
#     "rq":{"input_dim": 4, "layer_dim": 4},  # racket frame in quternions  (global)
#     }

def get_params(
) -> Dict[str, Any]:
    """method to return parameters object"""
    parameters = {
        "ball_density": 81.2,
        "ball_radius": 0.02,
        "coeff_elasticity_factor_racket": 1.928e-3,
        "coeff_restitution_racket": 0.79,
        "coeff_restitution_net": 0.05,
    }
    return parameters


def get_nakashima_matrix(physics):
    """
    Compute the combined matrix of naskashima as in physics_layer based on the instance attributes.

    :return: The combined matrix.
    """
    ball_radius = physics["ball_radius"]
    coeff_restitution_racket = physics["coeff_restitution_racket"]
    ball_density = physics["ball_density"]
    mass = (4.0 / 3.0) * np.pi * ball_radius**3 * ball_density
    kp_rcm = physics["coeff_elasticity_factor_racket"] / mass

    a_u = np.array([[-coeff_restitution_racket, 0, 0], [0, 1 - kp_rcm, 0], [0, 0, 1 - kp_rcm]])

    b_u = np.array([[0, 0, 0], [0, 0, kp_rcm * ball_radius], [0, -kp_rcm * ball_radius, 0]])

    a_o = np.array([[0, 0, 0], [0, 0, -1.5 * kp_rcm / ball_radius], [0, 1.5 * kp_rcm / ball_radius, 0]])

    b_o = np.array([[1, 0, 0], [0, 1 - 1.5 * kp_rcm, 0], [0, 0, 1 - 1.5 * kp_rcm]])

    # Combine the matrices into a single large matrix
    combined_matrix = np.block([[a_u, b_u], [a_o, b_o]])

    return combined_matrix


def get_dynamic_nakashima_res(nakashima_mat, model_input, method="default"):
    """
    method: 'tangential_COR' or 'default'
    """

    if method == "tangential_COR":
        r = 0.02
        e0 = 0.914327
        e1 = -0.022615
        alpha = 0.0
        et0 = 0.792148
        et1 = -0.007394
        c_spin = 0.276443

        batch_size = model_input.shape[0]

        vn = model_input[:, 0]
        u1 = model_input[:, 1] - r * model_input[:, 5]
        u2 = model_input[:, 2] + r * model_input[:, 4]
        u_mag = torch.sqrt(u1**2 + u2**2 + 1e-12)
        et = torch.clamp(et0 + et1 * u_mag, 0.0, 1.0)
        kp = 0.4 * (1 + et)

        # Build batched [batch_size, 6, 6] matrix from the base nakashima_mat
        batched_mat = nakashima_mat.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # a_u block (rows 0-2, cols 0-2): diagonal
        batched_mat[:, 0, 0] = -torch.clamp(e0 + e1 * torch.abs(vn), 0.0, 1.0)  # negative COR
        batched_mat[:, 1, 1] = 1 - kp + alpha * vn
        batched_mat[:, 2, 2] = 1 - kp + alpha * vn

        # b_u block (rows 0-2, cols 3-5): skew-symmetric in yz
        batched_mat[:, 1, 5] = kp * r  # w2 -> v1
        batched_mat[:, 2, 4] = -kp * r  # w1 -> v2

        # a_o block (rows 3-5, cols 0-2): skew-symmetric in yz
        batched_mat[:, 4, 2] = -3 / 2 * kp / r  # v2 -> w1
        batched_mat[:, 5, 1] = 3 / 2 * kp / r  # v1 -> w2

        # b_o block (rows 3-5, cols 3-5): diagonal
        batched_mat[:, 3, 3] = 1 - c_spin
        batched_mat[:, 4, 4] = 1 - 3 / 2 * kp
        batched_mat[:, 5, 5] = 1 - 3 / 2 * kp

        # Batched matmul: [B, 6, 6] @ [B, 6, 1] -> [B, 6, 1] -> [B, 6]
        nakashima_output = torch.bmm(batched_mat, model_input.unsqueeze(-1)).squeeze(-1)

        return nakashima_output

    elif method == "default":
        nakashima_output = (nakashima_mat @ model_input.T).T
        return nakashima_output

    else:
        raise ValueError


def read_params_from_yaml(yaml_path: Path) -> Tuple[float, float, float, float, float]:
    """
    Loads the parameters from the yaml file in this format:
    params: Tuple[float]
        - contact_time
        - normal stiffness - ball_stiffness
        - normal damping
        - transversal stiffness
        - transversal damping
    """
    if not os.path.exists(yaml_path):
        return (1e12,) * 5  # Return a high error value if the file doesn't exist

    with open(yaml_path, mode="r", encoding="utf-8") as file:
        try:
            data_tmp = yaml.safe_load(file)
            data = data_tmp.get("coefficients", (1e12,) * 5)
            data_list = (
                data["contact_time"],
                data["stiffness_normal - stiffness_ball"],
                data["normal_damping"],
                data["stiffness_trans"],
                data["trans_damping"],
            )
            return data_list
        except yaml.YAMLError as error:
            print(f"YAML error: {error}")
            return (1e12,) * 5


def check_params(cfg: DictConfig):
    """Check the parameters of the config file"""
    assert cfg.neural_network.output_dim == 3, "Output dimension should be 3"


############# cluster ######
def torch_calc_dist_cluster(samples, cluster_centers):

    res = torch.cdist(samples, cluster_centers, p=2)
    test_min_dist = torch.min(res, 1)[0]  # the tuple of two output tensors (min, min_indices)
    return test_min_dist


def torch_dist_att_func(x, type=2):
    if type == 1:
        # ### normal ###
        a = 118.060
        b = 59.756
        c = 0.009
    elif type == 2:
        # little effect ###
        a = 1.948
        b = 17.358
        c = -0.087
    elif type == 3:
        # # ### exaggerated ###
        a = 3.906
        b = 4.463
        c = -1.500
    elif type == 4:
        # even exaggerated ###
        a = 7.282
        b = 12.899
        c = -0.052
    else:
        raise ValueError
    return a * torch.exp(-b * x) + c
