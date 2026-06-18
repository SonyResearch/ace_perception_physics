"""Fuctions for transformations between coordinate spaces."""
# Confidential, Copyright 2024, Sony AI, All rights reserved.
# pylint: disable = invalid-name

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def obj2cam_transform(points, rvec, tvec):
    """
    Transform 3d points from object coords to camera coords.

    rvect and tvec represent the object pose with respect to the camera.
    """
    T_CO = np.eye(4, dtype=np.float64)
    T_CO[:3, :3] = Rot.from_rotvec(np.reshape(rvec, 3)).as_matrix()
    T_CO[:3, 3] = np.reshape(tvec, 3)

    O_p = np.transpose(points)
    O_p = np.vstack([O_p, np.ones((1, O_p.shape[1]))])
    C_p = T_CO.dot(O_p)
    return C_p[:3].T


def obj2world_transform(points, rvec, tvec):
    """
    Transform 3d points from object coords to world coords.

    rvect and tvec represent the object pose with respect to the world.
    """
    return obj2cam_transform(points, rvec, tvec)
