"""
Triangulation functions written in numba for maximum performance.

A wrapper Triangulator class is also provided for convenience.
"""
# pylint: disable = line-too-long, invalid-name, no-member, too-many-locals, redefined-outer-name
# Confidential, Copyright 2024, Sony AI, All rights reserved.

from time import time
import filecmp
import shutil
import os
import cv2
import numpy as np
from numba import njit, prange
import ace_yaml as yaml
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration


###########################
# Constant parameters
###########################


class TriangulationNumba:
    """Triangulation helper class"""

    def __init__(self, config_name):  # pylint: disable =too-many-statements, too-many-branches
        self.config_name = config_name

        self.camera_calibration = calibration.CameraCalibrationParameters()
        self.calibration_params_path = os.path.join(
            get_package_share_directory("calibration"),
            "parameters",
            "camera_calibration",
            self.config_name + ".yaml",
        )
        if not self.camera_calibration.initialize(self.calibration_params_path):
            raise Exception("Failed to load configuration file")

        self.num_cameras = len(self.camera_calibration.cameras)
        self.camera_names = [None] * self.num_cameras
        self.cameras = [None] * self.num_cameras

        for i, camera in enumerate(self.camera_calibration.cameras):
            self.camera_names[i] = camera
            self.cameras[i] = self.camera_calibration.cameras[camera]

        with open(self.calibration_params_path, "r", encoding="UTF-8") as f:
            self.calibration_params_yaml = yaml.full_load(f)
        f.close()

        self.camera_resolution = self.calibration_params_yaml[self.camera_names[0]]["resolution"]

        # Triangulation
        self.triangulation_params = {}
        self.triangulation_params["Ks"] = np.array(
            [self.calibration_params_yaml[cam]["camera_matrix"] for cam in self.camera_names]
        )
        self.triangulation_params["Ks_inv"] = np.array([np.linalg.inv(K) for K in self.triangulation_params["Ks"]])
        self.T_WO = np.array(self.calibration_params_yaml["T_WO"])
        self.triangulation_params["T_Ws"] = np.array(
            [np.dot(self.calibration_params_yaml[cam]["T_CW"], self.T_WO) for cam in self.camera_names]
        )
        self.triangulation_params["T_Ws_inv"] = np.array(
            [np.linalg.inv(T_W) for T_W in self.triangulation_params["T_Ws"]]
        )
        self.triangulation_params["Ps"] = np.array(
            [
                np.dot(K, T_W[:3, :])
                for K, T_W in zip(self.triangulation_params["Ks"], self.triangulation_params["T_Ws"])
            ]
        )
        self.camera_distortions = [self.calibration_params_yaml[cam]["distortion_coeffs"] for cam in self.camera_names]

        self.CAM_W = self.camera_resolution[0]
        self.CAM_H = self.camera_resolution[1]
        self.NCAMS = len(self.camera_names)
        self.DISTS = np.array(self.camera_distortions)
        self.Ks = self.triangulation_params["Ks"].astype(np.float32)
        self.Ps = self.triangulation_params["Ps"].astype(np.float32)
        self.CAM_DISTORTION_MODELS = [
            self.calibration_params_yaml[cam]["distortion_model"] for cam in self.camera_names
        ]
        self.EYE_3 = np.eye(3).astype(np.float32)

        if not os.path.exists(f"data/tmp/{self.config_name}.yaml") or not filecmp.cmp(
            self.calibration_params_path, f"data/tmp/{self.config_name}.yaml"
        ):
            print("Configuration files are not the same, recreating maps")
            if os.path.exists(f"data/tmp/{self.config_name}_distmap_xy_arr.yaml"):
                os.remove(f"data/tmp/{self.config_name}_distmap_xy_arr.npy")
            if os.path.exists(f"data/tmp/{self.config_name}_udistmap_xy_arr.yaml"):
                os.remove(f"data/tmp/{self.config_name}_udistmap_xy_arr.npy")
            shutil.copy(self.calibration_params_path, f"data/tmp/{self.config_name}.yaml")

        # (Optional) Save and load the precomputed matrix for a certain cam calibration
        try:
            self.map_xy_arr_path = f"data/tmp/{self.config_name}_distmap_xy_arr.npy"
            self.map_xy_arr = np.load(self.map_xy_arr_path).astype(np.float32)
        except FileNotFoundError:
            # map_xy_arr[cam_idx,y,x] contains distorted xy-coordinates of point [x,y]
            self.map_xy_arr = np.empty((self.NCAMS, self.CAM_H, self.CAM_W, 2))
            for i, (K, dist_model, dist) in enumerate(zip(self.Ks, self.CAM_DISTORTION_MODELS, self.DISTS)):
                if dist_model == "equidistant":
                    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                        K, dist, self.EYE_3, K, tuple(self.camera_resolution), cv2.CV_32F
                    )
                elif dist_model == "radtan":
                    map_x, map_y = cv2.initUndistortRectifyMap(
                        K, dist, self.EYE_3, K, tuple(self.camera_resolution), cv2.CV_32F
                    )
                else:
                    raise ValueError(  # pylint: disable=raise-missing-from
                        f"Unexpected distortion model '{dist_model}'"
                    )
                self.map_xy_arr[i, :, :, 0] = map_x
                self.map_xy_arr[i, :, :, 1] = map_y
            os.makedirs(os.path.dirname(self.map_xy_arr_path), exist_ok=True)
            np.save(self.map_xy_arr_path, self.map_xy_arr)

        try:
            self.umap_xy_arr_path = f"data/tmp/{self.config_name}_udistmap_xy_arr.npy"
            self.umap_xy_arr = np.load(self.umap_xy_arr_path).astype(np.float32)
        except FileNotFoundError:
            # umap_xy_arr[cam_idx,y,x] contains undistorted xy-coordinatse of point [x,y]
            self.umap_xy_arr = np.empty((self.NCAMS, self.CAM_H, self.CAM_W, 2))
            for i, (K, dist_model, dist) in enumerate(zip(self.Ks, self.CAM_DISTORTION_MODELS, self.DISTS)):
                for x in range(self.CAM_W):
                    for y in range(self.CAM_H):
                        pt_2d = [x, y]
                        pt_dist = np.array(pt_2d, dtype=np.float32).reshape((1, 1, 2))
                        if dist_model == "equidistant":
                            pt_undist = cv2.fisheye.undistortPoints(pt_dist, K, dist, self.EYE_3, K)
                        elif dist_model == "radtan":
                            pt_undist = cv2.undistortPoints(pt_dist, K, dist, self.EYE_3, K)
                        else:
                            raise ValueError(  # pylint: disable=raise-missing-from
                                f"Unexpected distortion model '{dist_model}'"
                            )
                        self.umap_xy_arr[i, y, x] = pt_undist[0, 0]
            np.save(self.umap_xy_arr_path, self.umap_xy_arr)

    # ###########################
    # # Function defs
    # ###########################
    @njit(parallel=True, cache=True)
    def triangulate_points(CAM_W, CAM_H, cam_idx_arr, xy_arr, Ps, umap_xy_arr):  # pylint: disable =no-self-argument
        """
        Triangulates correspoinding 2D image points into 3D world coords.

        Input: (input with xy_arr [np.nan, np.nan] for out of frame.)
            cam_idx_arr: [0,1,2,3,4,5]  shape: [ncams]
            xy_arr: [[[x,y] ..]]        shape: [npoints, ncams, 2]

        Output:
            xyz_arr: [[x,y,z], ..]   shape: [npoints, 3]
        """
        ncams = len(cam_idx_arr)
        npoints = xy_arr.shape[0]
        xyz_arr = np.empty((npoints, 3))

        for i in prange(npoints):  # pylint: disable = not-an-iterable
            cams_no_nan = ~(np.isnan(xy_arr[i, :]).sum(axis=1) > 1)
            ncams_no_nan = np.sum(cams_no_nan)
            if ncams_no_nan < 2:  # cannot triangulate with less than 2 points
                xyz_arr[i] = np.array([np.nan, np.nan, np.nan])
            else:
                A = np.empty((2 * ncams_no_nan, 4))
                pcount = 0
                for ii in range(ncams):
                    if cams_no_nan[ii]:
                        cam_idx = cam_idx_arr[ii]
                        x, y = xy_arr[i, ii]
                        u_low = int(x)
                        v_low = int(y)
                        # Undistort
                        if (u_low + 1 >= CAM_W) or (v_low + 1 >= CAM_H):
                            ux, uy = umap_xy_arr[cam_idx, int(y), int(x)]
                        else:  # bilinear interpolation
                            u_alpha = x - u_low
                            v_alpha = y - v_low

                            w0 = (1.0 - u_alpha) * (1.0 - v_alpha)
                            w1 = (u_alpha) * (1.0 - v_alpha)
                            w2 = (1.0 - u_alpha) * (v_alpha)
                            w3 = (u_alpha) * (v_alpha)

                            umap_x = umap_xy_arr[cam_idx, :, :, 0]
                            umap_y = umap_xy_arr[cam_idx, :, :, 1]
                            ux = (
                                w0 * umap_x[v_low, u_low]
                                + w1 * umap_x[v_low, u_low + 1]
                                + w2 * umap_x[v_low + 1, u_low]
                                + w3 * umap_x[v_low + 1, u_low + 1]
                            )
                            uy = (
                                w0 * umap_y[v_low, u_low]
                                + w1 * umap_y[v_low, u_low + 1]
                                + w2 * umap_y[v_low + 1, u_low]
                                + w3 * umap_y[v_low + 1, u_low + 1]
                            )

                        P = Ps[cam_idx]

                        A[2 * pcount, :] = ux * P[2, :] - P[0, :]
                        A[2 * pcount + 1, :] = uy * P[2, :] - P[1, :]
                        pcount = pcount + 1

                _, _, Vh = np.linalg.svd(A)
                xyz_arr[i] = Vh[3, :3] / Vh[3, 3]

        return xyz_arr

    @njit(parallel=True, cache=True)
    def project_points(CAM_W, CAM_H, cam_idx_arr, xyz_arr, Ps, map_xy_arr):  #  pylint: disable =no-self-argument
        """
        Projects 3D points in world coords to 2D images.

        Input:
            cam_idx_arr: [0,1,2,3,4,5]  shape: [ncams]
            xyz_arr: [[x, y, z] ..]     shape: [npoints, 3]

        Output: (output with xy_arr [np.nan, np.nan] for out of frame)
            xy_arr: [[[x,y] ..]]    shape: [npoints, ncams, 2]
        """
        ncams = len(cam_idx_arr)
        npoints = xyz_arr.shape[0]
        xy_arr = np.empty((npoints, ncams, 2))
        nan_arr = np.array([np.nan, np.nan])

        for i in prange(ncams):  # pylint: disable = not-an-iterable
            idx = cam_idx_arr[i]
            P = Ps[idx]
            X = np.ones(4, dtype=np.float32)
            for ii in range(npoints):
                X[:3] = xyz_arr[ii]
                x = P.dot(X)
                x_undist = x[:2] / x[2]

                # distort
                # bilinear interpolation
                u_low = int(x_undist[0])
                v_low = int(x_undist[1])

                if (u_low + 1 >= CAM_W) or (v_low + 1 >= CAM_H) or (u_low < 0) or (v_low < 0):
                    xy_arr[ii, i] = nan_arr
                else:
                    u_alpha = x_undist[0] - u_low
                    v_alpha = x_undist[1] - v_low

                    w0 = (1.0 - u_alpha) * (1.0 - v_alpha)
                    w1 = (u_alpha) * (1.0 - v_alpha)
                    w2 = (1.0 - u_alpha) * (v_alpha)
                    w3 = (u_alpha) * (v_alpha)

                    x_dist = np.zeros(2)
                    map_x = map_xy_arr[idx, :, :, 0]
                    map_y = map_xy_arr[idx, :, :, 1]
                    x_dist[0] = (
                        w0 * map_x[v_low, u_low]
                        + w1 * map_x[v_low, u_low + 1]
                        + w2 * map_x[v_low + 1, u_low]
                        + w3 * map_x[v_low + 1, u_low + 1]
                    )
                    x_dist[1] = (
                        w0 * map_y[v_low, u_low]
                        + w1 * map_y[v_low, u_low + 1]
                        + w2 * map_y[v_low + 1, u_low]
                        + w3 * map_y[v_low + 1, u_low + 1]
                    )
                    xy_arr[ii, i] = x_dist

        return xy_arr


###########################
# Class interface
###########################
class Triangulator:
    """Wrapper class over numba functions for easy of use."""

    def __init__(self, config):
        """Init."""
        self.triangulation_numba = TriangulationNumba(config)
        self.cam_idx_arr = np.array(list(range(len(self.triangulation_numba.camera_names))), dtype=np.int)

        # # compile triangulation
        # pts_2d = [
        #     [[600, 600], [600, 600], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
        # ] * 5  # 5 identical points
        # pts_2d = np.array(pts_2d, dtype=np.float)
        # pts_3d = self.triangulate_points(pts_2d)
        # # compile projection
        # pts_3d = [[0.3, 0.3, 0.3]] * 5  # 5 identical points
        # pts_3d = np.array(pts_3d, dtype=np.float)
        # self.project_points(pts_3d)

    def triangulate_points(self, xy_arr: np.ndarray, cam_idx_arr=None) -> np.ndarray:
        """
        Triangulates correspoinding 2D image points into 3D world coords.

        Input: (input with xy_arr [np.nan, np.nan] for out of frame.)
            cam_idx_arr: [0,1,2,3,4,5]  shape: [ncams]
            xy_arr: [[[x,y] ..]]        shape: [npoints, ncams, 2]

        Output:
            output_arr: [[x,y,z], ..]   shape: [npoints, 3]
        """
        if cam_idx_arr is None:  # Not recommended
            cam_idx_arr = self.cam_idx_arr
        return TriangulationNumba.triangulate_points(
            self.triangulation_numba.CAM_W,
            self.triangulation_numba.CAM_H,
            cam_idx_arr,
            xy_arr,
            self.triangulation_numba.Ps,
            self.triangulation_numba.umap_xy_arr,
        )

    def project_points(self, xyz_arr: np.ndarray, cam_idx_arr=None) -> np.ndarray:
        """
        Projects 3D points in world coords to 2D images.

        Input:
            cam_idx_arr: [0,1,2,3,4,5]  shape: [ncams]
            xyz_arr: [[x, y, z] ..]     shape: [npoints, 3]

        Output: (output with xy_arr [np.nan, np.nan] for out of frame)
            xy_arr: [[[x,y] ..]]    shape: [npoints, ncams, 2]
        """
        if cam_idx_arr is None:  # Not recommended
            cam_idx_arr = self.cam_idx_arr
        return TriangulationNumba.project_points(
            self.triangulation_numba.CAM_W,
            self.triangulation_numba.CAM_H,
            cam_idx_arr,
            xyz_arr,
            self.triangulation_numba.Ps,
            self.triangulation_numba.map_xy_arr,
        )


if __name__ == "__main__":
    start = time()
    triangulation_numba = TriangulationNumba("tyo02")
    triangulator = Triangulator("tyo02")
    print(f"Compilation Ends in: {time() - start}s")

    # Triangulate points
    start = time()
    pts_2d = np.array(
        [[[600, 600], [600, 600], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]] * 17,
        dtype=np.float,
    )
    for _ in range(2):  # 2 players
        for _ in range(1000):  # 1000 frames
            pts_3d = triangulator.triangulate_points(pts_2d)
    print(f"Triangulation Ends in: {time() - start}s")
    # print(pts_3d)

    # Project points
    start = time()
    pts_3d = np.array([[0.3, 0.3, 0.3]] * 17, dtype=np.float)
    for _ in range(2):  # 2 players
        for _ in range(1000):  # 1000 frames
            pts_2d = triangulator.project_points(pts_3d)
    print(f"Projection End in: {time() - start}s")
    # print(pts_2d)

    print(f"Error in meters for: {pts_3d[0]}")
    pts_3d_new = triangulator.triangulate_points(triangulator.project_points(pts_3d))
    print("E = ", (pts_3d - pts_3d_new)[0])  # reprojection errors (in meters) have to be small
    print(f"Error in pixels for: {pts_2d[0]}")
    pts_2d_new = triangulator.project_points(triangulator.triangulate_points(pts_2d))
    print("E = ", (pts_2d - pts_2d_new)[0])  # reprojection errors (in pixels) have to be small
