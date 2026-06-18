# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
Camera calibration toolbox - utilities
"""

import collections
import copy

import cv2
import g2o.g2opy as g2o
import numpy
import scipy.optimize
import scipy.spatial
import sklearn.cluster
import tqdm
from calibration.calibox import calibox_graph


def sample_features(features, n_clusters):
    """
    Select the most relevant observations using K-means clustering.

    @param features: A 2D array-like structure with observation rows and feature columns.
    @param n_clusters: The number of clusters to form using K-means.

    @return: An array of indices corresponding to the most relevant observations selected from the input features.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    tree = scipy.spatial.cKDTree(features)
    selection_indices = tree.query(kmeans.cluster_centers_, k=1)[1]
    return selection_indices


def calibrate_camera(object_points, camera_name, camera_calib, detection_infos, max_observations=100, use_graph=False):
    """
    Calibrate the camera given the object points, camera parameters, and detection information.

    @param object_points: An array of 3D points in the object coordinate space.
    @param camera_name: The name of the camera to be calibrated.
    @param camera_calib: A dictionary containing camera calibration parameters,
                         including 'camera_matrix' and 'distortion_coeffs'.
    @param detection_infos: A dictionary containing detection information for the camera,
                            representing image points per pattern_name per camera_name.
    @param max_observations: Maximum number of observations used in calibration.
                            If the provided detections exceed this, K-means is used to select the relevant observations.

    @param use_graph: A boolean indicating whether to use graph optimization for calibration.

    @return: A tuple containing:
        - camera_matrix: The calibrated camera matrix.
        - distortion_coeffs: The calibrated distortion coefficients.
        - cam_pose_infos: A dictionary containing estimated pattern poses and corresponding information matrix.
    """
    camera_matrix, distortion_coeffs = camera_calib["camera_matrix"], camera_calib["distortion_coeffs"]

    if camera_matrix is not None:
        camera_matrix = camera_matrix.copy()
        camera_matrix[:2, :2] = numpy.diag([numpy.sqrt(numpy.prod(numpy.diag(camera_matrix[:2, :2])))] * 2)
        camera_matrix[:2, 2] = 0.5 * (camera_calib["resolution"] - 1)

    if distortion_coeffs is not None:
        distortion_coeffs = distortion_coeffs.ravel().copy()
        distortion_coeffs[2:] = 0

    observation_infos, observation_features = [], []
    for pattern_name, detection_info in detection_infos.items():
        image_points = detection_info.get(camera_name)
        if image_points is None:
            continue

        visibility_mask = numpy.isfinite(image_points).all(axis=1)
        total_observations = numpy.count_nonzero(visibility_mask)
        if total_observations < 5:
            continue

        curr_object_points = object_points.astype(numpy.float32)[visibility_mask]
        curr_image_points = image_points.astype(numpy.float32)[visibility_mask]
        observation_infos.append((pattern_name, curr_object_points, curr_image_points))

        # A good coverage features different image positions and pattern sizes
        box_center, box_size, _ = cv2.minAreaRect(curr_image_points)
        observation_features.append(numpy.hstack([box_center, numpy.linalg.norm(box_size)]))

    if len(observation_infos) > max_observations:
        selection_indices = sample_features(observation_features, max_observations)
        observation_infos = [observation_infos[i] for i in selection_indices]

    _, obs_object_points, obs_image_points = zip(*observation_infos)
    flags = (
        0  # (cv2.CALIB_USE_INTRINSIC_GUESS if camera_matrix is not None else 0)
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    _, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obs_object_points,
        obs_image_points,
        camera_calib["resolution"],
        camera_matrix,
        distortion_coeffs,
        flags=flags,
    )

    cam_pose_infos = {}
    for (pattern_name, _, image_points), rvec, tvec in zip(observation_infos, rvecs, tvecs):
        pose = numpy.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.ravel()
        cam_pose_infos[pattern_name] = (pose, 0.1 * image_points.shape[0] * numpy.eye(6))

    if use_graph:
        camera_calib["camera_matrix"], camera_calib["distortion_coeffs"] = camera_matrix, distortion_coeffs
        camera_matrix, distortion_coeffs, cam_pose_infos = calibrate_camera2(
            object_points, camera_name, camera_calib, detection_infos, cam_pose_infos=cam_pose_infos
        )

    return camera_matrix, distortion_coeffs, cam_pose_infos


def add_camera_parameters(optimizer, camera_matrix, cam_id=0):
    """
    Add camera parameters to the optimizer.

    @param optimizer: The optimizer to which the camera parameters will be added.
    @param camera_matrix: A 3x3 matrix describing the camera intrinsic parameters.
    @param cam_id: An optional integer ID for the camera parameters (default is 0).

    @return: The created camera parameter.
    """
    cam = g2o.CameraParameters(numpy.sqrt(numpy.prod(numpy.diag(camera_matrix[:2, :2]))), camera_matrix[:2, 2], 0)
    cam.set_id(cam_id)
    optimizer.add_parameter(cam)
    return cam


def add_point_vertex(optimizer, point_3d, point_id):
    """
    Add a 3D point vertex to the optimizer.

    @param optimizer: The optimizer to which the point vertex will be added.
    @param point_3d: A 3D numpy array representing the coordinates of the point.
    @param point_id: An integer ID for the point vertex.

    @return: The created point vertex.
    """
    point_vertex = g2o.VertexPointXYZ()
    point_vertex.set_id(point_id)
    point_vertex.set_estimate(point_3d)
    point_vertex.set_marginalized(True)
    optimizer.add_vertex(point_vertex)
    return point_vertex


def get_optimized_params(intrinsics_vertex, pattern_vertices, cam_pose_infos, use_hessian=False):
    """
    Retrieve optimized camera parameters from the optimization process.

    @param intrinsics_vertex: The vertex representing the camera intrinsics in the optimization graph.
    @param pattern_vertices: A list of tuples containing pattern names, total observations, and pattern vertex objects.
    @param cam_pose_infos: A dictionary containing initial pattern poses and their corresponding information matrices.
    @param use_hessian: A boolean flag indicating whether to use the Hessian for pose information.

    @return: A tuple containing:
        - camera_matrix: The optimized camera matrix.
        - distortion_coeffs: The optimized distortion coefficients.
        - cam_pose_infos: A dictionary containing estimated pattern poses and corresponding information matrices.
    """
    camera_matrix = numpy.eye(3)
    camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2] = intrinsics_vertex.estimate()[
        :4
    ]

    distortion_coeffs = numpy.zeros(4)

    for pattern_name, total_observations, pattern_vertex in pattern_vertices:
        pose = pattern_vertex.estimate().matrix()
        if use_hessian:
            h_inv = numpy.linalg.pinv(pattern_vertex.hessian(), hermitian=True)
            pose_information = 1e7 * total_observations * h_inv
        else:
            pose_information = 0.1 * total_observations * numpy.eye(6)
        cam_pose_infos[pattern_name] = (pose, pose_information)

    return camera_matrix, distortion_coeffs, cam_pose_infos


def calibrate_camera2(object_points, camera_name, camera_calib, detection_infos, cam_pose_infos=None):
    """
    Calibrate the camera using a graph optimization approach.

    @param object_points: An array of 3D points in the object coordinate space.
    @param camera_name: The name of the camera to be calibrated.
    @param camera_calib: A dictionary containing camera calibration parameters,
                         including 'camera_matrix'.
    @param detection_infos: A dictionary containing detection information for the camera,
                            representing image points per pattern_name per camera_name.
    @param cam_pose_infos: An optional dictionary containing the initial pattern poses.

    @return: A tuple containing:
        - camera_matrix: The calibrated camera matrix.
        - distortion_coeffs: The calibrated distortion coefficients.
        - cam_pose_infos: A dictionary containing estimated pattern poses and corresponding information matrix.
    """
    use_ternary = False

    if cam_pose_infos is None:
        cam_pose_infos = {}
    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(False)
    linear_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(linear_solver)
    optimizer.set_algorithm(solver)

    camera_matrix = camera_calib["camera_matrix"]
    distortion_coeffs = camera_calib["distortion_coeffs"]
    add_camera_parameters(optimizer, camera_matrix)

    intrinsics_vertex = g2o.VertexIntrinsics()
    intrinsics_vertex.set_id(0)
    intrinsics_vertex.set_estimate(
        [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], 0]
    )
    optimizer.add_vertex(intrinsics_vertex)

    point_base = 1
    for i, point_3d in enumerate(object_points):
        add_point_vertex(optimizer, point_3d, i + point_base)

    pattern_base = 1 + len(object_points)
    pattern_vertices = []
    for i, (pattern_name, detection_info) in enumerate(detection_infos.items()):
        image_points = detection_info.get(camera_name)
        if image_points is None:
            continue
        visible_indices = numpy.where(numpy.isfinite(image_points).all(axis=1))[0]
        if len(visible_indices) < 5:
            continue

        pose, _ = cam_pose_infos.get(pattern_name, (None, None))
        if pose is None:
            pose, _ = solve_pnp(
                object_points[visible_indices], image_points[visible_indices], camera_matrix, distortion_coeffs
            )
        pattern_vertex = g2o.VertexSE3Expmap()
        pattern_vertex.set_id(len(pattern_vertices) + pattern_base)
        pattern_vertex.set_estimate(g2o.SE3Quat(pose[:3, :3], pose[:3, 3]))
        optimizer.add_vertex(pattern_vertex)
        pattern_vertices.append((pattern_name, len(visible_indices), pattern_vertex))

        for j in visible_indices:
            if use_ternary:
                edge = g2o.EdgeProjectPSI2UV()
                edge.resize(3)
            else:
                edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, optimizer.vertex(j + point_base))
            edge.set_vertex(1, pattern_vertex)
            if use_ternary:
                edge.set_vertex(2, intrinsics_vertex)
            edge.set_measurement(image_points[j])
            edge.set_information(numpy.eye(2))
            edge.set_robust_kernel(g2o.RobustKernelCauchy())
            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

    optimizer.initialize_optimization()
    optimizer.optimize(1000)

    return get_optimized_params(intrinsics_vertex, pattern_vertices, cam_pose_infos)


def solve_pnp(
    object_points, image_points, camera_matrix, distortion_coeffs, pose=None, pose_information=None, use_graph=True
):
    """
    Solve the PnP problem to estimate the camera pose given object and image points.

    @param object_points: Array of 3D points in the object coordinate space.
    @param image_points: Array of corresponding 2D points in the image coordinate space.
    @param camera_matrix: The intrinsic camera matrix.
    @param distortion_coeffs: The distortion coefficients.
    @param pose: Optional initial pose estimate (4x4 transformation matrix).
    @param pose_information: Optional information matrix for the pose estimation.

    @param use_graph: A boolean indicating whether to use graph optimization.

    @return: A tuple containing:
        - pose: A 4x4 transformation matrix representing the camera pose.
        - pose_information: Information matrix for the pose estimation.
    """
    if pose is None:
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coeffs)
        pose = numpy.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.ravel()
    if pose_information is None:
        pose_information = 0.1 * image_points.shape[0] * numpy.eye(6)

    if use_graph:
        pose, pose_information = solve_pnp2(
            object_points, image_points, camera_matrix, distortion_coeffs, pose=pose, pose_information=pose_information
        )

    return pose, pose_information


def solve_pnp2(
    object_points, image_points, camera_matrix, distortion_coeffs, pose=None, pose_information=None, use_hessian=False
):
    """
    Refine the camera pose estimation using optimization based on the PnP solution.

    @param object_points: Array of 3D points in the object coordinate space.
    @param image_points: Array of corresponding 2D points in the image coordinate space.
    @param camera_matrix: The intrinsic camera matrix.
    @param distortion_coeffs: The distortion coefficients.
    @param pose: Optional initial pose estimate (4x4 transformation matrix).
    @param pose_information: Optional information matrix for the pose estimation.
    @param use_hessian: A boolean indicating whether to use the Hessian for pose information.

    @return: A tuple containing:
        - pose: A 4x4 transformation matrix representing the refined camera pose.
        - pose_information: Information matrix for the refined pose estimation.
    """
    if pose is None:
        pose = numpy.eye(4)
        pose[:3, 3] = [0, 0, 3]
    if pose_information is None:
        pose_information = 0.1 * image_points.shape[0] * numpy.eye(6)

    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(False)
    linear_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(linear_solver)
    optimizer.set_algorithm(solver)

    add_camera_parameters(optimizer, camera_matrix)

    camera_vertex = g2o.VertexSE3Expmap()
    camera_vertex.set_id(0)
    camera_vertex.set_estimate(g2o.SE3Quat(pose[:3, :3], pose[:3, 3]))
    optimizer.add_vertex(camera_vertex)

    image_points = cv2.undistortImagePoints(image_points, camera_matrix, distortion_coeffs).reshape(-1, 2)
    for i, (object_point, image_point) in enumerate(zip(object_points, image_points)):
        point_vertex = add_point_vertex(optimizer, object_point, i + 1)

        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, point_vertex)
        edge.set_vertex(1, camera_vertex)
        edge.set_measurement(image_point)
        edge.set_information(numpy.eye(2))
        edge.set_robust_kernel(g2o.RobustKernelCauchy())
        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)

    optimizer.initialize_optimization()
    optimizer.optimize(100)

    pose = camera_vertex.estimate().matrix()
    if use_hessian:
        h_inv = numpy.linalg.pinv(camera_vertex.hessian(), hermitian=True)
        pose_information = 1e7 * image_points.shape[0] * h_inv

    return pose, pose_information


def graph_optimization(
    object_points, detection_infos, calibration_params, *, relative_poses=None, pattern_names=None, fix_cameras=False
):
    """
    Constructs and solves a pose graph optimization problem using the provided object points, detection information,
    and camera calibration parameters.

    @param object_points: Array of 3D object points to be used in the optimization.
    @param detection_infos: Dictionary mapping pattern names to detected image points for each camera.
    @param calibration_params: Dictionary containing the initial calibration parameters for each camera, including
                              transformation matrices and camera matrices.
    @param relative_poses: Optional dictionary containing relative poses for each camera with respect to patterns.
                           Defaults to None, in which case an empty dictionary is used.

    @param pattern_names: Optional list of pattern names to consider. If None, all keys in detection_infos are used.
    `@param fix_cameras: Boolean indicating whether to fix all camera vertices during graph optimization,
                         useful for assessing existing calibration quality without optimization.`

    @return: A tuple containing:
             - calibration_params: Updated camera calibration parameters with optimized transformation matrices.
             - pattern_poses: Dictionary mapping pattern names to their optimized poses.
    """
    if relative_poses is None:
        relative_poses = {}
    if pattern_names is None:
        pattern_names = detection_infos.keys()

    graph = calibox_graph.PoseGraph()
    graph.add_points(object_points)

    camera_names = list(calibration_params.keys())
    fixed_cameras = camera_names if fix_cameras else camera_names[:1]
    for camera_name, camera_calib in calibration_params.items():
        camera_calib = calibration_params[camera_name]
        graph.add_camera(camera_name, numpy.linalg.inv(camera_calib["T_CW"]), is_fixed=camera_name in fixed_cameras)

    for pattern_name in tqdm.tqdm(pattern_names, desc="Constructing pose-graph"):
        pattern_transform = None
        # Add pattern node for now, without a transform, which will be updated later
        pattern_vertex = graph.add_pattern(pattern_name, initial_transform=pattern_transform)
        for camera_name, image_points in detection_infos[pattern_name].items():
            camera_calib = calibration_params[camera_name]
            pose, pose_information = relative_poses.setdefault(camera_name, {}).get(pattern_name, (None, None))
            visibility_mask = numpy.isfinite(image_points).all(axis=1)
            total_observations = numpy.count_nonzero(visibility_mask)
            if total_observations < 6 and pose is None:
                continue

            pose, pose_information = solve_pnp(
                object_points[visibility_mask],
                image_points[visibility_mask],
                camera_calib["camera_matrix"],
                camera_calib["distortion_coeffs"],
                pose=pose,
                pose_information=pose_information,
            )
            relative_poses[camera_name][pattern_name] = (pose, pose_information)
            graph.add_pose(camera_name, pattern_name, pose, pose_information)

            if pattern_transform is None:
                # Update pattern transform in the world coordinates
                pattern_transform = numpy.matmul(numpy.linalg.inv(camera_calib["T_CW"]), pose)
                graph.update_transform(pattern_vertex, pattern_transform)

    graph.solve()

    # Update transforms
    for camera_name, camera_calib in calibration_params.items():
        camera_calib.update(
            {
                "T_CW": numpy.linalg.inv(graph.get_pose(camera_name)[0]),
            }
        )

    pattern_poses = {pattern_name: graph.get_pose(pattern_name) for pattern_name in detection_infos}
    return calibration_params, pattern_poses


def apply_intrinsics_deltas(deltas, camera_calib):
    """
    Updates the intrinsic parameters of the camera calibration.

    @param deltas: A list or array of deltas that will be applied to the camera's
                   intrinsic parameters. This should include adjustments for the
                   camera matrix and distortion coefficients.
    @param camera_calib: A dictionary containing the camera calibration parameters
                         that need to be updated.

    @return: A tuple containing two elements:
             - An integer indicating the number of deltas applied (which is 8).
             - regularization: A list or array of parameter penalties.
    """
    rows = [0, 1, 0, 1]
    cols = [0, 1, 2, 2]
    deltas = deltas[:8] * [100, 100, 10, 10, 0.1, 0.1, 1e-3, 1e-3]
    camera_calib["camera_matrix"][rows, cols] += deltas[:4]
    camera_calib["distortion_coeffs"] = camera_calib["distortion_coeffs"].ravel()[:4]
    camera_calib["distortion_coeffs"] += deltas[4:8]

    # Parameters regularization
    focal_lengths = camera_calib["camera_matrix"][rows[:2], cols[:2]]
    curr_axis = camera_calib["camera_matrix"][rows[2:4], cols[2:4]]
    expected_axis = 0.5 * (camera_calib["resolution"] - 1)
    regularization = numpy.hstack(
        [
            1e2 * numpy.diff(focal_lengths),
            1e3 * numpy.diff(curr_axis - expected_axis),
            1e5 * camera_calib["distortion_coeffs"][2:4],
        ]
    )
    return len(deltas), regularization


def apply_pose_deltas(deltas, pose):
    """
    Updates the pose based on the provided deltas.

    @param deltas: A list or array of deltas to be applied to the pose transformation.
                   This should include adjustments for the rotation (first 3 deltas)
                   and translation (next 3 deltas).
    @param pose: A 4x4 transformation matrix representing the pose of the pattern
                 that needs to be updated.

    @return: A tuple containing two elements:
             - An integer indicating the number of deltas applied (which is 6).
             - regularization: A list or array of parameter penalties.
    """
    deltas = deltas[:6]
    pose[:3, :3] = cv2.Rodrigues(cv2.Rodrigues(pose[:3, :3])[0].ravel() + deltas[:3])[0]
    pose[:3, 3] += deltas[3:6]
    return len(deltas), []


def apply_deltas(deltas, calibration_params, pattern_poses, camera_names, with_intrinsics, with_extrinsics):
    """
    Applies delta adjustments to camera calibration parameters and pattern poses.

    This function modifies the input calibration parameters and pattern poses
    based on the provided deltas. The deltas are expected to be structured to
    match the parameters being adjusted.

    @param deltas: A list or array of deltas to be applied. The length of
                   deltas must be sufficient to cover the required adjustments
                   for all cameras and pattern poses.
    @param calibration_params: Dictionary containing calibration parameters
                              for each camera. Each camera's parameters must
                              include keys for "camera_matrix", "distortion_coeffs",
                              and "T_CW".
    @param pattern_poses: Dictionary mapping pattern names to their estimated poses.
                          Each pose must be represented as a 4x4 transformation matrix.
    @param camera_names: List of camera names to consider in optimization.
    @param with_intrinsics: If True, optimizes for intrinsic parameters.
    @param with_extrinsics: If True, optimizes for extrinsic parameters.

    @return: A tuple containing two elements:
             - new_calibration_params: A dictionary of updated camera calibration parameters.
             - new_pattern_poses: A dictionary of updated pattern poses.
             - regularization: A list or array of parameter penalties.
    """
    new_calibration_params = copy.deepcopy(calibration_params)
    new_pattern_poses = copy.deepcopy(pattern_poses)
    offset = 0
    regularization = []
    for i, camera_name in enumerate(camera_names):
        camera_calib = new_calibration_params[camera_name]
        if with_intrinsics:
            curr_offset, curr_penalties = apply_intrinsics_deltas(deltas[offset:], camera_calib)
            regularization.extend(curr_penalties)
            offset += curr_offset
        if with_extrinsics and i != 0:
            offset += apply_pose_deltas(deltas[offset:], camera_calib["T_CW"])
    if with_extrinsics:
        for pose, _ in pattern_poses.values():
            curr_offset, curr_penalties = apply_pose_deltas(deltas[offset:], pose)
            regularization.extend(curr_penalties)
            offset += curr_offset
    assert len(deltas) == offset

    return new_calibration_params, new_pattern_poses, regularization


def compute_error(projection_infos, detection_infos):
    """
    Compute the total squared error between projected points and detected points.

    @param projection_infos: Dictionary mapping pattern names to projected points for each camera.
    @param detection_infos: Dictionary mapping pattern names to detected points for each camera.
    @return: The total squared error as a float.
    """
    errors = []
    for pattern_name, projection_info in projection_infos.items():
        for camera_name, projected_points in projection_info.items():
            image_points = detection_infos[pattern_name][camera_name]
            visibility_mask = numpy.isfinite(image_points).all(axis=1)
            proj_diff = projected_points[visibility_mask] - image_points[visibility_mask]
            errors.append(proj_diff)
    errors = numpy.vstack(errors).ravel()
    print(f"RMSE={numpy.mean(errors**2) ** 0.5:6.4f}px", end="\r")
    return errors


def project_points(object_points, detection_infos, calibration_params, pattern_poses, camera_names=None):
    """
    Project 3D object points into image space using camera parameters.

    @param object_points: Array of 3D object points to be projected.
    @param detection_infos: Dictionary mapping pattern names to detected points for each camera.
    @param calibration_params: Dictionary containing calibration parameters for each camera.
    @param pattern_poses: Dictionary mapping pattern names to their estimated poses.

    @param camera_names: Optional list of camera names to consider for projection.
                         If None, all cameras in calibration_params are used.

    @return: A dictionary mapping pattern names to detected and projected points for each camera.
    """
    projection_infos = collections.defaultdict(dict)
    for pattern_name, detection_info in detection_infos.items():
        pattern_pose, _ = pattern_poses[pattern_name]
        for camera_name in detection_info if camera_names is None else camera_names:
            camera_calib = calibration_params[camera_name]
            pose = numpy.matmul(camera_calib["T_CW"], pattern_pose)
            rvec = cv2.Rodrigues(pose[:3, :3])[0]
            tvec = pose[:3, 3:4]

            projected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_calib["camera_matrix"], camera_calib["distortion_coeffs"]
            )
            projection_infos[pattern_name][camera_name] = projected_points.reshape(-1, 2)
    return projection_infos


def get_delta_cost(
    deltas,
    object_points,
    detection_infos,
    calibration_params,
    pattern_poses,
    camera_names,
    with_intrinsics,
    with_extrinsics,
):
    """
    Computes the cost of applying the given deltas to the calibration parameters and pattern poses.

    @param deltas: A list or array of deltas to be applied to the camera calibration parameters and pattern poses.
    @param object_points: Array of 3D object points to be projected.
    @param detection_infos: Dictionary mapping pattern names to detected points for each camera.
    @param calibration_params: Dictionary containing calibration parameters for each camera.
    @param pattern_poses: Dictionary mapping pattern names to their estimated poses.
    @param camera_names: List of camera names to consider in optimization.
    @param with_intrinsics: If True, optimizes for intrinsic parameters.
    @param with_extrinsics: If True, optimizes for extrinsic parameters.

    @return: A float representing the computed error based on projected points and detection information.
    """
    new_calibration_params, new_pattern_poses, regularization = apply_deltas(
        deltas, calibration_params, pattern_poses, camera_names, with_intrinsics, with_extrinsics
    )

    projection_infos = project_points(
        object_points, detection_infos, new_calibration_params, new_pattern_poses, camera_names
    )
    error = compute_error(projection_infos, detection_infos)
    return numpy.hstack([error, regularization])


def least_squares_optimization(
    object_points,
    detection_infos,
    calibration_params,
    pattern_poses,
    *,
    camera_names=None,
    with_intrinsics=True,
    with_extrinsics=True,
):
    """
    Optimizes the camera calibration parameters and pattern poses using minimization.

    @param object_points: Array of 3D object points to be projected.
    @param detection_infos: Dictionary mapping pattern names to detected points for each camera.
    @param calibration_params: Dictionary containing the initial calibration parameters for each camera.
    @param pattern_poses: Dictionary mapping pattern names to their estimated poses.
    @param camera_names: List of camera names to consider in optimization.
    @param with_intrinsics: If True, optimizes for intrinsic parameters.
    @param with_extrinsics: If True, optimizes for extrinsic parameters.

    @return: A tuple containing the optimized camera calibration parameters and pattern poses.
    """
    if camera_names is None:
        camera_names = list(calibration_params.keys())
    intrinsics_size = 8 if with_intrinsics else 0
    extrinsics_size = 6 if with_extrinsics else 0
    deltas = numpy.zeros(
        len(camera_names) * intrinsics_size + (len(camera_names) - 1 + len(pattern_poses)) * extrinsics_size
    )

    opt = scipy.optimize.least_squares(
        get_delta_cost,
        deltas,
        loss="huber",
        args=(
            object_points,
            detection_infos,
            calibration_params,
            pattern_poses,
            camera_names,
            with_intrinsics,
            with_extrinsics,
        ),
    )
    return apply_deltas(opt.x, calibration_params, pattern_poses, camera_names, with_intrinsics, with_extrinsics)[:2]
