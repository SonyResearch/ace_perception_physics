# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
Camera Calibration Toolbox - Pose Graph
"""
import logging

import g2o.g2opy as g2o
import numpy
import tqdm

log = logging.getLogger(__name__)


class PoseGraph:
    """Pose-graph optimizer for camera calibration."""

    def __init__(self, verbose=False):
        """
        Initialize the PoseGraph optimizer.

        @param verbose: Boolean flag to enable verbose output during optimization.
        """
        self._optimizer = g2o.SparseOptimizer()
        self._optimizer.set_verbose(verbose)
        linear_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(linear_solver)
        self._optimizer.set_algorithm(solver)

        self._object_points = None
        self._vertices = []

    def add_points(self, object_points):
        """Store the calibration pattern points in the graph.

        @param object_points: Array of 3D points in the object coordinate space.
        """
        self._object_points = object_points
        base_id = len(self._vertices)
        for point_id, point in enumerate(self._object_points):
            vertex_name = f"point{point_id}"
            vertex_id = base_id + point_id
            self._vertices.append(vertex_name)
            vertex = g2o.VertexPointXYZ()
            vertex.set_id(vertex_id)
            vertex.set_estimate(point)
            vertex.set_marginalized(True)
            self._optimizer.add_vertex(vertex)

    def update_transform(self, vertex_id, transform):
        """
        Updates transformation matrix of a vertex.

        @param vertex_id: The ID assigned to the vertex.
        @param transform: 4x4 transformation matrix representing the current pose of the vertex.
        """
        vertex = self._optimizer.vertex(vertex_id)
        vertex.set_estimate(g2o.Isometry3d(transform[:3, :3], transform[:3, 3]))

    def _add_vertex(self, vertex_name, *, initial_transform=None, is_fixed=False):
        """
        Add a vertex to the graph.

        @param vertex_name: The name of the vertex.
        @param initial_transform: Initial transformation matrix for the vertex (optional).
        @param is_fixed: Boolean flag to fix the vertex in place (optional).

        @return: The ID assigned to the added vertex.
        """
        vertex_id = len(self._vertices)
        self._vertices.append(vertex_name)
        vertex = g2o.VertexSE3()
        vertex.set_id(vertex_id)
        vertex.set_fixed(is_fixed)
        self._optimizer.add_vertex(vertex)
        if initial_transform is not None:
            self.update_transform(vertex_id, initial_transform)
        return vertex_id

    def _add_edge(self, vertex_name1, vertex_name2, pose, pose_information):
        """
        Add an edge between two vertices representing a pose measurement.

        @param vertex_name1: Name of the first vertex.
        @param vertex_name2: Name of the second vertex.
        @param pose: 4x4 transformation matrix representing the current pose of the vertex
        @param pose_information: Information matrix for the edge.
        """
        pose_measurement = g2o.Isometry3d(pose[:3, :3], pose[:3, 3])
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, self._optimizer.vertex(self._vertices.index(vertex_name1)))
        edge.set_vertex(1, self._optimizer.vertex(self._vertices.index(vertex_name2)))
        edge.set_measurement(pose_measurement)
        edge.set_information(pose_information)
        edge.set_robust_kernel(g2o.RobustKernelCauchy())
        self._optimizer.add_edge(edge)

    def add_camera(self, camera_name, initial_transform=None, is_fixed=False):
        """Add a camera vertex to the graph.

        @param camera_name: Name of the camera.
        @param initial_transform: Initial transformation matrix for the camera (optional).
        @param is_fixed: Boolean flag to fix the camera in place (optional).

        @return: The ID assigned to the added camera vertex.
        """
        return self._add_vertex(camera_name, initial_transform=initial_transform, is_fixed=is_fixed)

    def add_pattern(self, pattern_name, initial_transform=None, is_fixed=False):
        """Add a calibration pattern vertex to the graph.

        @param pattern_name: Name of the pattern.
        @param initial_transform: Initial transformation matrix for the pattern (optional).
        @param is_fixed: Boolean flag to fix the pattern in place (optional).

        @return: The ID assigned to the added pattern vertex.
        """
        return self._add_vertex(pattern_name, initial_transform=initial_transform, is_fixed=is_fixed)

    def add_pose(self, vertex_name1, vertex_name2, pose, pose_information):
        """Add a relative pose edge between two vertices.

        @param vertex_name1: Name of the first vertex.
        @param vertex_name2: Name of the second vertex.
        @param pose: 4x4 transformation matrix representing the current pose of the vertex
        @param pose_information: Information matrix for the relative pose.
        """
        self._add_edge(vertex_name1, vertex_name2, pose, pose_information)

    def solve(self, n_iterations=100):
        """Minimize the error in the graph through optimization.

        @param n_iterations: The number of iterations to perform during optimization (default is 100).
        """
        self._optimizer.initialize_optimization()
        chi2 = self._optimizer.chi2()

        total_iterations = 0
        for _ in tqdm.tqdm(range(n_iterations), desc="Optimizing pose-graph"):
            total_iterations += 1
            self._optimizer.optimize(5)
            prev_chi2 = chi2
            chi2 = self._optimizer.chi2()
            if numpy.isclose(prev_chi2, chi2):
                break

        log.info("Finished optimization with iterations=%u, chi2=%.1f", total_iterations, chi2)

    def get_pose(self, vertex_name, use_hessian=False):
        """Retrieve the current pose of a vertex.

        @param vertex_name: Name of the vertex for which to retrieve the pose.
        @param use_hessian: A boolean indicating whether to use the Hessian for pose information.

        @return: A tuple containing:
            - A 4x4 transformation matrix representing the current pose of the vertex.
            - A 6x6 covariance matrix representing the current pose information.
        """
        vertex = self._optimizer.vertex(self._vertices.index(vertex_name))
        pose = vertex.estimate().matrix()
        pose_information = None
        if use_hessian:
            pose_information = numpy.linalg.pinv(vertex.hessian(), hermitian=True)
        return pose, pose_information
