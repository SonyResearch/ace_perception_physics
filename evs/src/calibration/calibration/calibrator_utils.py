# SPDX-License-Identifier: MIT
"""Provide various calibration-related routines."""

import logging

import numpy

log = logging.getLogger(__name__)


def get_angle_from_matrix(transform):
    """@return: The rotation angle of the axis-angle representation (in radians)."""
    return numpy.arccos(numpy.clip(0.5 * (numpy.trace(transform[:3, :3]) - 1), -1, 1))


def get_homogeneous(points):
    """Convert points into the homogeneous space."""
    return numpy.hstack([points, numpy.ones(points.shape[:-1] + (1,), dtype=points.dtype)])


def transform_points(transform_matrix, points, return_homo=False):
    """Apply a transformation to input points.

    @return: The transformed points
    """
    transformed_homo_points = numpy.matmul(get_homogeneous(points), numpy.transpose(transform_matrix))
    if return_homo:
        return transformed_homo_points
    return transformed_homo_points[:, :-1] / transformed_homo_points[:, [-1]]


def compute_affine_transformation(src_points, dst_points, scale_ratio=0.0, weights=None):
    """
    Find the unique homogeneous affine transformation that maps a set of 3 points to another set of 3 points.

    It computes rotation_matrix and translation_vect to minimize the error for:
        dst_points == numpy.matmul(src_points, rotation_matrix.T) + translation_vect
    where `rotation_matrix` is an unknown rotation matrix, `translation_vect` is an unknown
    translation vector, and `src_points` and `dst_points` are the original n x 3 pair of points
    The result of this function is an augmented 4-by-4
    matrix `transform` that represents this affine transformation:
        numpy.hstack([dst_points, numpy.ones((dst_points.shape[0], 1))]) == \
            numpy.matmul(numpy.hstack([src_points, numpy.ones((src_points.shape[0], 1))]), transform.T)
    Sources:
     - Umeyama, Shinji. "Least-squares estimation of transformation parameters between two point patterns."
     - Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using SVD."
    """
    if src_points.shape != dst_points.shape:
        raise ValueError("Points are not of the same shape! {} != {}".format(src_points.shape, dst_points.shape))

    # Filter invalid points out
    filter_mask = numpy.isfinite(src_points).all(axis=1) & numpy.isfinite(dst_points).all(axis=1)
    if weights is not None:
        filter_mask &= numpy.isfinite(weights)
    src_points = src_points[filter_mask]
    dst_points = dst_points[filter_mask]
    if weights is not None:
        weights = weights[filter_mask]

    if src_points.shape[0] < 3:
        raise ValueError("Just {} points is not enough to compute the affine transform".format(src_points.shape[0]))

    # Construct covariance matrix
    src_points_avg = numpy.average(src_points, axis=0, weights=weights)
    dst_points_avg = numpy.average(dst_points, axis=0, weights=weights)
    src_points_centered = src_points - src_points_avg
    dst_points_centered = dst_points - dst_points_avg
    co_var = (
        numpy.matmul(dst_points_centered.T, src_points_centered * weights[:, numpy.newaxis])
        if weights is not None
        else numpy.matmul(dst_points_centered.T, src_points_centered)
    )

    # Calculate rotation matrix
    matrix_u, _, matrix_v_h = numpy.linalg.svd(co_var)
    rotation_matrix = numpy.matmul(matrix_u, matrix_v_h)
    if numpy.linalg.det(rotation_matrix) < 0:  # Avoid reflection
        matrix_u[:, 2] *= -1
        rotation_matrix = numpy.matmul(matrix_u, matrix_v_h)

    # Compute scale component
    if scale_ratio != 0.0:
        scale_vect = (
            numpy.average(dst_points_centered**2, axis=0, weights=weights)
            / numpy.average(numpy.matmul(src_points_centered, rotation_matrix.T) ** 2, axis=0, weights=weights)
        ) ** 0.5
        rotation_matrix *= numpy.ones(3) * (1.0 - scale_ratio) + scale_vect * scale_ratio

    # Calculate translation vector
    translation_vect = (dst_points_avg - numpy.matmul(src_points_avg, rotation_matrix.T)).reshape((-1, 1))

    # Concatenate the final affine transformation matrix
    return numpy.vstack([numpy.hstack([rotation_matrix, translation_vect]), [0, 0, 0, 1]])


def compute_weighted_affine_transformation(
    src_points,
    dst_points,
    scale_ratio=0.0,
    error_to_sigma_multipliers=None,
    show_debug=False,
):
    """
    Compute the optimal weighted transform between two sets of points.

    This is done while reducing the effect of outliers via weighting.
    Weights are computed based on the fitting error,
    which starts with a relaxed standard-deviation (to capture the general trend first),
    and iteratively gets stricter (to exclude the outliers).
    By default three iterations are used, over 3, 1, and 0.3 standard-deviation multipliers.
    """
    transform = compute_affine_transformation(src_points, dst_points, scale_ratio=scale_ratio)
    transform_org = transform.copy()

    if error_to_sigma_multipliers is None:
        error_to_sigma_multipliers = [3.0, 1.0, 0.3]

    for error_to_sigma_multiplier in sorted(error_to_sigma_multipliers, reverse=True):
        errors_sqr = numpy.nansum((dst_points - transform_points(transform, src_points)) ** 2, axis=1)
        sigma = error_to_sigma_multiplier * errors_sqr.mean() ** 0.5
        if numpy.isclose(sigma, 0):  # Error is negligible. No need for weighing
            break
        weights = numpy.exp(-0.5 * errors_sqr / sigma**2)
        weights *= 1.0 / weights.sum()

        transform = compute_affine_transformation(src_points, dst_points, scale_ratio=scale_ratio, weights=weights)

        if show_debug:
            t_inc = numpy.matmul(transform, numpy.linalg.inv(transform_org))
            scale_vect = numpy.linalg.norm(t_inc[:3, :3], axis=0)
            rotation_matrix = numpy.matmul(numpy.diag(1.0 / scale_vect), t_inc[:3, :3])
            translation_vect = t_inc[:3, 3]
            log.debug(
                "Compared to LS, WLS with sigma=%.2f mm, has %d/%d outliers, and resulted in a transform"
                " with %.1f%% scale, %.1f deg rotation, and %.1f mm translation",
                1e3 * sigma,
                numpy.count_nonzero(weights < 0.1 / src_points.shape[0]),
                src_points.shape[0],
                1e2 * numpy.linalg.norm(scale_vect) / (3**0.5),
                numpy.rad2deg(get_angle_from_matrix(rotation_matrix)),
                1e3 * numpy.linalg.norm(translation_vect),
            )

    return transform


def triangulate_via_projection(src_points, dst_points):
    """
    Compute a triangulation point for input rays (represented by source and destination points).

    Sources:
    - https://stackoverflow.com/a/52089867
    - Least-Squares Intersection of Lines, by Johannes Traa - UIUC 2013
    """
    if src_points.shape != dst_points.shape:  # shape: [n lines, d dimensions]
        raise ValueError("Points are not of the same shape! {} != {}".format(src_points.shape, dst_points.shape))
    rays_dir = dst_points - src_points
    rays_dir /= numpy.linalg.norm(rays_dir, axis=1, keepdims=True)
    projection = (
        numpy.eye(rays_dir.shape[1]) - rays_dir[..., numpy.newaxis] * rays_dir[:, numpy.newaxis]
    )  # I - dir * dir.T (dims: n x d x d)
    directions = projection.sum(axis=0)
    intercepts = numpy.matmul(projection, src_points[..., numpy.newaxis]).sum(axis=0)
    point = numpy.linalg.lstsq(directions, intercepts, rcond=None)[0]
    return point.T


def get_mean_square_error(src, dst, axis=None):
    """@return: MSE error (not RMSE)."""
    return ((src - dst) ** 2).mean(axis=axis)


def get_root_mean_square_error(points_3d, matching_robot_points_3d, scale_factor=1.0):
    """@return: RMSE error between robot and valid camreas observations."""
    error_sqr = get_mean_square_error(points_3d, matching_robot_points_3d[:, numpy.newaxis], axis=2)[
        numpy.isfinite(points_3d[..., 2])
    ]
    return error_sqr.mean() ** 0.5 * scale_factor


def get_matching_points(points, time_stamps, query_time_stamps, filter_matches=True, max_time_diff=None):
    """@return: Trajctory points that corresponds to the query timestamps."""
    # FIXME(cv3d): Currently uses nearest interpolation. Need to improve that
    diff = numpy.abs(time_stamps[:, numpy.newaxis] - query_time_stamps[numpy.newaxis])
    indices = diff.argmin(axis=0)
    matching_points = points[indices]

    if filter_matches:
        if max_time_diff is None:
            max_time_diff = 2 * numpy.diff(time_stamps).mean()
        matching_points = matching_points.copy()
        matching_points[diff[indices, numpy.arange(diff.shape[1])] > max_time_diff] = numpy.nan
    return matching_points


def find_delay(src, dst):
    """
    Find delay between two signals.

    Currently, it assumes delay cannot exceed half of the signal length
    """
    start_offset, end_offset = -dst.shape[0] // 2, src.shape[0] // 2
    delays = numpy.arange(start_offset, end_offset).reshape((-1, 1))
    positive_negative_delays = numpy.hstack([numpy.maximum(delays, 0), numpy.maximum(-delays, 0)])
    errors = numpy.array(
        [
            get_mean_square_error(
                src[positive_delay : src.shape[0] - negative_delay],
                dst[negative_delay : dst.shape[0] - positive_delay],
            )
            for positive_delay, negative_delay in positive_negative_delays
        ]
    )
    delay = errors.argmin() + start_offset
    return delay
