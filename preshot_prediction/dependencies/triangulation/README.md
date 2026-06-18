# Triangulation

This package provides tools for triangulation of 2D points from a calibrated camera system and backprojection from 3D space to the camera system.

## Numba triangulator

Numba provides massively parallel triangulation functions. However, it needs to compile the python code. Expect the first run to be slower.

## Distortion and undistortion

Images captured by the camera system are passed around without correcting for distortion. We precompute the required undistortion matrices and save them for future use. Expect the first run to be slower.
