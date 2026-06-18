#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
import sys

import pytest
import trt_ball_detector.python_module as trt_ball_detector
import time 
import cv2

import numpy as np
def visualize_heatmap(
    heatmap: np.ndarray,
) -> None:
    """Visualize decoder heatmap output."""

    # Normalize to [0, 255] for visualization
    hm_min = heatmap.min()
    hm_max = heatmap.max()
    if hm_max > hm_min:
        heatmap_norm = ((heatmap - hm_min) / (hm_max - hm_min) * 255).astype(
            np.uint8
        )
    else:
        heatmap_norm = (heatmap * 255).astype(np.uint8)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Upscale for better visibility if needed
    if heatmap.shape[0] < 256:
        scale = max(1, 256 // heatmap.shape[0])
        heatmap_color = cv2.resize(
            heatmap_color,
            (heatmap.shape[1] * scale, heatmap.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imshow("Heatmap Visualization", heatmap_color)
    cv2.waitKey(0)


def test_engine():
    parameters=trt_ball_detector.ball_detector_parameters()
    parameters.onnx_engine_path="/mnt/tokyo_nas/shared_nas/models/ball_detector/zero_copy/model.onnx"
    parameters.batch_size=1


    detector=trt_ball_detector.ball_detector()
    detector.initialize(parameters)

    image=np.random.randint(0, 256, size=(1080, 1440), dtype=np.uint8)  # bayer 8 image
    for i in range(100):
        detector.encode_images(0,[image])
    t1=time.time()
    N=100
    for i in range(N):
        detector.encode_images(0,[image])
    t2=time.time()
    print(f"Encoding time for {N} runs: {(t2-t1)*1e3/N:.2f} milliseconds")
    results=detector.get_decoding_results()
    heatmap=results[0].heatmap
    # display it using opencv
    if parameters.copy_heatmap:
        visualize_heatmap(heatmap)
if __name__ == "__main__":
    test_engine()
