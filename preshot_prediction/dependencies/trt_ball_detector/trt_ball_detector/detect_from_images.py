#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import enum
import os
import sys
import argparse

from nbformat import convert
from polars import last
import pytest
import trt_ball_detector.python_module as trt_ball_detector
import time 
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, List, Union

def _make_session(
    model_path: str,
    providers: List[Union[str, Tuple[str, dict]]],
) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Args:
        model_path: Path to .onnx model file.
        providers: List of execution providers (strings or tuples).

    Returns:
        ort.InferenceSession configured with graph optimization.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    print(f"[Model] {Path(model_path).name}")
    print(f"  Providers: {providers}")

    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
    )

def load_image_converter(engine_path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    engine = _make_session(engine_path, providers)
    return engine

def convert_image(engine, image: np.ndarray) -> np.ndarray:

    ie_in_name = engine.get_inputs()[0].name
    ie_in_shape = engine.get_inputs()[0].shape

    image=image[np.newaxis, np.newaxis, :, :]


    converted = engine.run(None, {ie_in_name: image})[0]
    return converted



def bgr_to_bayer_rggb(bgr_tensor:  np.ndarray) -> np.ndarray:
    """
    Converts a BGR tensor [B, 3, H, W] to a Bayer RGGB tensor [B, 1, H, W].
    Assumes input is normalized [0, 255].
    """
    # Ensure dimensions are even for 2x2 pattern
    original_ndim=bgr_tensor.ndim
    if bgr_tensor.ndim == 3:
        bgr_tensor = np.expand_dims(bgr_tensor, axis=0)  # Add batch dimension if missing
    if bgr_tensor.shape[-1] == 3:
        bgr_tensor = np.transpose(bgr_tensor, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    
    B, C, H, W = bgr_tensor.shape

    if H % 2 != 0 or W % 2 != 0:
        bgr_tensor = bgr_tensor[:, :, :H-H%2, :W-W%2]
        H, W = bgr_tensor.shape[-2:]

    # Create an empty single-channel tensor
    bayer = np.zeros((B, 1, H, W), dtype=np.float32)

    # Extract channels (assuming BGR order)
    blue  = bgr_tensor[ :, 0, :, :]
    green = bgr_tensor[ :, 1, :, :]
    red   = bgr_tensor[ :, 2, :, :]

    # Apply RGGB Pattern
    # Red: (0,0) in the 2x2 block
    bayer[:, 0, 0::2, 0::2] = red[:, 0::2, 0::2]
    
    # Green 1: (0,1)
    bayer[:, 0, 0::2, 1::2] = green[:, 0::2, 1::2]
    
    # Green 2: (1,0)
    bayer[:, 0, 1::2, 0::2] = green[:, 1::2, 0::2]
    
    # Blue: (1,1)
    bayer[:, 0, 1::2, 1::2] = blue[:, 1::2, 1::2]

    if original_ndim == 3:
        bayer = np.squeeze(bayer, axis=0)  # Remove batch dimension if it was added
    return bayer

def load_and_resize_jpeg(
    image_path: str,
    target_size: Tuple[int, int] = (1440, 1080),
) -> np.ndarray:
    """Load JPEG image and resize to target size.

    Args:
        image_path: Path to JPEG file.
        target_size: Target (width, height). Default: 1440×1080 (HiSilicon 1/4-res crop).

    Returns:
        BGR image as uint8 [H, W, 3].

    Raises:
        FileNotFoundError: If image cannot be loaded.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # h, w = target_size[1], target_size[0]  # OpenCV uses (h, w)
    # if img.shape[:2] != (h, w):
    #     img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img

def prepare_image_encoder_input(
    image_paths: List[str],
    imgsz: tuple[int, int] = (1440, 1080),
) -> Tuple[np.ndarray, List[str]]:
    """Prepare batch of images for image_encoder ONNX model.

    Loads JPEGs and prepares [N, 1, H, W] as float32 [0, 1] (Bayer RGGB).

    Args:
        image_paths: List of JPEG file paths.
        imgsz: Model input size (width, height).

    Returns:
        Tuple of:
            - images_batch: [N, 1, H, W] float32 Bayer images [0-1 range]
            - image_names: List of image filenames for output
    """
    images = []
    names = []

    for img_path in image_paths:
        print(f"  Loading: {Path(img_path).name}")
        bgr = load_and_resize_jpeg(img_path, target_size=imgsz)
        img = bgr_to_bayer_rggb(bgr)[0].astype(np.uint8)  # [ 1, H, W]
        # cv2.imshow("Loaded Image", img)
        # cv2.waitKey(0)
        images.append(img)
        names.append(Path(img_path).name)

    return images, names

def visualize_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray:
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
    scale = max(1, 640 // heatmap.shape[0])
    heatmap_color = cv2.resize(
        heatmap_color,
        (heatmap.shape[1] * scale, heatmap.shape[0] * scale),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color=heatmap_color[80:-80,:,:]

    return heatmap_color

def decode(img, detector, parameters):

    results=detector.get_decoding_results()

    last_image=cv2.cvtColor(img,cv2.COLOR_BayerBG2BGR)

    print("Sequence ID: ", results[0].sequence_id)
    print("Confidence: ", results[0].confidences)
    print("Coordinates: ", results[0].ball_positions[0])
    print("Velocity: ", results[0].ball_velocities[0])
    print("Radius: ", results[0].radius[0])
    if parameters.copy_heatmap:
        heatmap=results[0].heatmap
        heatmap=visualize_heatmap(heatmap)

        hm_resized = cv2.resize(heatmap, (last_image.shape[1], last_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        last_image = cv2.addWeighted(last_image, 0.7, heatmap_color, 0.3, 0)

    # plot the coordinates and confidence
    for i, (conf, pos, vel, rad) in enumerate(zip(results[0].confidences, results[0].ball_positions, results[0].ball_velocities, results[0].radius)):
        if conf > 0.1:  # confidence threshold
            x, y = int(pos[0]*last_image.shape[1]), int(pos[1]*last_image.shape[0])
            rad=int(rad*last_image.shape[1])  # scale radius to image size
            cv2.circle(last_image, (x, y), rad, (0, 255, 0), 2)
            cv2.putText(last_image, f"{conf:.2f}", (x+25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # draw array for velocity
            vel_x, vel_y = int(vel[0]*last_image.shape[1]), int(vel[1]*last_image.shape[0])
            cv2.arrowedLine(last_image, (x, y), (x+vel_x, y+vel_y), (255, 0, 0), 2)

    cv2.imshow("Ball Detection", last_image)
    if cv2.waitKey(0)==27:  # ESC key to exit
        cv2.destroyAllWindows()
        return False
    return True
def detect(image_paths):
    images_batch, image_names = prepare_image_encoder_input(image_paths, imgsz=(1440, 1080))

    TEST_IMAGE_CONVERTER=False
    if TEST_IMAGE_CONVERTER:
        converter=load_image_converter("models/image_converter.onnx")
    else:
        converter=None

    parameters=trt_ball_detector.ball_detector_parameters()
    parameters.onnx_engine_path="/mnt/tokyo_nas/shared_nas/models/ball_detector/zero_copy/model.onnx"

    detector=trt_ball_detector.ball_detector()
    detector.initialize(parameters)

    for i,img in enumerate(images_batch):
        if converter is not None:
            converted_img=convert_image(converter, img).astype(np.uint8)
            converted_img=converted_img[0, :, :, :].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            cv2.imshow("Converted Image", converted_img)
            cv2.waitKey(0)

        if detector.encode_images(i,[img]):
            if not decode(img, detector, parameters):
                return

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="ONNX inference on JPEG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "images",
        nargs="*",
        help="JPEG image paths (supports wildcards)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.images:
        print("[ERROR] No images specified")
        parser.print_help()
        exit(-1)

    # Expand wildcard paths
    image_paths = []
    for pattern in args.images:
        expanded = list(Path(".").glob(pattern))
        if not expanded:
            # Try absolute path
            expanded = [Path(pattern)] if Path(pattern).exists() else []
        image_paths.extend([str(p) for p in expanded if p.suffix.lower() in (".jpg", ".jpeg")])

    if not image_paths:
        print(f"[ERROR] No valid JPEG images found in: {args.images}")
        exit(-1)


    image_paths = sorted(image_paths)
    print(f"\n[✓] Found {len(image_paths)} JPEG image(s)")
    for p in image_paths:
        print(f"    - {Path(p).name}")
    detect(image_paths)
