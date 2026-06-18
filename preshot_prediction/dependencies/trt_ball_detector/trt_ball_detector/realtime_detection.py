
from __future__ import annotations

import argparse
from collections import deque
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import trt_ball_detector.python_module as trt_ball_detector
from calibration import python_module as calibration
from aps.image_subscriber import ManagedDisplayImagesContext
from rclpy.node import Node
import rclpy
import os
import cv2


# ---------------------------------------------------------------------------
# RealtimeInference class
# ---------------------------------------------------------------------------
class RealtimeInference(ManagedDisplayImagesContext):
    def __init__(self, window_size=6,config="tyo01_player"):
        calibration_params_path = os.path.join(
            "/shared",
            "camera_calibration",
            config + ".yaml",
        )


        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calibration_params_path):
            raise Exception("Failed to load configuration file")


        self.last_frame = 0

        camera_mask = ["aps23287107"]

        super().__init__(config, 200, 200, camera_mask=camera_mask,decode_bgr=False)

        self.parameters=trt_ball_detector.ball_detector_parameters()
        self.parameters.onnx_engine_path="/mnt/tokyo_nas/shared_nas/models/ball_detector/model.onnx"
        self.parameters.batch_size=4
        self.parameters.device_id=0

        self.detector=trt_ball_detector.ball_detector()
        self.detector.initialize(self.parameters)
        self.last_heatmap=None
        self.bayer_image=None


    def _on_image_arrived(self, index, image: np.ndarray, topic_name):
        if index != 0:
            return
        self.bayer_image=image
        bgr_image=cv2.cvtColor(image,cv2.COLOR_BayerBG2BGR)

        
        if self.detector.encode_images(0,[image]):
            results=self.detector.get_decoding_results()
            heatmap=results[0].heatmap
            heatmap=self.visualize_heatmap(heatmap)
            self.last_heatmap = heatmap

            hm_resized = cv2.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            bgr_image = cv2.addWeighted(bgr_image, 0.7, heatmap_color, 0.3, 0)

            for conf, pos in zip(results[0].confidences, results[0].ball_positions):
                if conf > 0.1:  # confidence threshold
                    x, y = int(pos[0]*bgr_image.shape[1]), int(pos[1]*bgr_image.shape[0])
                    cv2.circle(bgr_image, (x, y), 20, (0, 255, 0), 2)
                    cv2.putText(bgr_image, f"{conf:.2f}", (x+25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # Process dec_outs as needed...
        super()._on_image_arrived(index, bgr_image, topic_name)


        
    def visualize_heatmap(
        self,
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

        # cv2.imshow("Heatmap Visualization", heatmap_color)

        return heatmap_color


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    realtime_inference=RealtimeInference()
    while rclpy.ok():
        realtime_inference.render(force_update=False)
        if realtime_inference.last_heatmap is not None:
            cv2.imshow("Heatmap Visualization", realtime_inference.last_heatmap)
            # cv2.imshow("Bayer Image", realtime_inference.bayer_image)

    rclpy.shutdown()

if __name__ == "__main__":
    main()