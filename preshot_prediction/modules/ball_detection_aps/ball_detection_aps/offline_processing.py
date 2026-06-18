"""Ball detection for pre-recorded datasets."""
# pylint: skip-file
# Confidential, Copyright 2024, Sony AI, All rights reserved.

import json
from typing import Any, Dict

from ball_detection_aps.ball_detector import BallDetector
from dataset_replay.data_loader import DataLoader
from dataset_replay.read_params import ball_det_path

if __name__ == "__main__":
    data_loader = DataLoader()  # change dataset_replay/config.yaml
    ball_detector = BallDetector(ball_det_path)

    ball_labels: Dict[str, Dict[int, Any]] = {"frames": {}}
    for frame in data_loader:
        # save for later
        ball_labels["frames"][frame.number] = {}
        ball_labels["frames"][frame.number]["ball_positions"] = {}
        ball_labels["frames"][frame.number]["ball_radius"] = {}

        for i, (cam, img) in enumerate(frame.imgs.items()):
            if img is None:
                continue
            # compute mask
            masks = ball_detector.compute_mask(img, i)
            # extract ball positions
            ball_positions, ball_radius = ball_detector.extract_ball_positions(masks)
            ball_labels["frames"][frame.number]["ball_positions"][cam] = ball_positions
            ball_labels["frames"][frame.number]["ball_radius"][cam] = ball_radius

    # Save labels
    filename = "ball.json"
    with open(filename, "w") as fp:  # pylint: disable = unspecified-encoding
        json.dump(ball_labels, fp)
