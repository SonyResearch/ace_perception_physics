"""Ball detector in python."""
# pylint: skip-file
# Confidential, Copyright 2025, Sony AI, All rights reserved.

# opencv
import cv2 as cv

# math stuff
import numpy as np

# parameter files
import ace_yaml as yaml


class BallDetector:
    def __init__(self, ball_detection_params):
        # ball detection parameters
        f = open(ball_detection_params, "r")
        self.params = yaml.full_load(f)
        f.close()

        # auxiliary variables
        self.cams = self.params["camera_names"]
        self.num_cams = len(self.cams)
        self.motion_queue_len = 5
        self.motion_filter_low = self.params["motion_filter"]["lower_boundary"]
        self.motion_filter_high = self.params["motion_filter"]["upper_boundary"]
        self.motion_img_queues = [None] * self.num_cams
        for i in range(self.num_cams):
            self.motion_img_queues[i] = list()

    def compute_mask(self, bgr_img, i):
        # blur image
        blurred_bgr_img = cv.GaussianBlur(
            bgr_img,
            (self.params["color_filter"]["blur_kernel_size"], self.params["color_filter"]["blur_kernel_size"]),
            0.0,
        )

        # color mask
        hsv_img = cv.cvtColor(blurred_bgr_img, cv.COLOR_BGR2HSV)
        color_mask = cv.inRange(
            hsv_img,
            np.array(self.params["color_filter"]["hsv_lower_boundary"]),
            np.array(self.params["color_filter"]["hsv_upper_boundary"]),
        )

        # compute motion mask
        if self.params["motion_filter"]["enable"]:
            gray_img = cv.cvtColor(blurred_bgr_img, cv.COLOR_BGR2GRAY)
            self.motion_img_queues[i].append(gray_img)
            delayed_gray_img = self.motion_img_queues[i][0]
            if len(self.motion_img_queues[i]) > self.motion_queue_len:
                delayed_gray_img = self.motion_img_queues[i][0]
                self.motion_img_queues[i].pop(0)

            diff_gray_img = cv.absdiff(delayed_gray_img, gray_img)
            motion_mask = cv.inRange(diff_gray_img, self.motion_filter_low, self.motion_filter_high)

            # combine mask
            combined_mask = cv.bitwise_and(color_mask, motion_mask)
            return combined_mask

        return color_mask

    def extract_ball_positions(self, mask):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        ball_positions = list()
        ball_radius = list()
        for i in range(len(contours)):
            U = cv.arcLength(contours[i], False)
            A = cv.contourArea(contours[i], False)
            if A < 0.01:
                continue

            circularity = 4.0 * np.pi * A / U**2
            if circularity >= self.params["appearance_filter"]["min_circularity_ratio"]:
                center, radius = cv.minEnclosingCircle(contours[i])

                if radius > self.params["appearance_filter"]["min_radius"]:
                    ball_positions.append(center)
                    ball_radius.append(radius)

        return ball_positions, ball_radius
