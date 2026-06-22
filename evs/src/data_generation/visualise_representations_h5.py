# pylint: disable=too-many-locals, too-many-statements, too-many-branches, invalid-name
# TODO(asude): clean pylint

"""
@brief Script for visualizing EVS data stored in h5 files.

@file visualize_representations_h5.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import argparse
import glob
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from matplotlib import animation
from matplotlib.patches import Circle
from data_generation.tools.e2frame import events_to_frame, events_to_timesurface


def parse_args():
    """
    Parses the input arguments.
    @return: Parsed argument struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Path to directory where the processed files are stored",
        required=True,
    )
    parser.add_argument(
        "--recording_name",
        type=str,
        help="Name of the EVS recording",
        required=True,
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        help="Name of the camera which is being processed",
        required=True,
    )
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path where the output video shall be stored",
        required=True,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        help="Scale factor of the velocity quiver",
        required=False,
        default=3,
    )

    args = parser.parse_args()

    return args


def visualize_representations(
    source_dir: str,
    target_path: str,
    scale_factor: float,
) -> None:
    """Create a movie from the representations, and localisation + velocity data. Saves the video at the target path.
    Args:
        `source_dir` (str): source directory, where all the .mat files are saved
        `target_path` (str): path where the movie sould be stored
        `scale_factor` (float): Scaling factor of the velocity quiver
    Note:
        This script might take a long time depending on the size and number of the images
    Returns:
        None
    """

    def init():
        ax1.set_ylim(0, 720)
        ax1.set_xlim(0, 1280)
        ax1.invert_xaxis()
        ax1.invert_yaxis()

        ax2.invert_xaxis()
        ax2.invert_yaxis()

        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.set_frame_on(False)

        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)

        return []

    # Define a function to update the plot for each frame
    def update(frame):
        ax1.clear()
        ax2.clear()

        position = frame["position"]  # Extract the row data from the frame dictionary
        velocity = frame["velocity"]
        radius = frame["radius"]
        image = frame["image"]  # Extract the image data from the frame dictionary
        recording = frame["recording"]
        camera = frame["camera"]
        frame_id = frame["frame_id"]
        traj_id = frame["traj_id"]

        ax1.imshow(image, animated=True)

        true_circle = Circle(
            (position[0], position[1]),
            radius=radius,
            color="orange",
            fill=False,
            linewidth=1,
        )

        ax1.quiver(
            position[0],
            position[1],
            velocity[0],
            velocity[1],
            color="orange",
            alpha=0.5,
            units="xy",
            scale=scale_factor,
            angles="xy",
        )

        ax1.add_patch(true_circle)

        ax1.set_title(f"Recording: {recording} Camera: {camera} Traj ID: {traj_id} Frame ID: {frame_id}")

        ax2.imshow(image, origin="lower")
        ax2.scatter(position[0], position[1], marker="x", color="orange")
        ax2.set_xlim(position[0] - 50, position[0] + 50)
        ax2.set_ylim(position[1] - 50, position[1] + 50)
        ax2.invert_yaxis()

        true_circle = Circle(
            (position[0], position[1]),
            radius=radius,
            color="orange",
            fill=False,
            linewidth=3,
            animated=True,
        )

        ax2.add_patch(true_circle)
        ax2.set_title("Zoom")

        return [true_circle]

    # Extract sequence names
    label_files = sorted(glob.glob(f"{source_dir}/*_label.h5"))

    for label_file_name in label_files:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7), gridspec_kw={"width_ratios": [2, 1]})

        # Initialise empty frames
        frames = []

        event_file_name = label_file_name.replace("_label", "")

        # read h5 files
        h5_label = h5py.File(label_file_name, "r")
        h5_events = h5py.File(event_file_name, "r")

        splitted_list = label_file_name.split("/")
        recording = splitted_list[-3]
        camera = splitted_list[-2]
        frame_id = splitted_list[-1].split("_")[2] + "_" + splitted_list[-1].split("_")[3]
        traj_id = splitted_list[-1].split("_")[1]

        mp4_file_name = f"traj_{traj_id}_{recording}_{camera}_{frame_id}.mp4"
        target_path_new = os.path.join(target_path, mp4_file_name)
        if os.path.exists(target_path_new):
            continue

        # Iterate through every trajectory and save video
        for idx in tqdm(range(len(h5_label["points"])), desc="Creating video {}".format(mp4_file_name)):
            ms_start = h5_events["ms_to_idx"][idx]
            if idx == len(h5_label["points"]) - 1:
                events_x = h5_events["events/x"][ms_start:].T
                events_y = h5_events["events/y"][ms_start:].T
                events_p = h5_events["events/p"][ms_start:].T
                events_t = h5_events["events/t"][ms_start:].T

            else:
                ms_end = h5_events["ms_to_idx"][idx + 1]

                events_x = h5_events["events/x"][ms_start:ms_end].T
                events_y = h5_events["events/y"][ms_start:ms_end].T
                events_p = h5_events["events/p"][ms_start:ms_end].T
                events_t = h5_events["events/t"][ms_start:ms_end].T

            events = np.stack((events_t, events_x, events_y, events_p), axis=1)

            # Load from .mat file
            frame_histogram = events_to_frame(events)
            frame_surface = events_to_timesurface(events)
            velocity = h5_label["velocities"][idx]
            position = h5_label["points"][idx]
            radius = h5_label["radius"][idx]

            # Create image array
            image = np.zeros_like(frame_histogram)
            image[0:2] = frame_histogram[0:2]
            image[2:3] = frame_surface[0:1]
            image = np.clip(image, 0, 1)
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)

            frame_data = {
                "velocity": velocity,
                "image": image,
                "radius": radius,
                "position": position,
                "recording": recording,
                "camera": camera,
                "traj_id": traj_id,
                "frame_id": frame_id,
            }
            frames.append(frame_data)

        # Create the animation using FuncAnimation with blit=True for efficiency
        start_time = time.time()
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            blit=True,
            interval=50,
        )

        ani.save(
            target_path_new,
            writer="ffmpeg",
            dpi=100,
            savefig_kwargs={"transparent": True, "facecolor": "none"},
        )
        end_time = time.time()
        print(f"Total amount of time passed: {end_time-start_time:.2f}")


def main():
    """
    Main script.
    """
    args = parse_args()
    source_dir = os.path.join(args.source_dir, args.recording_name, args.camera_name)
    if not os.path.isdir(os.path.dirname(args.target_path)):
        os.makedirs(args.target_path)
        print("Empty folder created for :", args.target_path)

    visualize_representations(
        source_dir,
        args.target_path,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
