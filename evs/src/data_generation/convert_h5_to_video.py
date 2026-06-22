# pylint: disable=too-many-locals, too-many-statements
# TODO(asude): clean pylint
"""
@brief This cscript visualizes .h5 events with corresponding labels.

@author Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import argparse
import glob
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import animation
from matplotlib.patches import Circle
from data_generation.tools.e2frame import events_to_frame, events_to_timesurface


def parse_args():
    """
    Parse the input arguments.

    @return: Parsed argument struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
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
        "--target_dir",
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

    parser.add_argument(
        "--start_idx",
        type=int,
        help="Video sequence number for start",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        help="Video sequence number for end",
        required=False,
        default=-1,
    )

    args = parser.parse_args()

    return args


def visualize_representations(
    source_dir: Path,
    target_dir: Path,
    scale_factor: float,
    start_idx: int,
    end_idx: int,
) -> None:
    """
    Create a movie from the representations, and localisation + velocity data. Saves the video at the target path.

    Args:
        `source_dir` (str): source directory, where all the .mat files are saved
        `target_dir` (str): path where the movie sould be stored
        `scale_factor` (float): Scaling factor of the velocity quiver
        `start_idx` (int): length of the movie. Full if the value is 0
        `end_idx` (int): length of the movie. Full if the value is -1
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
    label_files_sorted = sorted(glob.glob(f"{source_dir}/*_label.h5"))
    if end_idx == -1:
        end_idx = len(label_files_sorted)
    label_files = label_files_sorted[int(start_idx) : int(end_idx)]

    for label_file_name in label_files:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7), gridspec_kw={"width_ratios": [2, 1]})

        # Initialise empty frames
        frames = []

        event_file_name = label_file_name.replace("_label", "")

        # read h5 files
        h5_label = h5py.File(label_file_name, "r")
        h5_events = h5py.File(event_file_name, "r")

        splitted_list = Path(label_file_name).parts
        recording = splitted_list[-3]
        camera = splitted_list[-2]

        splitted_name = splitted_list[-1].split("_")
        frame_ids = f"{splitted_name[2]}_{splitted_name[3]}"
        traj_id = splitted_name[1]

        mp4_file_name = f"traj_{traj_id}_{recording}_{camera}_{frame_ids}.mp4"
        target_folder = target_dir / recording / camera

        target_folder.mkdir(parents=True, exist_ok=True)

        target_dir_new = target_folder / mp4_file_name
        if target_dir_new.exists():
            continue

        frame_id_start = int(splitted_list[-1].split("_")[2])
        frame_id_end = int(splitted_list[-1].split("_")[3])
        # Iterate through every frame and save video
        for idx in tqdm(
            range(len(h5_label["points"])),
            desc=f"Creating video {mp4_file_name}",
        ):
            frame_id = frame_id_start + idx * 2

            ms_start = h5_events["ms_to_idx"][idx]
            if idx == len(h5_label["points"]) - 1:
                x_events = h5_events["events/x"][ms_start:].T
                y_events = h5_events["events/y"][ms_start:].T
                p_events = h5_events["events/p"][ms_start:].T
                t_events = h5_events["events/t"][ms_start:].T

            else:
                ms_end = h5_events["ms_to_idx"][idx + 1]

                x_events = h5_events["events/x"][ms_start:ms_end].T
                y_events = h5_events["events/y"][ms_start:ms_end].T
                p_events = h5_events["events/p"][ms_start:ms_end].T
                t_events = h5_events["events/t"][ms_start:ms_end].T

            events = np.stack((t_events, x_events, y_events, p_events), axis=1)

            # Convert events into a representation
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

        assert frame_id_end == frame_id

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
            target_dir_new,
            writer="ffmpeg",
            dpi=100,
            savefig_kwargs={"transparent": True, "facecolor": "none"},
        )
        end_time = time.time()
        print(f"Total amount of time passed: {end_time-start_time:.2f}")


def main():  # NOQA D103
    """Main script."""
    args = parse_args()

    root_dir = Path(args.root_dir)
    recording_name = args.recording_name
    camera_name = args.camera_name
    target_dir = Path(args.target_dir)

    # Check and create target path for saving videos
    source_dir = root_dir / recording_name / camera_name
    target_dir.mkdir(parents=True, exist_ok=True)

    visualize_representations(
        source_dir,
        target_dir,
        scale_factor=args.scale_factor,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()
