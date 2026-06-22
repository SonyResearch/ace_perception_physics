"""
@brief This script visualizes .h5 events with corresponding labels by generating videos around the RoI.

@author Asude Aydin
@date 2024
@version 0.0
@copyright SPDX-License-Identifier: MIT
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
from data_generation.tools.e2frame import events_to_timesurface


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
        "--target_dir",
        type=str,
        help="Path where the output video shall be stored",
        required=True,
    )

    parser.add_argument(
        "--sequence_name",
        type=str,
        help="Name of the sequence to be converted into a video. \
        If unspecified, all sequences in a folder will be converted.",
        required=False,
    )

    parser.add_argument(
        "--accumulation_time",
        type=int,
        help="Duration of event accumulation time to be visualized (in ms).",
        required=False,
        default=5,
    )

    parser.add_argument(
        "--label_offset",
        type=int,
        help="Label offset to handle time synchronization issues.",
        required=False,
        default=0,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plots backprojected original APS trinagulations.",
    )

    args = parser.parse_args()

    return args


def visualize_representations(
    source_dir: Path,
    target_dir: Path,
    sequence_name: str,
    accumulation_time: int,
    debug: bool = False,
    label_offset: int = 0,
) -> None:
    """
    Create a movie from the representations, and localisation + velocity data. Saves the video at the target path.

    Args:
        `source_dir` (str): source directory, where all the .mat files are saved
        `target_dir` (str): path where the movie sould be stored
        `sequence_name` (str): Specific sequence name to be converted into video.
        `accumulation_time` (int): Duration of event accumulation time to be visualized.
        `debug` (bool): Debug mode will plot the original raw APS triangulation backprojections.
        `label_offset` (int): This is shift labels in time for better time synchronization.
    Note:
        This script might take a long time depending on the size and number of the images
    Returns:
        None
    """

    def init():
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        return []

    # Define a function to update the plot for each frame
    def update(frame):
        ax.clear()

        # Extract the row data from the frame dictionary
        position = frame["position"]
        radius = frame["radius"]
        velocity = frame["velocity"]
        image = frame["image"]
        frame_id = frame["frame_id"]

        ax.imshow(image, origin="lower", animated=True)
        ax.scatter(position[0], position[1], marker="x", color="orange")
        ax.set_xlim(position[0] - 50, position[0] + 50)
        ax.set_ylim(position[1] - 50, position[1] + 50)
        ax.invert_yaxis()

        true_circle = Circle(
            (position[0], position[1]),
            radius=radius,
            color="orange",
            fill=False,
            linewidth=3,
            animated=True,
        )

        if debug:
            position_orig = frame[
                "position_orig"
            ]  # Extract the row data from the frame dictionary
            radius_orig = frame["radius_orig"]

            orig_circle = Circle(
                (position_orig[0], position_orig[1]),
                radius=radius_orig,
                color="green",
                fill=False,
                linewidth=3,
                animated=True,
            )

            ax.add_patch(orig_circle)

        ax.quiver(
            position[0],
            position[1],
            velocity[0],
            velocity[1],
            color="orange",
            alpha=0.5,
            units="xy",
            scale=3,
            angles="xy",
        )
        ax.add_patch(true_circle)
        ax.set_title("Zoom, frame id: {}".format(frame_id))

        return [true_circle]

    # Extract sequence names
    label_files_sorted = sorted(glob.glob(f"{source_dir}/*_label.h5"))

    for label_file_name in label_files_sorted:
        if sequence_name is not None:
            if sequence_name != Path(label_file_name).stem.replace("_label", ""):
                continue

        # Create figure
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(7, 7),
        )

        # Initialise empty frames
        frames = []

        event_file_name = label_file_name.replace("_label", "")

        # read h5 files
        h5_label = h5py.File(label_file_name, "r")
        h5_event = h5py.File(event_file_name, "r")

        mp4_file_name = f"{Path(event_file_name).stem}.mp4"
        target_folder = target_dir

        target_folder.mkdir(parents=True, exist_ok=True)

        target_dir_new = target_folder / mp4_file_name
        if target_dir_new.exists():
            continue

        ms_duration = (
            min(len(h5_label["points"]), len(h5_event["ms_to_idx"]))
            // accumulation_time
        )

        assert (
            not debug or accumulation_time == 1
        ), "Accumulation time must be 1 when debug mode is activated."

        # Iterate through every frame and save video
        for idx in tqdm(
            range(ms_duration - 1),
            desc=f"Creating video {mp4_file_name}",
        ):
            frame_idx = idx * accumulation_time
            label_idx = idx * accumulation_time + accumulation_time - 1 + label_offset

            if label_idx < 0 or label_idx >= len(h5_label["points"]):
                continue

            ms_start = h5_event["ms_to_idx"][frame_idx]
            ms_end = h5_event["ms_to_idx"][frame_idx + accumulation_time]

            # Skip frame if there are no milliseconds in between
            if ms_start == ms_end:
                print("Frame with no events found and skipped.")
                continue

            events_x = h5_event["events/x"][ms_start:ms_end].T
            events_y = h5_event["events/y"][ms_start:ms_end].T
            events_p = h5_event["events/p"][ms_start:ms_end].T
            events_t = h5_event["events/t"][ms_start:ms_end].T

            assert (events_t[-1] - events_t[0]) < accumulation_time * 1e3
            events = np.stack((events_t, events_x, events_y, events_p), axis=1)

            velocity = h5_label["velocities"][label_idx]
            position = h5_label["points"][label_idx]
            radius = h5_label["radius"][label_idx]

            # Convert events into a representation
            frame_surface = events_to_timesurface(events)

            # Create image array
            image = np.zeros_like(frame_surface)
            image[0:2] = frame_surface[0:2]
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)

            frame_data = {
                "velocity": velocity,
                "image": image,
                "radius": radius,
                "position": position,
                "frame_id": label_idx,
            }

            if debug:
                position_orig = h5_label["points_orig"][label_idx]
                radius_orig = h5_label["radius_orig"][label_idx]

                frame_data["position_orig"] = position_orig
                frame_data["radius_orig"] = radius_orig

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
            target_dir_new,
            writer="ffmpeg",
            dpi=100,
            savefig_kwargs={"transparent": True, "facecolor": "none"},
        )
        end_time = time.time()
        print(f"Total amount of time passed: {end_time-start_time:.2f}")


def main():  # NOQA D103
    """Main script"""
    args = parse_args()

    root_dir = Path(args.root_dir)
    target_dir = Path(args.target_dir)

    # Check and create target path for saving videos
    source_dir = root_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    visualize_representations(
        source_dir,
        target_dir,
        sequence_name=args.sequence_name,
        accumulation_time=args.accumulation_time,
        debug=args.debug,
        label_offset=args.label_offset,
    )


if __name__ == "__main__":
    main()
