# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Example script to demonstrate racket pose correction using different estimators."""
# pylint: skip-file

import time
import h5py
import numpy as np
import os
import argparse
from enum import Enum
import matplotlib.pyplot as plt
from racket_pose_estimation.pose_correction.racket_pose_corrector import RacketPoseCorrector, CorrectorConfig
import ace_interfaces.msg
import rclpy
import rclpy.node
import ace_yaml as yaml

from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory


class DataKeysEnum(Enum):
    """Enum for keys used in HDF5 data loading."""

    SEQUENCE_NUMBERS = "sequence_numbers"
    BALL_TRAJECTORY = "ball_trajectory"
    PLAYER_POSES = "player_poses"
    RACKET_POSES = "racket_poses"
    STYLE_VECTOR = "style_vector"


def load_game_state_from_hdf5(
    hdf5_filepath: str,
    game_id: str,  # pylint: disable=redefined-outer-name
    rally_id: str,  # pylint: disable=redefined-outer-name
    start_seq_num: int = None,
    end_seq_num: int = None,
    modulus: int = 10,  # pylint: disable=redefined-outer-name
):
    """Load game state from HDF5 file."""
    sequence_numbers = []
    ball_triangulation = []
    player_poses = []
    racket_poses = []
    print(f"Loading data from {hdf5_filepath}\nGame ID={game_id} - Rally ID={rally_id}")
    with h5py.File(hdf5_filepath, "r") as hdf5_experiment:
        rally = hdf5_experiment[f"{game_id}/{rally_id}"]

        # log_prefix = rally["sensors/aps_ball_triangulation"].attrs["log_identifier"]

        start_seq_num = rally["labels"]["sequence_number"][0] if start_seq_num is None else start_seq_num
        end_seq_num = rally["labels"]["sequence_number"][-1] if end_seq_num is None else end_seq_num

        player_columns = {
            "keypoints": 3,
            "confidences": 1,
            "projection_error": 1,
        }
        racket_columns = {
            "position": 3,
            "orientation": 4,
            "orientation_error": 1,
            "orientation_confidence": 1,
            "reprojection_err": 1,
        }

        start_seq_num = int(np.floor(start_seq_num / modulus) * modulus)
        end_seq_num = int(np.ceil(end_seq_num / modulus) * modulus)

        if (
            "aps_player_pose_estimation" not in rally["sensors"]
            or "aps_ball_triangulation" not in rally["sensors"]
            or "aps_racket_pose_estimation" not in rally["sensors"]
        ):
            print("Missing sensor data")
            return None
        aps_ball_triangulation = rally["sensors"]["aps_ball_triangulation"]
        aps_player_pose_estimation = rally["sensors"]["aps_player_pose_estimation"]
        aps_racket_pose_estimation = rally["sensors"]["aps_racket_pose_estimation"]
        aps_ball_triangulation_seq = aps_ball_triangulation["sequence_number"][:]
        aps_player_pose_estimation_seq = aps_player_pose_estimation["sequence_number"][:]
        aps_racket_pose_estimation_seq = aps_racket_pose_estimation["sequence_number"][:]

        for seq_num in range(start_seq_num, end_seq_num, modulus):
            idx = np.where(aps_ball_triangulation_seq == seq_num)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]
            sequence_numbers.append(aps_ball_triangulation["sequence_number"][idx])
            vals = np.array([np.column_stack(aps_ball_triangulation["positions"][idx])[0]])[0]
            ball_triangulation.append(vals)

            length = np.sum([player_columns[k] for k in player_columns])  # pylint: disable=consider-using-dict-items
            idx = np.where(aps_player_pose_estimation_seq == seq_num)[0]
            if len(idx) == 0:
                vals = np.array([[np.nan] * length] * 17)
                player_poses.append(vals)
            else:
                idx = idx[0]
                data = [
                    aps_player_pose_estimation[c][idx][0] for c in player_columns
                ]  # # pylint: disable=redefined-outer-name
                vals = np.array([np.column_stack(data)])[0]
                player_poses.append(vals)

            length = np.sum([racket_columns[k] for k in racket_columns])  # pylint: disable=consider-using-dict-items
            idx = np.where(aps_racket_pose_estimation_seq == seq_num)[0]
            if len(idx) == 0:
                racket_poses.append([np.nan] * length)
            else:
                idx = idx[0]
                data = [aps_racket_pose_estimation[c][idx][0] for c in racket_columns]
                vals = np.array([np.concatenate(data)])[0]
                racket_poses.append(vals)
    return {
        DataKeysEnum.SEQUENCE_NUMBERS: np.array(sequence_numbers),
        DataKeysEnum.BALL_TRAJECTORY: np.array(ball_triangulation),
        DataKeysEnum.PLAYER_POSES: np.array(player_poses),
        DataKeysEnum.RACKET_POSES: np.array(racket_poses),
    }


def get_game_info(hdf5_filepath: str):
    """Get game information from HDF5 file."""
    games = {}  # pylint: disable=redefined-outer-name
    try:
        with h5py.File(hdf5_filepath, "r") as hdf5_experiment:
            for game in hdf5_experiment.keys():
                rallies = []
                for rally in hdf5_experiment[game].keys():
                    rallies.append(rally)
                games[game] = rallies
    except OSError as err:
        print(f"Error reading HDF5 file {hdf5_filepath}: {err}")
        return None
    return games


def quaternion_to_euler_stable(q, prev_euler=None):
    """
    Convert quaternion to Euler angles (degrees), minimizing discontinuities ("flipping").
    If prev_euler is provided, the output will be adjusted to be as close as possible to prev_euler.
    """
    r = R.from_quat(q)
    euler = r.as_euler("xyz", degrees=True)
    if prev_euler is not None:
        # Minimize angle jumps by wrapping to nearest equivalent
        for i in range(3):
            diff = euler[i] - prev_euler[i]
            if diff > 180:
                euler[i] -= 360
            elif diff < -180:
                euler[i] += 360
    return euler


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Racket pose correction example script")
    parser.add_argument("--data", type=str, help="Path to the HDF5 data file")
    parser.add_argument("--modulus", type=int, default=10, help="Sampling rate modulus")
    parser.add_argument("--publish", action="store_true", help="Publish to ACE monitor topics")
    parser.add_argument("--rally_id", type=int, default=-1, help="Rally ID to process (default: -1 for all rallies)")
    args = parser.parse_args()
    path = args.data
    modulus = args.modulus
    publish_ace_monitor = args.publish
    rally_id = args.rally_id

    dt = modulus / 1e3

    # Load configuration from YAML file
    estimators = {}
    config_base_path = os.path.join(get_package_share_directory("racket_pose_estimation"), "parameters", "corrector")
    files = os.listdir(config_base_path)
    for file in files:
        if file.startswith("config_") and file.endswith(".yaml"):
            config_path = os.path.join(config_base_path, file)
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            config = CorrectorConfig.from_dict(config_data)
            corrector = RacketPoseCorrector(config)
            estimators[file.replace(".yaml", "")] = corrector

    rclpy.init()
    node = rclpy.node.Node("player_pose")
    player_pose_pub = node.create_publisher(ace_interfaces.msg.PlayerPose, "/sensors/player_pose/gt", 1)
    ball_pub = node.create_publisher(ace_interfaces.msg.PointsWithCovariance, "/sensors/ball_triangulation/points", 1)

    gt_racket_pub = node.create_publisher(ace_interfaces.msg.RacketPose, "/sensors/racket_pose/gt", 1)
    pred_racket_pubs = {}
    for estimator_name, corrector in estimators.items():
        pred_racket_pubs[estimator_name] = node.create_publisher(
            ace_interfaces.msg.RacketPose, f"/sensors/racket_pose/corrected/{estimator_name}", 1
        )

    ball_msg = ace_interfaces.msg.PointsWithCovariance()
    racket_msg = ace_interfaces.msg.RacketPose()
    player_msg = ace_interfaces.msg.PlayerPose()
    point = ace_interfaces.msg.PointWithCovariance()
    ball_msg.points.append(point)

    games = get_game_info(path)

    time_data = []
    raw_data = []
    corrected_data = {}

    count = 0

    prev_euler = {}

    for game in games:
        for rally in games[game]:
            count += 1
            if count != rally_id and rally_id != -1:
                continue
            print(f"Processing game {game} - rally {rally}")
            data = load_game_state_from_hdf5(path, game_id=game, rally_id=rally, modulus=modulus)
            for estimator_name, corrector in estimators.items():
                corrector.reset()

            counter = 0

            last_time = -1

            for seq_id, ball, player, racket in zip(
                data[DataKeysEnum.SEQUENCE_NUMBERS],
                data[DataKeysEnum.BALL_TRAJECTORY],
                data[DataKeysEnum.PLAYER_POSES],
                data[DataKeysEnum.RACKET_POSES],
            ):
                counter += 1
                # if counter<250:
                #     continue
                # if counter>300:
                #     break
                # extract racket pose confidence values
                quat_confidence = 1 - min(1, racket[7] / 10)  # Orientation confidence: 1 - orientation_error
                overall_confidence = racket[
                    8
                ]  # Overall confidence, reflects the number of cameras observing the racket
                pos_confidence = 1 - min(
                    1, racket[9] / 10
                )  # Position confidence: 1 - reprojection error normalized to 10px

                dt = (seq_id - last_time) * 1e-3
                if last_time < 0:
                    dt = 0

                last_time = seq_id

                # Fill in the 6D pose data structure used by the corrector

                quat = racket[3 : 3 + 4]
                pose = {
                    "delta_time": dt,
                    "position": racket[:3],
                    "orientation": np.array([quat[0], quat[1], quat[2], quat[3]]),  # Reorder to x, y, z, w
                    "pos_confidence": pos_confidence * overall_confidence,
                    "quat_confidence": quat_confidence * overall_confidence,
                }
                # (Optional) Ensure quaternion has a positive scalar part.
                # If not, then flip the entire quat. Note that q==-q
                # This is used for plotting to avoid flip/flop visualization issues
                if pose["orientation"][3] < 0:
                    pose["orientation"] = -pose["orientation"]

                # Fill in data for plotting
                time_data.append(seq_id)
                if not np.any(np.isnan(pose["orientation"])):
                    pose["euler"] = quaternion_to_euler_stable(pose["orientation"], prev_euler.get("raw", None))
                    prev_euler["raw"] = pose["euler"]
                else:
                    pose["euler"] = [np.nan, np.nan, np.nan]
                    prev_euler["raw"] = None
                raw_data.append(pose)

                for estimator_name, corrector in estimators.items():
                    corrected_pose = corrector.correct_pose(pose.copy())  # Perform pose correction.

                    # Fill in data for plotting
                    if corrected_pose:
                        corrected_pose["euler"] = quaternion_to_euler_stable(
                            corrected_pose["orientation"], prev_euler.get(estimator_name, None)
                        )
                        prev_euler[estimator_name] = corrected_pose["euler"]
                        corrected_data.setdefault(estimator_name, []).append(corrected_pose)
                        if publish_ace_monitor:
                            # To publish in ACE monitor
                            racket_msg.position.x = float(corrected_pose["position"][0])
                            racket_msg.position.y = float(corrected_pose["position"][1])
                            racket_msg.position.z = float(corrected_pose["position"][2])

                            racket_msg.orientation.x = float(corrected_pose["orientation"][0])
                            racket_msg.orientation.y = float(corrected_pose["orientation"][1])
                            racket_msg.orientation.z = float(corrected_pose["orientation"][2])
                            racket_msg.orientation.w = float(corrected_pose["orientation"][3])

                            pred_racket_pubs[estimator_name].publish(racket_msg)
                    else:
                        corrected_data.setdefault(estimator_name, []).append(pose)
                        prev_euler[estimator_name] = None

                if publish_ace_monitor:
                    # To publish in ACE monitor
                    for i in range(17):
                        player_msg.keypoints[i].x = float(player[i, 0])
                        player_msg.keypoints[i].y = float(player[i, 1])
                        player_msg.keypoints[i].z = float(player[i, 2])
                        player_msg.confidences[i] = float(player[i, 3])
                        player_msg.projection_error[i] = float(player[i, 4])
                    player_pose_pub.publish(player_msg)

                    point.position.x = float(ball[0])
                    point.position.y = float(ball[1])
                    point.position.z = float(ball[2])
                    ball_pub.publish(ball_msg)

                    racket_msg.header = player_msg.header
                    racket_msg.position.x = float(pose["position"][0])
                    racket_msg.position.y = float(pose["position"][1])
                    racket_msg.position.z = float(pose["position"][2])

                    racket_msg.orientation.x = float(pose["orientation"][0])
                    racket_msg.orientation.y = float(pose["orientation"][1])
                    racket_msg.orientation.z = float(pose["orientation"][2])
                    racket_msg.orientation.w = float(pose["orientation"][3])
                    racket_msg._tracked = True

                    gt_racket_pub.publish(racket_msg)
                    time.sleep(dt * 4)

    rclpy.shutdown()

    # Plot the results
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    # 3D trajectory comparison
    ax = fig.add_subplot(341, projection="3d")
    ax.plot(
        [x["position"][0] for x in raw_data],
        [x["position"][1] for x in raw_data],
        [x["position"][2] for x in raw_data],
        "g-",
        label="Raw",
        linewidth=3,
    )
    for estimator_name, poses in corrected_data.items():
        ax.plot(
            [x["position"][0] for x in poses],
            [x["position"][1] for x in poses],
            [x["position"][2] for x in poses],
            label=f"Corrected ({estimator_name})",
            linewidth=2,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Comparison")
    ax.legend()

    axis_names = ["x", "y", "z", "w"]
    for i in range(3):
        axes[0, i + 1].plot([x["position"][i] for x in raw_data], "r-", label="Raw", alpha=0.7, linewidth=1)
        for estimator_name, poses in corrected_data.items():
            axes[0, i + 1].plot([x["position"][i] for x in poses], label=f"Corrected ({estimator_name})", linewidth=2)
        axes[0, i + 1].set_title(f"Position {axis_names[i].upper()}")
        axes[0, i + 1].legend()
        axes[0, i + 1].grid(True, alpha=0.3)

    for i in range(3):
        axes[1, i].plot([x["euler"][i] for x in raw_data], "r-", label="Raw", alpha=0.7, linewidth=1)
        for estimator_name, poses in corrected_data.items():
            axes[1, i].plot([x["euler"][i] for x in poses], label=f"Corrected ({estimator_name})", linewidth=2)
        axes[1, i].set_title(f"Euler.{axis_names[i].upper()}")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    for i in range(4):
        axes[2, i].plot([x["orientation"][i] for x in raw_data], "r-", label="Raw", alpha=0.7, linewidth=1)
        for estimator_name, poses in corrected_data.items():
            axes[2, i].plot([x["orientation"][i] for x in poses], label=f"Corrected ({estimator_name})", linewidth=2)
        axes[2, i].set_title(f"Quaternion.{axis_names[i].upper()}")
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
