# Confidential, Copyright 2024, Sony AI, All rights reserved.
# pylint: skip-file

import click
import subprocess
import time
from python_helpers.keyboard_input import KBHit
from ament_index_python.packages import get_package_share_directory
import os


def process_rosbag(cluster, path, perform_player_extraction, config):
    """Main entry point"""
    keyboard = KBHit()

    name = os.path.split(path)[-1]
    print(f"Processing for {name}")

    images = path + "_images"
    if cluster != "":
        images = os.path.join(images, "cluster" + cluster)
    print("Playing from: ", images)

    aps_package = get_package_share_directory("aps")
    racket_pose_package = get_package_share_directory("racket_pose_estimation")

    aps_play_publisher_path = os.path.join(aps_package, "tools", "aps_play_publisher")
    racket_to_csv_path = os.path.join(racket_pose_package, "scripts", "racket_to_csv.py")

    image_playback_cmd = f"python3 {aps_play_publisher_path} --path {images}"
    csv_recorder_cmd = f"python3 {racket_to_csv_path} --clusters {cluster} --name {name}"
    ros_bag_cmd = f"ros2 bag play {path} -r 0.7"
    racket_estimator_cmd = []
    player_pose_estimator_cmd = []
    racket_estimator_cmd.append(
        f"ros2 launch racket_pose_estimation racket_pose_estimation.launch.py config:={config} cluster:={cluster}"
    )
    if perform_player_extraction:
        player_pose_estimator_cmd.append(
            f"ros2 launch player_pose player_pose_estimation.launch.py config:={config} cluster:={cluster}"
        )

    image_playback = subprocess.Popen(image_playback_cmd.split(" "), start_new_session=True)
    csv_recorder = subprocess.Popen(csv_recorder_cmd.split(" "), start_new_session=True)
    racket_estimators = []
    player_estimators = []
    for est in racket_estimator_cmd:
        print(f"Running {est}")
        racket_estimators.append(subprocess.Popen(est.split(" "), start_new_session=True))
        time.sleep(1)
    for est in player_pose_estimator_cmd:
        print(f"Running {est}")
        player_estimators.append(subprocess.Popen(est.split(" "), start_new_session=True))
        time.sleep(1)
    time.sleep(10)  # wait until estimators are ready
    ros_bag = subprocess.Popen(ros_bag_cmd.split(" "), start_new_session=True)

    try:
        while True:
            if ros_bag.poll() is not None:
                break
            if keyboard.kbhit():
                key = keyboard.getch()
                if key == "q":
                    break
            time.sleep(1)
    except:
        pass

    print("Killing processes")
    ros_bag.kill()
    ros_bag.wait()
    time.sleep(3)

    for est in racket_estimators:
        est.kill()
        est.wait()
    for est in player_estimators:
        est.kill()
        est.wait()

    image_playback.kill()
    image_playback.wait()
    csv_recorder.kill()
    csv_recorder.wait()
    subprocess.Popen("pkill player".split(" "))
    subprocess.Popen("pkill racket".split(" "))

    keyboard.set_normal_term()


@click.command()
@click.option("--cluster", help="Cluster name", required=True, type=str)
@click.option("--path", help="ros bag path", required=True, type=str)
@click.option("--config", help="config name", required=True, type=str)
def main(cluster, path, config):
    """Main entry point"""
    process_rosbag(cluster, path, config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
