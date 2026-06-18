# Confidential, Copyright 2024, Sony AI, All rights reserved.
# pylint: skip-file

from process_rosbag import process_rosbag
import click
from typing import List


# Add here pathes for ros bags. Images should be split into cluster_a/cluster_b folders corresponding to cluster assignments if there are clusters
recordings: List[str] = [
    # Entry Example:
    # "/media/yamen/Backup/Set1/20240507/20240507_Isaki_Li_set_1",
]


@click.command()
@click.option("--clusters", help="Clusters name (e.g. '_a,_b')", required=True, type=str)
@click.option("--config", help="config name", required=True, type=str)
@click.option("--player_pose", help="extract player pose too", is_flag=True)
def main(clusters: str, player_pose, config):
    """Main entry point"""
    clusters_list = []

    if clusters == "":
        clusters_list = [""]
    else:
        clusters_list = clusters.split(",")

    for r in recordings:
        for c in clusters_list:
            print("===================================")
            print(f"Processing: {r}{c}")
            process_rosbag(c, r, player_pose, config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
