#!/bin/bash

"""
@brief Example script for visualizing labeled pro player data in h5 format.

@file run_pro_player_visualizations.sh
@author Asude Aydin
@date 2024
@version 0.0
@copyright SPDX-License-Identifier: MIT
"""

REC_NAME="20240510_101340_uechi_vs_li_set1"

CAM_NAME="evs00050026"

python3 convert_h5_to_video_zoomed.py --root_dir="/path/to/recordings/VICTAS_2024-05-10_test_recording/$REC_NAME/h5/$CAM_NAME" \
--target_dir="/path/to/output/videos/VICTAS_2024-05-10_test_recording/$REC_NAME/$CAM_NAME" --accumulation_time=5 --debug --label_offset=-1

CAM_NAME="evs00050027"

python3 convert_h5_to_video_zoomed.py --root_dir="/path/to/recordings/VICTAS_2024-05-10_test_recording/$REC_NAME/h5/$CAM_NAME" \
--target_dir="/path/to/output/videos/VICTAS_2024-05-10_test_recording/$REC_NAME/$CAM_NAME" --accumulation_time=5 --debug

CAM_NAME="evs00050028"

python3 convert_h5_to_video_zoomed.py --root_dir="/path/to/recordings/VICTAS_2024-05-10_test_recording/$REC_NAME/h5/$CAM_NAME" \
--target_dir="/path/to/output/videos/VICTAS_2024-05-10_test_recording/$REC_NAME/$CAM_NAME/" --accumulation_time=5 --debug

CAM_NAME="evs00050034"

python3 convert_h5_to_video_zoomed.py --root_dir="/path/to/recordings/VICTAS_2024-05-10_test_recording/$REC_NAME/h5/$CAM_NAME" \
--target_dir="/path/to/output/videos/VICTAS_2024-05-10_test_recording/$REC_NAME/$CAM_NAME/" --accumulation_time=5 --debug
