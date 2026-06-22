#!/bin/bash
"""
@brief Bash script for labeling all professional player data.

@file run_pro_player_label_scripts_all.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2026
@version 0.0
@copyright Confidential, Copyright 2026, Sony AI, All rights reserved.
"""

# Define variables for the root directory containing sets and camera names
SETS_DIR="<path_to_sets_directory>"  # Replace with the actual path to the sets directory
CAM_NAMES=("evs00050028" "evs00050027" "evs00050026" "evs00050034")
PLOT_DIR="<path_to_plot_directory>"  # Replace with the actual path to the plot directory

# Iterate over each subdirectory in the SETS_DIR
for ROOT_DIR in "$SETS_DIR"/*; do
  if [ -d "$ROOT_DIR" ]; then
    echo "Processing directory: $ROOT_DIR"

    # Iterate over each camera name
    for CAM_NAME in "${CAM_NAMES[@]}"; do
      echo "Processing camera: $CAM_NAME"

      # Run the first Python script
      python3 label_pro_player_data.py --root_dir="$ROOT_DIR" --cam_name="$CAM_NAME" --plot_dir="$PLOT_DIR"

      # Check if the first script was successful
      if [ $? -eq 0 ]; then
        echo "First script completed successfully for $ROOT_DIR and $CAM_NAME."

        # Run the second Python script
        python3 label_pro_player_data_events.py --root_dir="$ROOT_DIR" --cam_name="$CAM_NAME"

        # Check if the second script was successful
        if [ $? -eq 0 ]; then
          echo "Second script completed successfully for $ROOT_DIR and $CAM_NAME."
        else
          echo "Second script failed for $ROOT_DIR and $CAM_NAME."
        fi
      else
        echo "First script failed for $ROOT_DIR and $CAM_NAME. Aborting the second script."
      fi
    done
  else
    echo "$ROOT_DIR is not a directory. Skipping."
  fi
done
