#!/bin/bash

"""
@brief Example script for labeling pro player data in h5 format.

@file run_pro_player_label_scripts.sh
@author Asude Aydin (asude.aydin@sony.com)
@date 2026
@version 0.0
@copyright Confidential, Copyright 2026, Sony AI, All rights reserved.
"""

# Define variables for the root directory and camera name
ROOT_DIR="<path_to_root_directory>"  # Replace with the actual path to the root directory
CAM_NAME="<camera_name>"  # Replace with the actual camera name

# Run the first Python script
python3 label_pro_player_data.py --root_dir="$ROOT_DIR" --cam_name="$CAM_NAME"

# Check if the first script was successful
if [ $? -eq 0 ]; then
  echo "First script completed successfully."

  # Run the second Python script
  python3 label_pro_player_data_events.py --root_dir="$ROOT_DIR" --cam_name="$CAM_NAME"

  # Check if the second script was successful
  if [ $? -eq 0 ]; then
    echo "Second script completed successfully."
  else
    echo "Second script failed."
  fi

else
  echo "First script failed. Aborting the second script."
fi
