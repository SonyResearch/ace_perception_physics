
- `process_raw_files.py` script was developed by Claudio Fanconic to label EVS data with ball positions and velocity using the APS triangulations and the CMA physics optimizer. This method has not showed scalability and robustness to labeling professional player recording. Furthermore, the dataset format has shown incompatabilities with the intended use of data and experiments to be followed. Hence we develop two pipelines:

1. Converting the data from Claudio's .mat files to .h5 format without the need for processing the raw data from scratch for compatability with experiments.
2. Developing a new observation-based data labeling pipeline without utilizing any physics optimizer.  

# Data Reformatting for Standalone EVS Ball Localization and Velocity Prediction
This folder contains the code to reformat the original dataset provided by Claudio from `.mat` format to `.h5` format for flexible on-the-go event representation generation during training. In terms of content, the following three files were removed due to corruption or short sequence durations:
1) 621726.mat & 621728.mat were removed from 20230614_120843/evs00050027 due to sequence length 1
2) 556470.mat was removed from 20230614_120123/evs00050034 due to corruption

You can find the final dataset containing raw files, metadata, and `.mat` files along with the reformatted `.h5` files in NAS at [`/shared/project_ace/processed_dataset_perception/evs_pos_vel_prediction/evs_pos_vel_prediction_reformatted.zip`](https://sainas01.eu.sony.com:5001/sharing/Qh5osOfr9).

## Folder Structure

A rough outline of the folder structure can be found below for the dataset stored at NAS.

    └── evs_pos_vel_prediction_reformatted
        └── metadata
            ├── 20230321_zrh00.yaml
            └── 20230614_tyo01.yaml
        ├── raw
            ├── intermediate_files # Intermediate folder, where triggers.txt and points_3D.csv files are stored per recording
            └── raw_files          # Raw folder, where .raw event files per camera view and .bag recordings are stored
        ├── processed_files_1ms_corrected
            └──recording_id
                └──camera_id
                    ├── ...
                    └── frame_id.mat
        └── processed_files_1ms_corrected_reformatted
            └──recording_id
                └──camera_id
                    ├── ...
                    └── sequence_id.h5


## How to run
To run this script, we recommend you run it on a copy of the original dataset as this scripts renames the files.

`source ~/ws/src/project_ace/install/setup.bash`

You don't need to have the folders `processed_files_1ms_corrected` and `processed_files_1ms_corrected_reformatted` under the same root directory as in the example above, they can be anywhere.
However, metadata and raw folders need to be under the same root directory. The `raw` folder should have the following structure:
```
<your_path>
    ├── raw_files               # Input folder, where the recording .raw files come in
        ├── ...        # Folder of the recording data
        └── 20230320_042069  
            ├── evs                     # Subfolder named evs that contains the raw files from the EVS cameras
                ├── ...
                └── evs00050246.raw  
            └── ros_bag_file.db3        # Rosbag file, where the locations from the APS are stored
    └── intermediate_files  
        ├── ...         # Folder of the recording data
        └── 20230320_042069
            ├── points_3d.csv  
            └── triggers.txt  
    ...
```

Example command for reformatting would look as follows:

```
python process_raw_files.py \
    --raw_dir <path_to_processed_files_1ms_corrected> # path to root folder containing recordings of mat data
    --data_dir <path_to_evs_pos_vel_prediction_reformatted> # path to containing metadata and raw folders
    --new_dir <path_to_processed_files_1ms_corrected_reformatted> # path to root folder for saving reformatted dataset in h5 format
```

## Tests

### Unittest

You are advised to run the following unittest that runs 2 simple checks after reformatting the dataset that compares it against the original dataset and does simple checks within the new data format.

```
python test/test_suite.py <root_dir_to_old_dataset> <root_dir_to_new_dataset>
```

### Qualitative Analysis

If you wish to visualize your final outputs, you can run the following command (careful, as it creates an animation with matplotlib it can take some time --> set `start_idx` and `end_idx` for selecting spesific sequences. Careful, these arguments have a different use case than in Claudio's code.):

```
python visualise_representations_h5.py
    --source_dir <your_path>/processed_files_1ms_corrected_reformatted
    --recording_name 20230320_042069
    --camera_name evs00050244
    --target_path <your_path>/Videos
    --start_idx 0
    --end_idx -1
```

If you have multiple recordings with multiple cameras to run, I recommend creating a simple bash file, like my `run_test_visualization.sh` file, and let it run.

```
bash run_test_visualization.sh
```

## Assumptions, Notes and Observations
Here we display some problems, observations, etc. that I observed in the original dataset:
1. In visualizations of some sequences, there were missing frames, ie. jumps in the ball position that might be interfering with the RNN performance.
2. Tokyo recordings have inconsistent time differnece between APS frames and trigger differences don't match these. As a result there is an inconsistent shift between the actual ball position and the label in some Tokyo sequences. These are to be fixed by bootstrapping at a later stage.

Note: See sprint 29.02.2024 for a visual example of these 2 points.


# Professional player labeling

This tool was specifically developed for labeling player data from Victas recordings of 10/05/24 and University recordings of 28/05/24 and 29/05/24. The download links for the raw files used can be found below along with the explanation of the scripts and required folder format for processing.

**The overarching aim**: Given raw EVS recordings and segmented rally regions in pytorch file format, we generate h5 files of events and time-synchronized corresponding labels to be used in future experiments for ball position and velocity estimation from EVS data.

This pipeline consists of two main scripts and some supporting ones as explained below:

- `label_pro_player_data.py`: This script processes the segmented rallies and converts them into h5 format compatible with EVS data. The essential pipeline consists of segmenting ball trajectories based on bounce/contact points, fitting a polynomial of degree 3 for interpolated ball positions and finite differenced velocity estimates, and finally stitching these segmentations into uninterrupted continous sequences that contain 2D back projections in the respective EVS camera frame.

- `label_pro_player_data_events.py`: This script should be run after the previous one. It uses the timestamps of labels in order to iterate through the raw events and fetch overlapping ones, which are then saved in h5 files containing events.

Example code for these can be found in the `run_pro_player_label_scripts.sh` and `run_pro_player_label_scripts_all.sh` bash scripts.

- `convert_h5_to_video_zoomed.py`: This script can be used to visualize the events and corresponding labels stored in h5 files.

Example code can be found in `run_pro_player_visualization.sh` bash script.

- `trim_h5_sequences.py`: This script can be used to trim h5 sequences from the start and end to remove shifted labels seen in visualizations. This requires a .csv file with the "Recording name", "Camera name", "Sequence Number", "Trim Start", and "Trim End".

## Folder Structure

```
<your_path>
    ├── recording_name      # Folder containing files specific to a recording (ie. set for Victas & Univ. players)
        ├── evs             # contains raw EVS recordings
        ├── labels          # contains torch format processed labels  
        ├── rosbag          # contains raw APS rosbags
        └── tyo01.yaml      # yaml file containing camera calibration matrices

    ...
```

## To access data

- For access to raw APS recordings in rosbag format, segmented rallies, and pre-processed ball positions ask for access to the Google Drive folder `TT_MM_game_dataset` from Naoya Takahashi.
- For access to the raw EVS recordings for matches that took place on 10/05, 28/05, and 29/05 ask for access to the Google Drive folder from Fabian Schilling.
