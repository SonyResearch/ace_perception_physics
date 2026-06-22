# Instructions for dataset labeling

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

- TODO
