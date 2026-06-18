import os

from tqdm import tqdm

def list_files(path, extension):
    """TODO"""
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extension):
                file_list.append(os.path.join(root, file))

    return file_list

def convert_to_dict(obj):
    """
    Convert a generic object into a dict
    """
    if isinstance(obj, list):
        return [convert_to_dict(x) for x in obj]

    if not hasattr(obj, "__module__"):
        return obj
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()

    result = {
        attr: convert_to_dict(getattr(obj, attr))
        for attr in dir(obj)
        if not attr.startswith("_") and not callable(getattr(obj, attr))
    }
    result.update({"type": type(obj).__name__})

    return result

def process_players(path):
    from player_pose.pose_corrector import PlayerPoseCorrector
    from ace_interfaces.msg import PlayerPose as PlayerPoseMsg
    from player_pose import datalogger_utils
    paths=list_files(path,"player_pose.ace")
    config={}
    config["process_noise_pos"]=0.1
    config["process_noise_vel"]=0.1
    config["measurement_noise_pos"]=0.5

    msg=PlayerPoseMsg()

    def write_csv_row(fp, msg, predicted):
        row=[]
        row.append(str(msg.header.sequence_number))
        for i in range(17):
            row.append(str(msg.keypoints[i].x))
            row.append(str(msg.keypoints[i].y))
            row.append(str(msg.keypoints[i].z))
            row.append(str(msg.confidences[i]))
            row.append(str(msg.projection_error[i]))
        row.append("1" if predicted else "0")
        fp.append(row)

    header=["sequence_number"]
    for i in range(17):
        header.append(f"kp_{i}_x")
        header.append(f"kp_{i}_y")
        header.append(f"kp_{i}_z")
        header.append(f"conf_{i}")
        header.append(f"proj_err_{i}")
    header.append("predicted")
    for p in paths:
        print("Processing " + p)

        raw_csv=p.replace(".ace","_raw.csv")
        filtered_csv=p.replace(".ace","_filtered.csv")

        if os.path.exists(filtered_csv):
            print(f"Skipping {p} since {filtered_csv} already exists")
            continue

        raw_csv_rows=[]
        filtered_csv_rows=[]


        corrector=PlayerPoseCorrector(config)
        data_list = datalogger_utils.read_log_file(p)
        prev_seq_number = -1
        data_list = tqdm(data_list, desc=f"Frames ({os.path.basename(p)})", unit="frame")
        for i, feature in enumerate(data_list):
            feature_dict = convert_to_dict(feature)
            if feature_dict is None or feature_dict["type"]!="player_frame_features":
                continue

            msg.header.sequence_number = feature_dict["sequence_number"]

            if prev_seq_number >= 0 and feature_dict["sequence_number"] != prev_seq_number+10:
                print("Warning: Missing frames detected between {} and {}".format(prev_seq_number, feature_dict["sequence_number"]))

            dt=(feature_dict["sequence_number"]-prev_seq_number)*0.001 if prev_seq_number>=0 else 0.01
            prev_seq_number=feature_dict["sequence_number"]
            if dt<=0:
                continue

            predicted=False
            if len(feature_dict["estimated_players"])==0:
                corrector.predict(dt)
                corrector.get_filtered_pose(msg)
                predicted=True
            else:
                player=feature_dict["estimated_players"][0]

                for i in range(17):
                    msg.keypoints[i].x = float(player["keypoints"][i][0])
                    msg.keypoints[i].y = float(player["keypoints"][i][1])
                    msg.keypoints[i].z = float(player["keypoints"][i][2])
                    msg.confidences[i] = float(player["confidences"][i])
                    msg.projection_error[i] =float(player["projection_error"][i])

                write_csv_row(raw_csv_rows, msg, False)
                corrector.correct_pose(msg)
            write_csv_row(filtered_csv_rows, msg,predicted)

        raw_csv_fp=open(raw_csv,"w")
        filtered_csv_fp=open(filtered_csv,"w")

        raw_csv_fp.write(",".join(header) + "\n")
        filtered_csv_fp.write(",".join(header) + "\n")

        raw_csv_rows=sorted(raw_csv_rows, key=lambda x:int(x[0]))
        filtered_csv_rows=sorted(filtered_csv_rows, key=lambda x:int(x[0]))
        for row in raw_csv_rows:
            raw_csv_fp.write(",".join(row) + "\n")
        for row in filtered_csv_rows:
            filtered_csv_fp.write(",".join(row) + "\n")



def process_rackets(path):
    from racket_pose_estimation.pose_correction.racket_pose_corrector import RacketPoseCorrector, CorrectorConfig
    from racket_pose_estimation import datalogger_utils
    from ament_index_python import get_package_share_directory
    from ace_interfaces.msg import RacketPoseEstimate 
    import ace_yaml as yaml
    import numpy as np

    paths=list_files(path,"racket_pose.ace")

    config_path = os.path.join(
        get_package_share_directory("racket_pose_estimation"), "parameters", "corrector", "config_mekf.yaml"
    )

    with open(config_path, "r", encoding="utf8") as file_handler:
        config_data = yaml.safe_load(file_handler)

    config = CorrectorConfig.from_dict(config_data)

    msg=RacketPoseEstimate()

    def write_csv_row(fp, msg:RacketPoseEstimate, predicted):
        row=[]
        row.append(str(msg.pose.header.sequence_number))
        row.append(str(msg.pose.position.x))
        row.append(str(msg.pose.position.y))
        row.append(str(msg.pose.position.z))
        row.append(str(msg.pose.orientation.x))
        row.append(str(msg.pose.orientation.y))
        row.append(str(msg.pose.orientation.z))
        row.append(str(msg.pose.orientation.w))
        row.append(str(msg.confidence))
        row.append(str(msg.projection_error))
        row.append(str(msg.orientation_error))
        row.append("1" if predicted else "0")
        fp.append(row)

    header=["sequence_number"]
    for field in ["position.x","position.y","position.z","orientation.x","orientation.y","orientation.z","orientation.w","confidence","projection_error","orientation_error","predicted"]:
        header.append(field)
    for p in paths:
        print("Processing " + p)

        raw_csv=p.replace(".ace","_raw.csv")
        filtered_csv=p.replace(".ace","_filtered.csv")

        if os.path.exists(filtered_csv):
            print(f"Skipping {p} since {filtered_csv} already exists")
            continue

        raw_csv_rows=[]
        filtered_csv_rows=[]


        corrector=RacketPoseCorrector(config)
        data_list = datalogger_utils.read_log_file(p)
        prev_seq_number=-1

        data_list = tqdm(data_list, desc=f"Frames ({os.path.basename(p)})", unit="frame")
        for i, feature in enumerate(data_list):
            feature_dict = convert_to_dict(feature)
            if feature_dict is None or feature_dict["type"]!="RacketFrameFeatures"  :
                continue
            msg.pose.header.sequence_number = feature_dict["sequence_number"]
            if prev_seq_number >= 0 and feature_dict["sequence_number"] != prev_seq_number+10:
                print("Warning: Missing frames detected between {} and {}".format(prev_seq_number, feature_dict["sequence_number"]))
            dt=(feature_dict["sequence_number"]-prev_seq_number)*0.001 if prev_seq_number>=0 else 0.01
            prev_seq_number=feature_dict["sequence_number"]
            if dt<=0:
                continue
            predicted=False
            if len(feature_dict["estimated_rackets"])==0:
                corrected_pose=corrector.predict(dt)
                predicted=True
            else:
                racket=feature_dict["estimated_rackets"][0]

                msg.pose.position.x = float(racket["position"][0])
                msg.pose.position.y = float(racket["position"][1])
                msg.pose.position.z = float(racket["position"][2])
                msg.pose.orientation.x = float(racket["orientation"][0])
                msg.pose.orientation.y = float(racket["orientation"][1])
                msg.pose.orientation.z = float(racket["orientation"][2])
                msg.pose.orientation.w = float(racket["orientation"][3])
                msg.confidence = np.clip(np.nan_to_num(racket["orientation_confidence"], nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
                msg.projection_error = np.clip(np.nan_to_num(racket["reprojection_err"], nan=0.0, posinf=0.0, neginf=0.0), 0.0, 100.0)
                msg.orientation_error = np.clip(np.nan_to_num(racket["orientation_error"], nan=0.0, posinf=0.0, neginf=0.0), -2.0, 2.0)

                quat_confidence = 1 - min(1, msg.orientation_error / 10)  # Orientation confidence: 1 - orientation_error
                overall_confidence = (
                    msg.confidence
                )  # Overall confidence, reflects the number of cameras observing the racket
                pos_confidence = 1 - min(
                    1, msg.projection_error / 10
                )  # Position confidence: 1 - reprojection error normalized to 10px

                pose = {
                    "delta_time": dt,
                    "position": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                    "orientation": [
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w,
                    ],
                    "pos_confidence": pos_confidence * overall_confidence,
                    "quat_confidence": quat_confidence * overall_confidence,
                }
                write_csv_row(raw_csv_rows, msg, False)
                corrected_pose=corrector.correct_pose(pose)
            if corrected_pose:
                msg.pose.position.x = float(corrected_pose["position"][0])
                msg.pose.position.y = float(corrected_pose["position"][1])
                msg.pose.position.z = float(corrected_pose["position"][2])

                msg.pose.orientation.x = float(corrected_pose["orientation"][0])
                msg.pose.orientation.y = float(corrected_pose["orientation"][1])
                msg.pose.orientation.z = float(corrected_pose["orientation"][2])
                msg.pose.orientation.w = float(corrected_pose["orientation"][3])
            write_csv_row(filtered_csv_rows, msg, predicted)

        raw_csv_fp=open(raw_csv,"w")
        filtered_csv_fp=open(filtered_csv,"w")

        raw_csv_fp.write(",".join(header) + "\n")
        filtered_csv_fp.write(",".join(header) + "\n")

        raw_csv_rows=sorted(raw_csv_rows, key=lambda x:int(x[0]))
        filtered_csv_rows=sorted(filtered_csv_rows, key=lambda x:int(x[0]))
        for row in raw_csv_rows:
            raw_csv_fp.write(",".join(row) + "\n")
        for row in filtered_csv_rows:
            filtered_csv_fp.write(",".join(row) + "\n")


def main():
    # process_players("/mnt/tokyo_nas/shared_nas/logs/tyo01/20260206/smash_vs_kawamata_match_0/smash_vs_kawamata_game_0/")
    # process_rackets("/mnt/tokyo_nas/shared_nas/logs/tyo01/20260206/smash_vs_kawamata_match_0/smash_vs_kawamata_game_0/")
    process_players("/mnt/tokyo_nas/shared_nas/logs/tyo02/OfflineProcessing/")
    process_rackets("/mnt/tokyo_nas/shared_nas/logs/tyo02/OfflineProcessing/")

if __name__ == "__main__":
    main()