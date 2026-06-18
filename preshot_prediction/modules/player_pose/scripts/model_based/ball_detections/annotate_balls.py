from csv import writer
import json
import os

from utilities import list_files
import argparse
import numpy as np 
import cv2
from data_processing.video_aligner.video_meta import VideoFileMeta
import ball_detector.python_module as ball_detector
import ball_detection_aps.python_module as ball_detection_aps
from ament_index_python.packages import get_package_share_directory


def rgb2bayer(bgr) -> np.ndarray:
    """implementation for rgb2bayer
    # https://github.com/guochengqian/TENet/issues/5
    Converts from a RGB representation to Bayer8 representation
    """
    bayer = np.zeros((bgr.shape[0], bgr.shape[1]))
    bayer[0::2, 1::2] = bgr[0::2, 1::2, 1]
    bayer[0::2, 0::2] = bgr[0::2, 0::2, 2]
    bayer[1::2, 1::2] = bgr[1::2, 1::2, 0]
    bayer[1::2, 0::2] = bgr[1::2, 0::2, 1]
    bayer = bayer.astype(np.uint8)
    return bayer.reshape((bayer.shape[0], bayer.shape[1], 1))

class BallDetectorImageAugmentation:
    def __init__(self, hsv_multiplier=[1,0,1.1],hsv_offset=[0,0,0]):
        self.hsv_multiplier=np.array(hsv_multiplier)
        self.hsv_offset=np.array(hsv_offset)

    def augment_image(self, image, mask):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hsv_ball=hsv.copy()
        indicies=np.where(mask.reshape(-1)!=0)
        shape=hsv_ball.shape
        hsv_ball=hsv_ball.reshape(-1,3)
        hsv_ball[indicies]=np.clip(hsv_ball[indicies]*self.hsv_multiplier+self.hsv_offset,0,255)
        hsv_ball=hsv_ball.reshape(shape)
        rgb_ball=cv2.cvtColor(hsv_ball,cv2.COLOR_HSV2BGR)

        return rgb_ball

class VideoBallAnnotationTool:
    def __init__(self):
        config_path=os.path.join(get_package_share_directory("ball_detection_aps"),"parameters","tyo01.yaml")
        params=ball_detection_aps.BallDetectionParameters()
        params.initialize(config_path)
        self.detector = ball_detector.BallDetector()
        self.detector.set_parameters(params)
        self.detector.set_cuda_device_id(0)

        self.augmentor=BallDetectorImageAugmentation(hsv_multiplier=[0,0,1],hsv_offset=[0,0,0])

    def process_video(self, video_frames_path:str, output:str):
        ball_annotations = []
        video_path=video_frames_path.replace(".frames","")

        with open(video_frames_path, "r") as f:
            frames=[int(f.strip()) for f in f.readlines()]

        # get camera name
        camera_name=video_frames_path.split("_aps")[1].split("_video")[0]
        log_name=os.path.basename(video_frames_path).split("_aps")[0]
        dirname=os.path.dirname(video_frames_path)
        triangulations_path=os.path.join(dirname, f"{log_name}_multiball_triangulation.ace")

        meta_dict={
            'video_file': os.path.basename(video_path),
            'multiball_triangulation': os.path.basename(triangulations_path),
            'config': "tyo01_player",
            'camera_name': f"aps{camera_name}",
            'log_name': log_name,
            'initial_seq_id': frames[0],
            'triangulation_offset': 0,
        }
        valid_annotations=0

        meta=VideoFileMeta("")
        meta.dir_name=dirname
        meta.load_from_dict(meta_dict)

        if output is not None:
            output_video_path=os.path.join(output, os.path.basename(meta.video_file)+"_augmented.mp4")
            writer=cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (meta.width, meta.height))
            print(f"Initialized video writer with path: {output_video_path}")
        else:
            writer=None

        meta.load_triangulations()

        print(f"Triangulation keys: {list(meta.triangulations.keys())[0]}")

        meta.triangulation_offset=frames[0]-list(meta.triangulations.keys())[0] 

        print(f"Extracting ball annotations from video {video_path}")
        while True:
            image, frame_index, timestamp=meta.read_image(undistort=False)
            if image is None:
                break

            annotation, mask = self.annotate_frame(meta, image, frames[frame_index], frame_index, timestamp)

            if writer is not None:
                if mask is not None:
                    image=self.augmentor.augment_image(image, mask)
                    # cv2.imshow("Augmented Image", image)
                    # cv2.waitKey(0)
                writer.write(image)
            
            if annotation is not None:
                ball_annotations.append(annotation)
                if len(annotation["detected_balls"])>0:
                    valid_annotations += 1

        if writer is not None:
            writer.release()
            print(f"Released video writer for path: {output_video_path}")
            exit()
        print(f"Extracted {len(ball_annotations)} annotations with {valid_annotations} valid ball detections.")
        return ball_annotations
    
    def find_closest_detection(self, detections, target_location):
        min_distance = float('inf')
        if np.isnan(target_location[0]) or np.isnan(target_location[1]):
            return None
        closest_detection = [target_location[0], target_location[1], 10]  # Default to target location with a small radius if no detections are close enough
        for detection in detections:
            distance = np.sqrt((detection[0] - target_location[0]) ** 2 + (detection[1] - target_location[1]) ** 2)
            if distance < min_distance and distance< 100:
                min_distance = distance
                closest_detection = detection
        return closest_detection
    def annotate_frame(self, meta, image, seq_id, frame_index, timestamp):
        seq_id= seq_id + meta.triangulation_offset
        ball_visible=False
        if seq_id not in meta.triangulations:
            print(f"Sequence ID {seq_id} not found in triangulations. Skipping annotation for this frame.")
            return None, None
        frame = meta.triangulations[seq_id]


        self.detector.set_bgr_image(image)
        balls=np.array(self.detector.detect_balls(None),dtype=int)
        mask=np.array(self.detector.get_detection_mask())
        

        game_id=os.path.basename(meta.video_file).split('.')[0]
        image_size=(meta.width, meta.height)
        detected_balls=[]
        if len(balls)>0:
            detection=self.find_closest_detection(balls, frame["ball_projection"])
            if detection is not None:
                ball_location_x=float(detection[0]/image_size[0]) if detection is not None else 0.0
                ball_location_y=float(detection[1]/image_size[1]) if detection is not None else 0.0
                width=float(detection[2]/image_size[0]) *2.5
                height=float(detection[2]/image_size[1]) *2.5
                detected_balls.append({
                    "x": ball_location_x,
                    "y": ball_location_y,
                    "width": width,
                    "height": height
                })
        annotation = {
            "frame_id": f"{frame_index:04d}",
            "frame_path": f"{game_id}/{frame_index:04d}.jpg",
            "video_id": game_id,
            "frame_number": frame_index,
            "timestamp": timestamp,
            "detected_balls": detected_balls,
            "occlusion_level": "none",
            "sport": "table_tennis",
            "metadata": {
                "camera_angle": meta.camera_name,
                "lighting": "unknown",
                "weather": "unknown"
            }
        }
        if False:
            # Display the frame and wait for user input to annotate ball location
            if ball_visible:
                cv2.circle(image, (int(ball_location_x*image_size[0]), int(ball_location_y*image_size[1])), 10, (0, 255, 0), 2)
            cv2.imshow("Frame", image)
            # cv2.imshow("MASK", mask)
            key = cv2.waitKey(2)

            if key == ord('b'):  # Press 'b' to annotate ball location
                x, y, w, h = cv2.selectROI("Frame", image, fromCenter=False, showCrosshair=True)
                annotation["ball_visible"] = True
                annotation["ball_location"] = {"x": int(x), "y": int(y), "radius": 8}  # Assuming a fixed radius for simplicity
        return annotation, mask

    def save_annotations(self, annotations, path):
        with open(path, 'w') as f:
            json.dump(annotations, f, indent=4)
            print(f"Saved annotations to {path}")

def process_directory(path: str, output: str) -> None:
    """Process all video files in a directory."""
    files=list_files(path,'_video.avi.frames')
    tool = VideoBallAnnotationTool()
    for file in files:
        print(f"Processing {file}")
        annotations = tool.process_video(file, output)
        tool.save_annotations(annotations, file.replace("_video.avi.frames", "_ball_annotations.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline trajectory overlay visualizer for pre-recorded videos"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the game which contains video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the video",
    )
    args = parser.parse_args()

    process_directory(args.path, args.output)
