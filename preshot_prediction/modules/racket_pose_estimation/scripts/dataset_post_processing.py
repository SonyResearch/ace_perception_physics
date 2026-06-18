# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""Helper script to process a dataset used for racket pose detection training"""

import argparse
import sys
import os
import cv2
import numpy as np
import pyautogui


class Processor:
    """Main processing class for the images"""

    def create_racket_extractor(self):
        """Create racket pose extractor"""
        import racket_pose_estimation.python_module as racket_pose  # pylint: disable=import-outside-toplevel

        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "racket_v8n.onnx")
        self.extractor = racket_pose.racket_inference()
        self.extractor.initialize(model_path)

    def __init__(self, db_path, rec_type, start_index) -> None:
        self.extractor = None
        # self.create_racket_extractor()
        self.db_path = db_path
        self.rec_type = rec_type

        self.rcnn_format = False
        self.append_keypoints = True
        self.append_position = False
        self.append_angles = True
        if self.rcnn_format:
            self.img_path = os.path.join(self.db_path, self.rec_type, "images", "")
            self.label_path = os.path.join(self.db_path, self.rec_type, "labels", "")
            self.annotation_path = os.path.join(self.db_path, self.rec_type, "annotations", "")
            self.pose_path = os.path.join(self.db_path, self.rec_type, "pose", "")
        else:
            self.img_path = os.path.join(self.db_path, "images", self.rec_type, "")
            self.label_path = os.path.join(self.db_path, "labels", self.rec_type, "")
            self.annotation_path = os.path.join(self.db_path, "annotations", self.rec_type, "")
            self.pose_path = os.path.join(self.db_path, "pose", self.rec_type, "")

        labels = Processor._list_files(self.label_path, ".txt")
        self.ids = sorted([os.path.basename(file).split(".")[0] for file in labels])
        self.current_id = start_index - 1
        self.current_image = None
        self.current_label = None
        self.hide = False
        cv2.namedWindow("Racket")
        while self.next():
            if self.current_image is not None:
                break

        cv2.setMouseCallback("Racket", self._on_mouse_callback)

    def remove(self):
        """Remove current label file"""
        self.current_label = []

        label_path = os.path.join(self.label_path, f"{self.ids[self.current_id]}.txt")
        with open(label_path, "w", encoding="utf8") as file:
            file.write("")

    def _on_mouse_callback(self, event, x, y, flags, param):  # pylint: disable=unused-argument, invalid-name
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_label) == 0:
                return
            label = self.current_label[0]
            x /= 640
            y /= 640
            center_x = x - label[1]
            center_y = y - label[2]

            label[1] += center_x
            label[2] += center_y
            for i in range(4):
                label[5 + 3 * i] += center_x
                label[5 + 3 * i + 1] += center_y

            label_path = os.path.join(self.label_path, f"{self.ids[self.current_id]}.txt")
            with open(label_path, "w", encoding="utf8") as file:
                file.write(" ".join(str(x) for x in label))

    @classmethod
    def _list_files(cls, path, ext):
        file_list = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(ext):
                    file_list.append(os.path.join(root, file))

        return file_list

    def _parse(self, label_id):
        img_path = os.path.join(self.img_path, label_id + ".jpg")
        label_path = os.path.join(self.label_path, label_id + ".txt")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            return None, None
        label = []

        with open(label_path, "r", encoding="utf8") as file:
            if file is None:
                return None, None
            lines = file.readlines()
            for line in lines:
                if line == "\n":
                    continue
                label.append(np.array(line.strip(" \n").split(" "), dtype=float))

        return image, label

    def delete(self, label_id):
        """Delete an image & label"""
        img_path = os.path.join(self.img_path, f"{label_id}.jpg")
        annotation_path = os.path.join(self.label_path, f"{label_id}.txt")

        try:
            os.remove(img_path)
            os.remove(annotation_path)
        except:  # pylint:disable=bare-except
            pass
        self.ids.pop(self.current_id)
        if self.current_id >= len(self.ids):
            self.current_id = len(self.ids) - 1

    def refresh(self):
        """Refresh current image"""
        self.current_image, self.current_label = self._parse(self.ids[self.current_id])
        self.hide = False
        return True

    def next(self, batch_size=1):
        """Move to next image"""
        if self.current_id >= len(self.ids) - batch_size:
            return False
        self.current_id += batch_size

        self.current_image, self.current_label = self._parse(self.ids[self.current_id])
        if len(self.current_label) != 0:
            image = np.array(self.current_image)
            width = image.shape[1]
            height = image.shape[0]
            label = self.current_label[0]
            center_x, center_y = label[3] * width / 2, label[4] * height / 2
            px_1, py_1 = label[1] * width - center_x, label[2] * height - center_y
            racket_window = cv2.getWindowImageRect("Racket")
            pyautogui.moveTo(px_1 + center_x + racket_window[0], py_1 + center_y + racket_window[1])
        self.hide = False

        return True

    def check_empty(self):
        """Check if a file is empty of labels, and remove it"""
        while len(self.current_label) == 0:
            print("Removed")
            self.delete(self.ids[self.current_id])
            self.refresh()

    def scan_next(self):
        """Scan until we find an image that is quite different from its label based on the detector"""
        while True:
            if not self.next():
                return
            if self.current_image is not None:
                break
        if self.extractor is None:
            return

        racket = self.extractor.detect(cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
        if racket is not None and len(self.current_label) == 0:
            print("Racket was detected, but label has no rackets")
            bbox = racket.bbox
            keypoints = racket.keypoints
            cv2.rectangle(
                self.current_image,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                thickness=2,
                color=(0, 255, 0),
            )

            point_1 = np.array((keypoints[0][0], keypoints[0][1]), dtype=int)
            point_2 = np.array((keypoints[1][0], keypoints[1][1]), dtype=int)
            point_3 = np.array((keypoints[2][0], keypoints[2][1]), dtype=int)
            point_4 = np.array((keypoints[3][0], keypoints[3][1]), dtype=int)

            cv2.line(
                self.current_image,
                np.array(point_1, dtype=int),
                np.array(point_2, dtype=int),
                (0, 255, 0),
                thickness=4,
            )
            cv2.line(
                self.current_image,
                np.array(point_3, dtype=int),
                np.array(point_4, dtype=int),
                (0, 0, 255),
                thickness=4,
            )

        elif racket is None and len(self.current_label) > 0:
            print(f"No racket detected but the image has {len(self.current_label)} rackets")
        else:
            self.scan_next()

    def back(self):
        """Move back one image"""
        if self.current_id <= 0:
            return False
        self.current_id -= 1

        self.current_image, self.current_label = self._parse(self.ids[self.current_id])
        self.hide = False
        return True

    def is_done(self):
        """Check if done"""
        return self.current_id >= len(self.ids)

    def draw(self):
        """Draw image"""
        if self.current_image is None:
            return False

        image = np.array(self.current_image)
        width = image.shape[1]
        height = image.shape[0]
        if not self.hide:
            for label in self.current_label:
                center_x, center_y = label[3] * width / 2, label[4] * height / 2

                px_1, py_1 = label[1] * width - center_x, label[2] * height - center_y
                px_2, py_2 = px_1 + 2 * center_x, py_1 + 2 * center_y

                cv2.rectangle(
                    image,
                    np.array((px_1, py_1), dtype=int),
                    np.array((px_2, py_2), dtype=int),
                    (255, 0, 0),
                    thickness=1,
                )

                index = 5

                if self.append_keypoints:
                    point_1 = np.array((label[index + 0] * width, label[index + 1] * height), dtype=int)
                    point_2 = np.array((label[index + 3] * width, label[index + 4] * height), dtype=int)
                    point_3 = np.array((label[index + 6] * width, label[index + 7] * height), dtype=int)
                    point_4 = np.array((label[index + 9] * width, label[index + 10] * height), dtype=int)
                    index += 3 * 4

                    cv2.circle(
                        image,
                        center=np.array(point_1, dtype=int),
                        color=(0, 255, 0),
                        radius=3,
                        thickness=1,
                    )
                    cv2.circle(
                        image,
                        center=np.array(point_3, dtype=int),
                        color=(255, 0, 0),
                        radius=3,
                        thickness=1,
                    )
                    cv2.line(
                        image,
                        np.array(point_1, dtype=int),
                        np.array(point_2, dtype=int),
                        (0, 255, 0),
                        thickness=1,
                    )
                    cv2.line(
                        image,
                        np.array(point_3, dtype=int),
                        np.array(point_4, dtype=int),
                        (0, 0, 255),
                        thickness=1,
                    )

                    center = ((point_1 + point_2) // 2 + (point_3 + point_4) // 2) // 2
                    axes = np.array(
                        (np.linalg.norm(point_1 - point_2) / 2, np.linalg.norm(point_3 - point_4) / 2), dtype=int
                    )
                    diff = point_2 - point_1
                    angle = np.arctan2(diff[1], diff[0]) * 180 / 3.1415

                    cv2.ellipse(image, center, axes, angle, 0, 360, (255, 255, 255), 1)
                    # cv2.circle(self.current_image,center,int(np.linalg.norm(point_1-point_2)//2),1,1)

                if self.append_position:
                    center = np.array([label[index + 0] * width, label[index + 1] * height], dtype=int)
                    cv2.circle(
                        image,
                        center=center,
                        color=(255, 0, 0),
                        radius=5,
                        thickness=4,
                    )
                    index += 2

                if self.append_angles:
                    angles = np.array(label[index : index + 3]) * 360 - 180
                    cv2.putText(
                        image,
                        f"Angles: {np.array(angles,dtype=int)}",
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (85, 196, 196),
                        1,
                        cv2.LINE_AA,
                    )

                    # x_axis=np.array([1,0,0])
                    # y_axis=np.array([0,1,0])
                    # z_axis=np.array([0,0,1])
                    # rotation=R.from_euler("xyz",angles,degrees=True).as_matrix()
                    # x_axis=np.array(np.matmul(rotation,x_axis)*100,dtype=int)
                    # y_axis=np.array(np.matmul(rotation,y_axis)*100,dtype=int)
                    # z_axis=np.array(np.matmul(rotation,z_axis)*100,dtype=int)

                    # cv2.line(image,
                    #         center,
                    #         center+np.array([x_axis[0],x_axis[1]]),
                    #         (0,0,255),
                    #         thickness=4,)
                    # cv2.line(image,
                    #         center,
                    #         center+np.array([y_axis[0],y_axis[1]]),
                    #         (0,255,0),
                    #         thickness=4,)
                    # cv2.line(image,
                    #         center,
                    #         center+np.array([z_axis[0],z_axis[1]]),
                    #         (255,0,0),
                    #         thickness=4,)

        cv2.putText(
            image,
            f"ID= {self.ids[self.current_id]}/{self.ids[-1]}",
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (85, 196, 196),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Racket", image)
        return True

    def process_key(self, key):
        """Process key"""
        if key == ord(" "):
            self.next()
        if key == ord("b"):
            self.back()
        if key == ord("s"):
            self.scan_next()
        elif key == ord("d"):
            self.delete(self.ids[self.current_id])
            self.refresh()
            # self.scan_next()
        elif key == ord("r"):
            self.remove()
            self.refresh()
        elif key == ord("q"):
            return False
        elif key == ord("v"):
            self.hide = not self.hide
        if key == 86:
            self.next(100)
        # self.check_empty()
        return True


def main(args):
    """Entrypoint"""
    processor = Processor(args.path, args.type, args.start)
    while not processor.is_done():
        processor.draw()
        key = cv2.waitKey(200)
        if not processor.process_key(key):
            break


if __name__ == "__main__":
    USAGE = """
    """
    # setup argument list
    parser = argparse.ArgumentParser(description="Dataset post-processing script.", usage=USAGE)
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        help="Training dataset location.",
        required=True,
    )
    parser.add_argument(
        "--type",
        dest="type",
        type=str,
        help="type of recording.",
        required=True,
    )
    parser.add_argument(
        "--start",
        dest="start",
        type=int,
        help="starting index.",
        default=0,
        required=False,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    # parse argument list
    try:
        parsed = parser.parse_args()
    except:  # pylint: disable = bare-except
        sys.exit(0)

    main(parsed)
