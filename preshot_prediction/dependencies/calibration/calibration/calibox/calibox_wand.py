# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
Camera calibration toolbox - E-Wand detector
"""


class Detector:
    """
    A wand detector that identifies and processes ball detections in images.
    """

    def __init__(self, args):
        """
        Constructs the E-wand detector instance.

        @param args (argparse.Namespace): The command-line options containing parameters for the detector.
        """

    def get_object_points(self):
        """
        Retrieves the object points defined by the E-wand detector.

        @return numpy.ndarray: An array of object points as defined by the E-wand detector.
        """

    def detect(self, img_bgr):
        """
        Detects feature points in the provided image.

        @param img_bgr (numpy.ndarray): Input image in BGR format as a NumPy array.

        @return tuple: A tuple containing:
            - retval (bool): Indicates whether the detection was successful.
            - image_points (numpy.ndarray): An array of detected 2D points,
              with NaN values for points that were not detected.
        """


def add_parser(subparsers):
    """
    Adds a sub-parser to the main arguments parser for E-wand detection.

    @param subparsers (argparse._SubParsersAction): The subparsers action to which the new parser will be added.
    """
    wand_parser = subparsers.add_parser("wand", help="Use the E-wand for detection")
    wand_parser.add_argument(
        "--wand-short-dist", type=float, default=0.2, help="Short distance between wand balls [in meters]"
    )
    wand_parser.add_argument(
        "--wand-long-dist", type=float, default=0.3, help="Long distance between wand balls [in meters]"
    )
