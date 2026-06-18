// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace perception {

class PlayerPoseParameters;
class YoloObject {
 public:
  int label{-1};
  float confidence{0};
  cv::Rect2i rect;
  cv::Mat mask;
  std::vector<Eigen::Vector3f> keypoints{};
};
enum class YoloModelType { kClasses, kKeypoints, kSegmentation };
class YoloConfig {
 public:
  using SharedPtr = std::shared_ptr<YoloConfig>;
  using ConstSharedPtr = std::shared_ptr<const YoloConfig>;

  YoloModelType type{YoloModelType::kClasses};
  // Probability threshold used to filter detected objects
  float conf_threshold = 0.25F;
  // Non-maximum suppression threshold
  float nms_threshold = 0.65F;
  // Max number of detected objects to return
  int top_k = 100;
  // Segmentation config options
  int seg_channels = 32;
  int seg_h = 160;
  int seg_w = 160;
  float segmentation_threshold = 0.5F;
  // Pose estimation options
  int num_kps = 17;
  int keypoint_length = 3;  // number of elements per keypoint
  // Class thresholds
  std::vector<std::string> class_names;

  // Yolov8 default COCO class names order
  /*{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};*/
};

class YoloObjectDetector {
 public:
  using SharedPtr = std::shared_ptr<YoloObjectDetector>;
  using ConstSharedPtr = std::shared_ptr<const YoloObjectDetector>;

 private:
  class YoloObjectDetectorImpl;
  std::shared_ptr<YoloObjectDetectorImpl> impl_;

 public:
  YoloObjectDetector();
  virtual ~YoloObjectDetector() = default;
  bool Initialize(const std::string& model_path, YoloConfig::SharedPtr config,
                  std::shared_ptr<PlayerPoseParameters> player_params);
  bool DetectObjects(const std::vector<cv::cuda::GpuMat>& input, std::vector<std::vector<YoloObject>>& results,
                     const std::vector<std::pair<float,cv::Size2i>>& ratios);

  virtual cv::cuda::GpuMat ProcessImage(const cv::cuda::GpuMat& image, float& ratio, bool is_rgb);
  const Eigen::Vector2i& GetInputSize();
};

}  // namespace perception
