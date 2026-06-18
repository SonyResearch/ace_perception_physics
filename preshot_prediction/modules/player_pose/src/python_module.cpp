// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include <opencv2/opencv.hpp>

#include "player_pose/PersonDetector.hpp"
#include "player_pose/PlayerPoseExtractor.hpp"
#include "player_pose/ros/PlayerPoseROSNode.hpp"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "vision_common/pybind11/opencv.hpp"

namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<perception::PlayerPoseParameters, perception::PlayerPoseParameters::SharedPtr>(m, "PlayerParameters")
    .def(py::init<>())
    .def("initialize", &perception::PlayerPoseParameters::Initialize)
    .def("initialize_cameras", &perception::PlayerPoseParameters::InitializeCameras);

  py::class_<perception::PlayerPoseDetection, perception::PlayerPoseDetection::SharedPtr>(m, "player_pose_features")
    .def_readwrite("camera_index", &perception::PlayerPoseDetection::camera_index)
    .def_readwrite("bbox", &perception::PlayerPoseDetection::bbox)
    .def_readwrite("keypoints", &perception::PlayerPoseDetection::keypoints);

  py::class_<perception::PlayerBBoxDetection, perception::PlayerBBoxDetection::SharedPtr>(m, "player_bbox_detection")
    .def_readwrite("sequence_number", &perception::PlayerBBoxDetection::sequence_number)
    .def_readwrite("camera_index", &perception::PlayerBBoxDetection::camera_index)
    .def_readwrite("confidence", &perception::PlayerBBoxDetection::confidence)
    .def_readwrite("bbox", &perception::PlayerBBoxDetection::bbox)
    .def_readwrite("keypoints", &perception::PlayerBBoxDetection::keypoints);

  py::class_<perception::PlayerPoseEstimate, perception::PlayerPoseEstimate::SharedPtr>(m, "player_pose_estimate")
    .def_readwrite("player_id", &perception::PlayerPoseEstimate::player_id)
    .def_readwrite("keypoints", &perception::PlayerPoseEstimate::keypoints)
    .def_readwrite("projection_error", &perception::PlayerPoseEstimate::projection_error)
    .def_readwrite("confidences", &perception::PlayerPoseEstimate::confidences);

  py::class_<perception::PlayerFrameFeatures, perception::PlayerFrameFeatures::SharedPtr>(m, "player_frame_features")
    .def_readwrite("sequence_number", &perception::PlayerFrameFeatures::sequence_number)
    .def_readwrite("features", &perception::PlayerFrameFeatures::features)
    .def_readwrite("estimated_players", &perception::PlayerFrameFeatures::estimated_players);

  py::class_<perception::PersonDetector, perception::PersonDetector::SharedPtr>(m, "person_detector")
    .def(py::init<>())
    .def("initialize", &perception::PersonDetector::Initialize)
    .def("add_detection_request",
         [](perception::PersonDetector* self, size_t seq_id, size_t cam_idx, size_t player_idx, cv::Mat image,
            bool correct_orientation) {
           cv::cuda::GpuMat gpu_image;
           gpu_image.upload(image);
           self->AddBBoxDetectionRequest(seq_id, player_idx, cam_idx, gpu_image, correct_orientation);
         })
    .def("get_last_detection", &perception::PersonDetector::GetLastBBoxDetection)
    .def("set_callback",
         [](perception::PersonDetector* self, py::function cb) {
           self->SetPlayerDetectedCallback([cb](perception::PlayerBBoxDetection::SharedPtr detection) {
             py::gil_scoped_acquire acquire;
             cb(detection);
           });
         })
    .def(
      "extract_person",
      [](perception::PersonDetector* self, cv::Mat image, int camera_index, bool correct_orientation) {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (image.dims == 3) {
          // convert to 2D image with x channels
          image = cv::Mat(cv::Size(image.size[1], image.size[0]), CV_8UC(image.size[2]), image.data);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image);
        auto t3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> mat_time = t2 - t1;
        std::chrono::duration<double, std::milli> upload_time = t3 - t2;
        // std::cout << "PersonDetector::extract_person() - cv::Mat conversion time: " << mat_time.count()
        //           << " ms, upload time: " << upload_time.count() << " ms" << std::endl;
        auto detection = std::make_shared<perception::PlayerBBoxDetection>();
        detection->camera_index = camera_index;
        std::vector<perception::PlayerBBoxDetection::SharedPtr> detections;
        detections.push_back(detection);
        bool ret = self->ExtractPerson({gpu_image}, nullptr, detections, correct_orientation);
        if (ret) {
          return detection;
        }
        return perception::PlayerBBoxDetection::SharedPtr(nullptr);
      },
      py::call_guard<py::gil_scoped_release>());
  py::class_<perception::PlayerPoseExtractor, perception::PlayerPoseExtractor::SharedPtr>(m, "player_pose_extractor")
    .def(py::init<std::string>(), py::arg("log_name"))
    .def("get_last_camera_bbox_detection", &perception::PlayerPoseExtractor::GetLastBBoxDetectionForCamera)
    .def("get_last_camera_detection", &perception::PlayerPoseExtractor::GetLastDetectionForCamera)
    .def("get_detections", &perception::PlayerPoseExtractor::GetLastDetections)
    .def("on_images", &perception::PlayerPoseExtractor::OnImages)
    .def("start", &perception::PlayerPoseExtractor::Start, py::call_guard<py::gil_scoped_release>())
    .def("stop", &perception::PlayerPoseExtractor::Stop, py::call_guard<py::gil_scoped_release>())
    .def("set_player_detection_callback",
         [](perception::PlayerPoseExtractor* self, py::function cb) {
           self->SetPlayerDetectedCallback(
             [cb](size_t seq_id, size_t player_idx, size_t cam_idx, const Eigen::Vector4i& bbox, float conf) {
               py::gil_scoped_acquire acquire;
               cb(seq_id, player_idx, cam_idx, bbox, conf);
             });
         })
    .def(
      "set_player_keypoints_callback",
      [](perception::PlayerPoseExtractor* self, py::function cb) {
        self->SetPlayerPose2DCallback([cb](size_t seq_id, const perception::PlayerPoseDetection::SharedPtr& detection) {
          py::gil_scoped_acquire acquire;
          cb(seq_id, detection);
        });
      })
    .def("set_player_pose_callback", [](perception::PlayerPoseExtractor* self, py::function cb) {
      self->SetPlayerPose3DCallback(
        [cb](size_t seq_id, int player_id, const perception::PlayerPoseEstimate::SharedPtr& pose) {
          py::gil_scoped_acquire acquire;
          cb(seq_id, player_id, pose);
        });
    });

  py::class_<perception::PlayerPoseROSNode, perception::PlayerPoseROSNode::SharedPtr>(m, "player_pose_ros_node")
    .def(py::init<>())
    .def("start", &perception::PlayerPoseROSNode::Start)
    .def("stop", &perception::PlayerPoseROSNode::Stop);
}
