// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "racket_pose_estimation/RacketInference.hpp"
#include "racket_pose_estimation/RacketPoseExtractor.hpp"
#include "racket_pose_estimation/RacketPoseFitter.hpp"
#include "racket_pose_estimation/ros/RacketPoseROSNode.hpp"
#include "vision_common/pybind11/opencv.hpp"

namespace py = ::pybind11;
namespace perception {

PYBIND11_MODULE(python_module, m) {
  py::module::import("triangulation.python_module");

  py::class_<RacketParameters::CameraCaptureParameters>(m, "RacketCameraCaptureParameters")
    .def(py::init())
    .def_readwrite("camera_buffer_length", &RacketParameters::CameraCaptureParameters::camera_buffer_length)
    .def_readwrite("worker_queue_length", &RacketParameters::CameraCaptureParameters::worker_queue_length)
    .def_readwrite("use_variable_response_time",
                   &RacketParameters::CameraCaptureParameters::use_variable_response_time);

  py::class_<RacketParameters::PlayerParameters>(m, "RacketPlayerParameters")
    .def(py::init())
    .def_readonly("racket_id", &RacketParameters::PlayerParameters::racket_id)
    .def_readonly("name", &RacketParameters::PlayerParameters::name)
    .def_readonly("cameras", &RacketParameters::PlayerParameters::cameras);

  py::class_<RacketParameters::KeypointFitterParameters>(m, "RacketKeypointFitterParameters")
    .def(py::init())
    .def_readwrite("max_skips_reset", &RacketParameters::KeypointFitterParameters::max_skips_reset)
    .def_readwrite("max_delta_err_reset", &RacketParameters::KeypointFitterParameters::max_delta_err_reset)
    .def_readwrite("max_error", &RacketParameters::KeypointFitterParameters::max_error)
    .def_readwrite("max_iterations", &RacketParameters::KeypointFitterParameters::max_iterations)
    .def_readwrite("initial_rate", &RacketParameters::KeypointFitterParameters::initial_rate)
    .def_readwrite("min_axis_length", &RacketParameters::KeypointFitterParameters::min_axis_length);

  py::class_<RacketParameters::ExtractorParameters>(m, "RacketExtractorParameters")
    .def(py::init())
    .def_readwrite("workers_count", &RacketParameters::ExtractorParameters::workers_count)
    .def_readwrite("outdated_diff", &RacketParameters::ExtractorParameters::outdated_diff)
    .def_readwrite("sampling", &RacketParameters::ExtractorParameters::sampling)
    .def_readwrite("confidence_threshold", &RacketParameters::ExtractorParameters::confidence_threshold)
    .def_readwrite("nms_threshold", &RacketParameters::ExtractorParameters::nms_threshold)
    .def_readwrite("latest_weight", &RacketParameters::ExtractorParameters::latest_weight)
    .def_readwrite("history_difference", &RacketParameters::ExtractorParameters::history_difference);

  py::class_<RacketParameters, RacketParameters::SharedPtr>(m, "RacketParameters")
    .def(py::init())
    .def("initialize", &RacketParameters::Initialize)
    .def_readwrite("capture", &RacketParameters::capture)
    .def_readwrite("fitter", &RacketParameters::fitter)
    .def_readwrite("extractor", &RacketParameters::extractor)
    .def_readwrite("triangulator", &RacketParameters::triangulator);

  py::class_<RacketPoseFitter, RacketPoseFitter::SharedPtr>(m, "RacketPoseFitter")
    .def(py::init<>())
    .def("initialize", &RacketPoseFitter::Initialize)
    .def("reset", &RacketPoseFitter::ResetInitialRotation)
    .def("set_cameras", &RacketPoseFitter::SetCameras)
    .def("set_calibration", &RacketPoseFitter::SetCalibration)
    .def("fit_points", [](RacketPoseFitter* self, const Eigen::Vector3f& position,
                          const std::vector<std::vector<Eigen::Vector3f>>& keypoints) {
      int iter_count = 0;
      float err = 0;

      py::list rotation;
      auto result = self->FitPoints(position, keypoints, iter_count, err);
      rotation.append(result.x());
      rotation.append(result.y());
      rotation.append(result.z());
      rotation.append(result.w());
      return py::make_tuple(rotation, iter_count, err);
    });

  py::class_<RacketPoseFeatures, RacketPoseFeatures::SharedPtr>(m, "racket_pose_detection")
    .def(py::init<>())
    .def_readwrite("camera_index", &RacketPoseFeatures::camera_index)
    .def_readwrite("confidence", &RacketPoseFeatures::confidence)
    .def_readwrite("bbox", &RacketPoseFeatures::bbox)
    .def_readwrite("center", &RacketPoseFeatures::center)
    .def_readwrite("keypoints", &RacketPoseFeatures::keypoints);
  py::class_<RacketEstimatedPose, RacketEstimatedPose::SharedPtr>(m, "RacketEstimatedPose")
    .def(py::init<>())
    .def_readwrite("racket_id", &RacketEstimatedPose::racket_id)
    .def_readwrite("position", &RacketEstimatedPose::position)
    .def_property("orientation", &RacketEstimatedPose::GetOrientation, &RacketEstimatedPose::SetOrientation)
    .def_readwrite("iterations_count", &RacketEstimatedPose::iterations_count)
    .def_readwrite("reprojection_err", &RacketEstimatedPose::reprojection_err)
    .def_readwrite("orientation_error", &RacketEstimatedPose::orientation_error)
    .def_readwrite("orientation_confidence", &RacketEstimatedPose::orientation_confidence);
  py::class_<RacketFrameFeatures, RacketFrameFeatures::SharedPtr>(m, "RacketFrameFeatures")
    .def(py::init<>())
    .def_readwrite("sequence_number", &RacketFrameFeatures::sequence_number)
    .def_readwrite("estimated_rackets", &RacketFrameFeatures::estimated_rackets)
    .def_readwrite("features", &RacketFrameFeatures::features);

  py::class_<RacketPoseExtractor, RacketPoseExtractor::SharedPtr>(m, "racket_pose_extractor")
    .def(py::init<std::string>(), py::arg("file_name"))
    .def("set_roi", &RacketPoseExtractor::SetROI)
    .def("get_roi",
         [](RacketPoseExtractor* self, int player_index, int camera_index) {
           Eigen::Vector4i roi;
           if (self->GetROI(player_index, camera_index, roi)) {
             return py::cast(roi);
           }
           return py::object(py::cast(nullptr));
         })
    .def("get_detections", &RacketPoseExtractor::GetLastDetections)
    .def("get_last_camera_detection", &RacketPoseExtractor::GetLastDetectionForCamera)
    .def("start", &RacketPoseExtractor::Start)
    .def("stop", &RacketPoseExtractor::Stop)
    .def("on_images", &RacketPoseExtractor::OnImages)
    .def("set_racket_pose_callback",
         [](perception::RacketPoseExtractor* self, py::function cb) {
           self->SetRacketPose3DCallback(
             [cb](size_t seq_id, int racket_id, perception::RacketEstimatedPose::SharedPtr pose) {
               py::gil_scoped_acquire acquire;
               cb(seq_id, racket_id, pose);
             });
         })
    .def("set_racket_detection_callback", [](perception::RacketPoseExtractor* self, py::function cb) {
      self->SetRacketDetection2DCallback(
        [cb](size_t racket_id, size_t cam_id, perception::RacketPoseFeatures::SharedPtr detection) {
          py::gil_scoped_acquire acquire;
          cb(racket_id, cam_id, detection);
        });
    });

  py::class_<RacketInference, RacketInference::SharedPtr>(m, "racket_inference")
    .def(py::init<>())
    .def("initialize", &RacketInference::Initialize)
    .def("detect", [](RacketInference* self, cv::Mat image) {
      std::vector<cv::cuda::GpuMat> inputs;
      RacketFrameFeatures results;
      inputs.emplace_back(self->PreProcessImage(image));
      results.features.emplace_back(std::make_shared<RacketPoseFeatures>());
      if (!self->DetectRacketKeypoints(inputs, results)) {
        return py::object(py::cast(nullptr));
      }
      if (results.features[0] == nullptr) {
        return py::object(py::cast(nullptr));
      }
      return py::cast(results.features[0]);
    });

  py::class_<RacketPoseROSNode, RacketPoseROSNode::SharedPtr>(m, "racket_pose_ros_node")
    .def(py::init<>())
    .def("start", &RacketPoseROSNode::Start)
    .def("stop", &RacketPoseROSNode::Stop);
}
}  // namespace perception
