// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ball_detector/ball_detector_opencv.hpp"
#include "eigen3/Eigen/Eigen"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace bd = ball_detector;
namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<bd::IBallDetectorParameters, bd::IBallDetectorParameters::SharedPtr>(m, "IBallDetectorParameters")
    .def(py::init())
    .def_readwrite("blur_kernel_size", &bd::IBallDetectorParameters::blur_kernel_size)
    .def_readwrite("hsv_lower_boundary", &bd::IBallDetectorParameters::hsv_lower_boundary)
    .def_readwrite("hsv_upper_boundary", &bd::IBallDetectorParameters::hsv_upper_boundary)
    .def_readwrite("motion_filter_enable", &bd::IBallDetectorParameters::motion_filter_enable)
    .def_readwrite("motion_filter_delay", &bd::IBallDetectorParameters::motion_filter_delay)
    .def_readwrite("motion_filter_lower_boundary", &bd::IBallDetectorParameters::motion_filter_lower_boundary)
    .def_readwrite("motion_filter_upper_boundary", &bd::IBallDetectorParameters::motion_filter_upper_boundary)
    .def_readwrite("min_circularity_ratio", &bd::IBallDetectorParameters::min_circularity_ratio)
    .def_readwrite("min_radius", &bd::IBallDetectorParameters::min_radius);

  py::class_<bd::BallDetectorOpenCV, bd::BallDetectorOpenCV::SharedPtr>(m, "BallDetector")
    .def(py::init<>())
    .def("set_parameters", &bd::BallDetectorOpenCV::SetParameters)
    .def("set_frame_rate", &bd::BallDetectorOpenCV::SetFrameRate)
    .def("set_cuda_device_id", &bd::BallDetectorOpenCV::SetCudaDeviceID)
    .def("set_bayer_image",
         [](bd::BallDetectorOpenCV::SharedPtr& self,
            const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& bayer_img) {
           cv::Mat bayer_img_cv(static_cast<int>(bayer_img.shape(0)), static_cast<int>(bayer_img.shape(1)), CV_8UC1,
                                const_cast<unsigned char*>(bayer_img.data()));
           return self->SetBayerImage(bayer_img_cv);
         })
    .def("set_bgr_image",
         [](bd::BallDetectorOpenCV::SharedPtr& self,
            const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& bgr_img) {
           if (bgr_img.ndim() != 3 || bgr_img.shape(2) != 3) {
             throw std::runtime_error("bgr_img must be a three-channel uint8 image");
           }
           cv::Mat bgr_img_cv(static_cast<int>(bgr_img.shape(0)), static_cast<int>(bgr_img.shape(1)), CV_8UC3,
                              const_cast<unsigned char*>(bgr_img.data()));
           return self->SetBgrImage(bgr_img_cv);
         })
    .def("get_bgr_image", &bd::BallDetectorOpenCV::GetBgrImage)
    .def(
      "detect_balls",
      [](bd::BallDetectorOpenCV::SharedPtr& self, const py::object& valid_mask) {
        cv::Mat valid_mask_cv;
        if (!valid_mask.is_none()) {
          const py::array_t<uint8_t> valid_mask_cast =
            valid_mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
          if (valid_mask_cast.ndim() != 2) {
            throw std::runtime_error("valid_mask must be a single-channel uint8 image");
          }
          valid_mask_cv =
            cv::Mat(static_cast<int>(valid_mask_cast.shape(0)), static_cast<int>(valid_mask_cast.shape(1)), CV_8UC1,
                    const_cast<unsigned char*>(valid_mask_cast.data()));
        }

        std::vector<std::pair<cv::Point2f, float>> ball_detections;
        self->DetectBalls(ball_detections, valid_mask_cv);
        static_assert(sizeof(cv::Point2f) == 2 * sizeof(float), "cv::Point2f size changed since last compile");
        return py::array_t<float, py::array::c_style>({ball_detections.size(), static_cast<size_t>(3)},
                                                      reinterpret_cast<const float*>(ball_detections.data()));
      },
      py::arg("valid_mask") = py::none())
    .def("get_detection_mask", &bd::BallDetectorOpenCV::GetDetectionMask);
}
