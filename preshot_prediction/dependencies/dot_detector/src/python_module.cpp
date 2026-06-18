// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "dot_detector/dot_detector_opencv.hpp"
#include "dot_detector/dot_detector_opencv_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace dd = dot_detector;
namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<dd::IDotDetectorParameters, dd::IDotDetectorParameters::SharedPtr>(m, "IDotDetectorParameters")
    .def(py::init())
    .def_readwrite("markers_enable", &dd::IDotDetectorParameters::markers_enable)
    .def_readwrite("markers_min_circularity_ratio", &dd::IDotDetectorParameters::markers_min_circularity_ratio)
    .def_readwrite("markers_min_radius_ratio", &dd::IDotDetectorParameters::markers_min_radius_ratio)
    .def_readwrite("markers_max_radius_ratio", &dd::IDotDetectorParameters::markers_max_radius_ratio)
    .def_readwrite("markers_cone_angle_threshold", &dd::IDotDetectorParameters::markers_cone_angle_threshold);

  py::class_<dd::DotDetectorOpenCVParameters, dd::DotDetectorOpenCVParameters::SharedPtr>(m,
                                                                                          "DotDetectorOpenCVParameters")
    .def(py::init())
    .def_readwrite("markers_opencv_adaptive_threshold_power",
                   &dd::DotDetectorOpenCVParameters::markers_opencv_adaptive_threshold_power)
    .def_readwrite("markers_opencv_adaptive_threshold_ratio",
                   &dd::DotDetectorOpenCVParameters::markers_opencv_adaptive_threshold_ratio);

  py::class_<dd::DotDetectorOpenCV, dd::DotDetectorOpenCV::SharedPtr>(m, "DotDetector")
    .def(py::init<const dd::DotDetectorOpenCVParameters::SharedPtr&>())
    .def("set_parameters", &dd::DotDetectorOpenCV::SetParameters)
    .def("set_cuda_device_id", &dd::DotDetectorOpenCV::SetCudaDeviceID)
    .def("set_bayer_image",
         [](dd::DotDetectorOpenCV::SharedPtr& self,
            const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& bayer_img) {
           cv::Mat const bayer_img_cv(static_cast<int>(bayer_img.shape(0)), static_cast<int>(bayer_img.shape(1)),
                                      CV_8UC1, const_cast<unsigned char*>(bayer_img.data()));
           return self->SetBayerImage(bayer_img_cv);
         })
    .def("set_bgr_image",
         [](dd::DotDetectorOpenCV::SharedPtr& self,
            const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& bgr_img) {
           if (bgr_img.ndim() != 3 || bgr_img.shape(2) != 3) {
             throw std::runtime_error("bgr_img must be a three-channel uint8 image");
           }
           cv::Mat const bgr_img_cv(static_cast<int>(bgr_img.shape(0)), static_cast<int>(bgr_img.shape(1)), CV_8UC3,
                                    const_cast<unsigned char*>(bgr_img.data()));
           return self->SetBgrImage(bgr_img_cv);
         })
    .def("set_gray_image",
         [](dd::DotDetectorOpenCV::SharedPtr& self,
            const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& gray_img) {
           cv::Mat const gray_img_cv(static_cast<int>(gray_img.shape(0)), static_cast<int>(gray_img.shape(1)), CV_8UC1,
                                     const_cast<unsigned char*>(gray_img.data()));
           return self->SetGrayImage(gray_img_cv);
         })
    .def("detect_markers", [](dd::DotDetectorOpenCV::SharedPtr& self, const Eigen::Vector3f& ball_center_radius) {
      const cv::Point2f ball_center(ball_center_radius[0], ball_center_radius[1]);
      const float& ball_radius = ball_center_radius[2];

      std::vector<std::pair<cv::RotatedRect, float>> marker_detections;
      self->DetectMarkers(ball_center, ball_radius, marker_detections);
      static_assert(sizeof(cv::RotatedRect) == 5 * sizeof(float), "cv::RotatedRect size changed since last compile");
      return py::array_t<float, py::array::c_style>({marker_detections.size(), static_cast<size_t>(6)},
                                                    reinterpret_cast<const float*>(marker_detections.data()));
    });
}
