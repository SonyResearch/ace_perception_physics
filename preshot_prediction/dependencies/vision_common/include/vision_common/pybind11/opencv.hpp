// Confidential, Copyright 2024, Sony AI, All rights reserved.
// code origin from: https://github.com/pybind/pybind11/issues/2004
// Confidential, Copyright 2024, Sony AI, All rights reserved
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

template <>
struct type_caster<cv::Point> {
  PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

  bool load(handle obj, bool) {
    if (!isinstance<tuple>(obj)) {
      std::logic_error("Point(x,y) should be a tuple!");
      return false;
    }

    tuple pt = reinterpret_borrow<tuple>(obj);
    if (pt.size() != 2) {
      std::logic_error("Point(x,y) tuple should be size of 2");
      return false;
    }

    value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
    return true;
  }

  static handle cast(const cv::Point& pt, return_value_policy, handle) { return make_tuple(pt.x, pt.y).release(); }
};

template <>
struct type_caster<cv::Rect> {
  PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_xywh"));

  bool load(handle obj, bool) {
    if (!isinstance<tuple>(obj)) {
      std::logic_error("Rect should be a tuple!");
      return false;
    }
    tuple rect = reinterpret_borrow<tuple>(obj);
    if (rect.size() != 4) {
      std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
      return false;
    }

    value = cv::Rect(rect[0].cast<int>(), rect[1].cast<int>(), rect[2].cast<int>(), rect[3].cast<int>());
    return true;
  }

  static handle cast(const cv::Rect& rect, return_value_policy, handle) {
    return make_tuple(rect.x, rect.y, rect.width, rect.height).release();
  }
};
/*
template <>
struct type_caster<cv::Mat> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  //! 1. cast numpy.ndarray to cv::Mat
  bool load(handle obj, bool) {
    array b = reinterpret_borrow<array>(obj);
    buffer_info info = b.request();

    int nh = 1;
    int nw = 1;
    int nc = 1;
    int ndims = info.ndim;
    if (ndims == 2) {
      nh = info.shape[0];
      nw = info.shape[1];
    } else if (ndims == 3) {
      nh = info.shape[0];
      nw = info.shape[1];
      nc = info.shape[2];
    } else {
      char msg[64];
      std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
      throw std::logic_error(msg);
      return false;
    }

    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8UC(nc);
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32SC(nc);
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32FC(nc);
    } else {
      throw std::logic_error("Unsupported type, only support uchar, int32, float");
      return false;
    }

    value = cv::Mat(nh, nw, dtype, info.ptr);
    return true;
  }

  //! 2. cast cv::Mat to numpy.ndarray
  static handle cast(const cv::Mat& mat, return_value_policy, handle parent) {
    CV_UNUSED(parent);

    std::string format = format_descriptor<unsigned char>::format();
    size_t elemsize = sizeof(unsigned char);
    int nw = mat.cols;
    int nh = mat.rows;
    int nc = mat.channels();
    int depth = mat.depth();
    int type = mat.type();
    int dim = (depth == type) ? 2 : 3;

    if (depth == CV_8U) {
      format = format_descriptor<unsigned char>::format();
      elemsize = sizeof(unsigned char);
    } else if (depth == CV_32S) {
      format = format_descriptor<int>::format();
      elemsize = sizeof(int);
    } else if (depth == CV_32F) {
      format = format_descriptor<float>::format();
      elemsize = sizeof(float);
    } else {
      throw std::logic_error("Unsupport type, only support uchar, int32, float");
    }

    std::vector<size_t> bufferdim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferdim = {(size_t)nh, (size_t)nw};
      strides = {elemsize * (size_t)nw, elemsize};
    } else if (dim == 3) {
      bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
      strides = {(size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize};
    }
    return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
  }
};*/

template <>
struct type_caster<cv::MatND> {
 public:
  PYBIND11_TYPE_CASTER(cv::MatND, _("numpy.ndarray"));

  //! 1. cast numpy.ndarray to cv::Mat
  bool load(handle obj, bool) {
    array b = reinterpret_borrow<array>(obj);
    buffer_info info = b.request();
    std::vector<int> shape(info.ndim);
    for (int i = 0; i < info.ndim; ++i) {
      shape[i] = info.shape[i];
    }

    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8U;
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32S;
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32F;
    } else {
      throw std::logic_error("Unsupported type, only support uchar, int32, float");
      return false;
    }

    value = cv::MatND(info.ndim, shape.data(), dtype, info.ptr);
    return true;
  }

  //! 2. cast cv::Mat to numpy.ndarray
  static handle cast(const cv::MatND& mat, return_value_policy, handle parent) {
    CV_UNUSED(parent);

    std::string format = format_descriptor<unsigned char>::format();
    size_t elemsize = sizeof(unsigned char);
    int nw = mat.cols;
    int nh = mat.rows;
    int nc = mat.channels();
    int depth = mat.depth();
    int type = mat.type();
    int dim = (depth == type) ? 2 : 3;

    if (depth == CV_8U) {
      format = format_descriptor<unsigned char>::format();
      elemsize = sizeof(unsigned char);
    } else if (depth == CV_32S) {
      format = format_descriptor<int>::format();
      elemsize = sizeof(int);
    } else if (depth == CV_32F) {
      format = format_descriptor<float>::format();
      elemsize = sizeof(float);
    } else {
      throw std::logic_error("Unsupport type, only support uchar, int32, float");
    }

    std::vector<size_t> bufferdim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferdim = {(size_t)nh, (size_t)nw};
      strides = {elemsize * (size_t)nw, elemsize};
    } else if (dim == 3) {
      bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
      strides = {(size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize};
    }
    return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
  }
};
}  // namespace pybind11::detail
