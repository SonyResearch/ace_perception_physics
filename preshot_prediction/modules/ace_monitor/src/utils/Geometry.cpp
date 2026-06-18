// Confidential, Copyright 2025, Sony AI, All rights reserved

#include "ace_monitor/utils/Geometry.hpp"

#include <iostream>

namespace ace_monitor {

glm::vec3 eigen_to_glm(const Eigen::Vector3f& v) { return glm::vec3(v.x(), v.y(), v.z()); }
Eigen::Vector3f glm_to_eign(const glm::vec3& v) { return Eigen::Vector3f(v.x, v.y, v.z); }

Plane3D::Plane3D(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
  normal = (b - a).cross((c - a)).normalized();
  distance = -normal.dot(a);
}
Plane3D::Plane3D(const Eigen::Vector3f& point, Eigen::Vector3f n) : normal(std::move(n)) {
  distance = -normal.dot(point);
}
float Plane3D::GetDistance(const Eigen::Vector3f& point) const { return normal.dot(point) + distance; }

Eigen::Vector3f Plane3D::Project(const Eigen::Vector3f& point) const { return point - normal * GetDistance(point); }

Eigen::Vector3f Plane3D::IntersectLine(const Line3D& line, bool& intersected) const {
  auto diff = line.start - line.end;
  float denom = normal.dot(diff);
  if (denom == 0) {
    // parallel
    intersected = false;
    return line.start;
  }
  auto u = (normal.dot(line.start) + distance) / denom;
  if (u <= 0 || u >= 1) {
    intersected = false;
    return line.start;
  }
  intersected = true;

  return line.start + (line.end - line.start) * u;
}
Eigen::Vector3f Plane3D::IntersectRay(const Ray3D& ray, bool& intersected) const {
  float denom = normal.dot(ray.normal);
  if (std::abs(denom) <= 1e-5) {
    // parallel
    intersected = false;
    return ray.start;
  }
  auto u = (normal.dot(ray.start) + distance) / denom;
  intersected = true;

  return ray.start - ray.normal * u;
}
Eigen::Vector3f Plane3D::SolveIntersectingPoints(int ref_idx, int a_idx, int b_idx, const Plane3D& p1,
                                                 const Plane3D& p2) {
  auto a1 = p1.normal(a_idx);
  auto b1 = p1.normal(b_idx);
  auto d1 = p1.distance;

  auto a2 = p2.normal(a_idx);
  auto b2 = p2.normal(b_idx);
  auto d2 = p2.distance;

  Eigen::Vector3f ret;
  ret(ref_idx) = 0;

  ret(a_idx) = ((b2 * d1) - (b1 * d2)) / ((a1 * b2) - (a2 * b1));
  ret(b_idx) = ((a1 * d2) - (a2 * d1)) / ((a1 * b2) - (a2 * b1));

  return ret;
}
std::optional<Ray3D> Plane3D::IntersectPlane(const Plane3D& plane) const {
  // https://web.archive.org/web/20160306153443/http://geomalgorithms.com/a05-_intersect-1.html
  // need to solve for the equations:
  // P1: a1x + b1y + c1z + d1 = 0
  // P2: a2x + b2y + c2z + d2 = 0

  auto p3_norm = normal.cross(plane.normal);
  auto det = p3_norm.norm();
  if (det <= 1e-4) {
    return std::nullopt;
  }
  p3_norm /= det;
  Eigen::Vector3f point = ((p3_norm.cross(plane.normal) * distance) + (normal.cross(p3_norm) * plane.distance)) / det;
  return Ray3D(point, p3_norm);
}

Frustum::Frustum(const Eigen::Matrix4f& pose, double hfov, double vfov, double near, double far)
  : hfov_(hfov), vfov_(vfov) {
  planes_.resize(6);
  FillCorners(hfov, vfov, near, far, corners_);
  // transform points
  for (auto& corner : corners_) {
    auto v4 = Eigen::Vector4f(corner.x(), corner.y(), corner.z(), 1);
    auto pt = pose * v4;
    corner = pt.head<3>();
  }

  planes_[0] = Plane3D(corners_[0], corners_[2], corners_[1]);  // +x plane
  planes_[1] = Plane3D(corners_[2], corners_[4], corners_[3]);  // -y plane
  planes_[2] = Plane3D(corners_[4], corners_[6], corners_[5]);  // -x plane
  planes_[3] = Plane3D(corners_[6], corners_[0], corners_[7]);  // +y plane
  planes_[4] = Plane3D(corners_[0], corners_[6], corners_[2]);  // near plane
  planes_[5] = Plane3D(corners_[1], corners_[3], corners_[7]);  // far plane
}

void Frustum::FillCorners(double hfov, double vfov, double near, double far, std::vector<Eigen::Vector3f>& corners) {
  double half_hfov = hfov * 0.5;
  double half_vfov = vfov * 0.5;
  corners.resize(8);
  corners[0] = Eigen::Vector3d(near * sin(half_hfov), near * sin(half_vfov), near).cast<float>();
  corners[1] = Eigen::Vector3d(far * sin(half_hfov), far * sin(half_vfov), far).cast<float>();

  corners[2] = Eigen::Vector3d(near * sin(half_hfov), -near * sin(half_vfov), near).cast<float>();
  corners[3] = Eigen::Vector3d(far * sin(half_hfov), -far * sin(half_vfov), far).cast<float>();

  corners[4] = Eigen::Vector3d(-near * sin(half_hfov), -near * sin(half_vfov), near).cast<float>();
  corners[5] = Eigen::Vector3d(-far * sin(half_hfov), -far * sin(half_vfov), far).cast<float>();

  corners[6] = Eigen::Vector3d(-near * sin(half_hfov), near * sin(half_vfov), near).cast<float>();
  corners[7] = Eigen::Vector3d(-far * sin(half_hfov), far * sin(half_vfov), far).cast<float>();
}

std::vector<Eigen::Vector3f> Frustum::Intersect(const Plane3D& plane) const {
  std::vector<Eigen::Vector3f> res;

  std::vector<Line3D> intersections;
  for (const auto& plane_i : planes_) {
    auto ray = plane_i.IntersectPlane(plane);
    if (ray) {
      intersections.emplace_back(ray->SamplePoint(-100), ray->SamplePoint(100));
    }
  }

  // now intersect the lines vs the planes
  for (const auto& plane_i : planes_) {
    int j = 0;
    for (const auto& line : intersections) {
      bool intersected;
      auto pt = plane_i.IntersectLine(line, intersected);
      if (intersected && IsPointInside(pt)) {
        res.push_back(pt);
      }
      ++j;
    }
  }

  return res;
}
bool Frustum::IsPointInside(const Eigen::Vector3f& point) const {
  return std::all_of(planes_.begin(), planes_.end(),
                     [&point](const Plane3D& plane) { return !(plane.GetDistance(point) < -1e-2); });
}

}  // namespace ace_monitor
