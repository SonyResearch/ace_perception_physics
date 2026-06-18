// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace ace_monitor {

glm::vec3 eigen_to_glm(const Eigen::Vector3f& v);
Eigen::Vector3f glm_to_eign(const glm::vec3& v);

class Line3D {
 public:
  Line3D() : start(0, 0, 0), end(0, 0, 0) {}
  Line3D(Eigen::Vector3f a, Eigen::Vector3f b) : start(std::move(a)), end(std::move(b)) {}

  [[nodiscard]] Eigen::Vector3f GetNormal() const { return (end - start).normalized(); }

  Eigen::Vector3f start;
  Eigen::Vector3f end;
};
class Ray3D {
 public:
  Ray3D() : start(0, 0, 0), normal(0, 0, 1) {}
  Ray3D(Eigen::Vector3f a, Eigen::Vector3f n) : start(std::move(a)), normal(std::move(n)) {}
  [[nodiscard]] Eigen::Vector3f SamplePoint(float distance) const { return start + normal * distance; }
  Eigen::Vector3f start;
  Eigen::Vector3f normal;
};

class Plane3D {
  static Eigen::Vector3f SolveIntersectingPoints(int ref_idx, int a_idx, int b_idx, const Plane3D& p1,
                                                 const Plane3D& p2);

 public:
  Plane3D() : distance(0), normal(0, 0, 1) {}
  Plane3D(float d, Eigen::Vector3f n) : distance(d), normal(std::move(n)) {}
  Plane3D(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c);
  Plane3D(const Eigen::Vector3f& point, Eigen::Vector3f normal);

  [[nodiscard]] float GetDistance(const Eigen::Vector3f& point) const;
  [[nodiscard]] Eigen::Vector3f Project(const Eigen::Vector3f& point) const;
  [[nodiscard]] Eigen::Vector3f IntersectLine(const Line3D& line, bool& intersected) const;
  [[nodiscard]] Eigen::Vector3f IntersectRay(const Ray3D& ray, bool& intersected) const;
  [[nodiscard]] std::optional<Ray3D> IntersectPlane(const Plane3D& plane) const;

  [[nodiscard]] Eigen::Vector3f GetPoint() const { return -normal * distance; }

  float distance;
  Eigen::Vector3f normal;
};

class Frustum {
  std::vector<Plane3D> planes_;
  std::vector<Eigen::Vector3f> corners_;
  double hfov_;
  double vfov_;

 public:
  Frustum() = default;
  Frustum(const Eigen::Matrix4f& pose, double hfov, double vfov, double near, double far);

  [[nodiscard]] double GetHFov() const { return hfov_; }
  [[nodiscard]] double GetVFov() const { return vfov_; }

  [[nodiscard]] std::vector<Eigen::Vector3f> Intersect(const Plane3D& plane) const;
  [[nodiscard]] bool IsPointInside(const Eigen::Vector3f& point) const;

  [[nodiscard]] const std::vector<Plane3D>& GetPlanes() const { return planes_; }
  [[nodiscard]] const std::vector<Eigen::Vector3f>& GetCorners() const { return corners_; }

  static void FillCorners(double hfov, double vfov, double near, double far, std::vector<Eigen::Vector3f>& corners);
};
}  // namespace ace_monitor
