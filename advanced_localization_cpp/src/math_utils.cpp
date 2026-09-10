#include "advanced_localization_cpp/math_utils.hpp"

#include <cmath>

namespace advanced_localization
{

Matrix3 skew_symmetric(const Vector3 & vector)
{
  Matrix3 matrix;
  matrix << 0.0, -vector.z(), vector.y(),
    vector.z(), 0.0, -vector.x(),
    -vector.y(), vector.x(), 0.0;
  return matrix;
}

double normalize_angle(double angle_rad)
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

Eigen::VectorXd clip_vector_norm(const Eigen::VectorXd & vector, double max_norm)
{
  const double norm = vector.norm();
  if (norm <= max_norm || norm < 1e-12) {
    return vector;
  }
  return vector * (max_norm / norm);
}

Vector4 quaternion_normalize(const Vector4 & quaternion_wxyz)
{
  const double norm = quaternion_wxyz.norm();
  if (norm < 1e-12) {
    return Vector4(1.0, 0.0, 0.0, 0.0);
  }
  return quaternion_wxyz / norm;
}

Vector4 quaternion_multiply(const Vector4 & lhs_wxyz, const Vector4 & rhs_wxyz)
{
  const auto lhs = quaternion_normalize(lhs_wxyz);
  const auto rhs = quaternion_normalize(rhs_wxyz);
  const double lw = lhs(0), lx = lhs(1), ly = lhs(2), lz = lhs(3);
  const double rw = rhs(0), rx = rhs(1), ry = rhs(2), rz = rhs(3);
  return Vector4(
    lw * rw - lx * rx - ly * ry - lz * rz,
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw);
}

Vector4 quaternion_from_small_angle(const Vector3 & delta_theta_rad)
{
  const double theta = delta_theta_rad.norm();
  if (theta < 1e-12) {
    return quaternion_normalize(Vector4(1.0, 0.5 * delta_theta_rad.x(), 0.5 * delta_theta_rad.y(), 0.5 * delta_theta_rad.z()));
  }
  const double half_theta = 0.5 * theta;
  const Vector3 axis = delta_theta_rad / theta;
  const double sin_half = std::sin(half_theta);
  return quaternion_normalize(Vector4(std::cos(half_theta), axis.x() * sin_half, axis.y() * sin_half, axis.z() * sin_half));
}

Matrix3 rotation_matrix_from_quaternion(const Vector4 & quaternion_wxyz)
{
  const auto q = quaternion_normalize(quaternion_wxyz);
  const double w = q(0), x = q(1), y = q(2), z = q(3);
  Matrix3 matrix;
  matrix <<
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return matrix;
}

double yaw_from_quaternion(const Vector4 & quaternion_wxyz)
{
  const auto q = quaternion_normalize(quaternion_wxyz);
  const double w = q(0), x = q(1), y = q(2), z = q(3);
  return normalize_angle(std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)));
}

Vector4 quaternion_from_yaw(double yaw_rad)
{
  const double half = 0.5 * yaw_rad;
  return Vector4(std::cos(half), 0.0, 0.0, std::sin(half));
}

Matrix16 force_symmetric(const Matrix16 & matrix)
{
  return 0.5 * (matrix + matrix.transpose());
}

Eigen::MatrixXd force_symmetric_dynamic(const Eigen::MatrixXd & matrix)
{
  return 0.5 * (matrix + matrix.transpose());
}

Vector3 enu_to_ned_position(const Vector3 & enu)
{
  return Vector3(enu.y(), enu.x(), -enu.z());
}

Vector3 ned_to_enu_position(const Vector3 & ned)
{
  return Vector3(ned.y(), ned.x(), -ned.z());
}

Vector4 enu_quaternion_to_ned(const Vector4 & q_enu_wxyz)
{
  constexpr double half_pi = 1.57079632679489661923;
  const double yaw_ned = normalize_angle(half_pi - yaw_from_quaternion(q_enu_wxyz));
  return quaternion_from_yaw(yaw_ned);
}

}  // namespace advanced_localization
