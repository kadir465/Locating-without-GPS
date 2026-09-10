#pragma once

#include "advanced_localization_cpp/types.hpp"

namespace advanced_localization
{

Matrix3 skew_symmetric(const Vector3 & vector);
double normalize_angle(double angle_rad);
Eigen::VectorXd clip_vector_norm(const Eigen::VectorXd & vector, double max_norm);
Vector4 quaternion_normalize(const Vector4 & quaternion_wxyz);
Vector4 quaternion_multiply(const Vector4 & lhs_wxyz, const Vector4 & rhs_wxyz);
Vector4 quaternion_from_small_angle(const Vector3 & delta_theta_rad);
Matrix3 rotation_matrix_from_quaternion(const Vector4 & quaternion_wxyz);
double yaw_from_quaternion(const Vector4 & quaternion_wxyz);
Vector4 quaternion_from_yaw(double yaw_rad);
Matrix16 force_symmetric(const Matrix16 & matrix);
Eigen::MatrixXd force_symmetric_dynamic(const Eigen::MatrixXd & matrix);

Vector3 enu_to_ned_position(const Vector3 & enu);
Vector3 ned_to_enu_position(const Vector3 & ned);
Vector4 enu_quaternion_to_ned(const Vector4 & q_enu_wxyz);

}  // namespace advanced_localization
