#include "advanced_localization_cpp/failsafe.hpp"

namespace advanced_localization
{

FailsafeManager::FailsafeManager(int max_rejections, double max_position_trace_m2)
: max_rejections_(max_rejections),
  max_position_trace_m2_(max_position_trace_m2)
{
}

FailsafeDecision FailsafeManager::evaluate(const FilterHealth & health, const std::optional<UpdateResult> & latest_update)
{
  std::string requested_mode = mode_;
  std::string reason = "hold";
  FailsafeCode code = FailsafeCode::None;

  if (mode_ == "AUTO") {
    if (health.consecutive_lightglue_rejections >= max_rejections_) {
      requested_mode = "LOITER";
      reason = "lightglue_rejection_limit";
      code = FailsafeCode::Inconsistency;
    } else if (health.consecutive_vision_loss >= max_rejections_) {
      requested_mode = "LOITER";
      reason = "vision_loss_limit";
      code = FailsafeCode::VisionLoss;
    } else if (health.position_trace_xy_m2 > max_position_trace_m2_) {
      requested_mode = "LOITER";
      reason = "covariance_limit";
      code = FailsafeCode::HighCovariance;
    }
  } else if (mode_ == "LOITER") {
    if (latest_update.has_value() && latest_update->accepted && health.position_trace_xy_m2 <= max_position_trace_m2_) {
      requested_mode = "AUTO";
      reason = "vision_recovered";
      code = FailsafeCode::None;
    }
  }

  const bool changed = requested_mode != mode_;
  mode_ = requested_mode;
  return FailsafeDecision{mode_, reason, code, changed};
}

}  // namespace advanced_localization
