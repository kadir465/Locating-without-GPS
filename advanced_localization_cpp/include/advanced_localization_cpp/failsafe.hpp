#pragma once

#include "advanced_localization_cpp/eskf.hpp"
#include "advanced_localization_cpp/types.hpp"

namespace advanced_localization
{

struct FailsafeDecision
{
  std::string mode{"AUTO"};
  std::string reason{"hold"};
  FailsafeCode code{FailsafeCode::None};
  bool changed{false};
};

class FailsafeManager
{
public:
  FailsafeManager(int max_rejections, double max_position_trace_m2);

  const std::string & mode() const { return mode_; }
  FailsafeDecision evaluate(const FilterHealth & health, const std::optional<UpdateResult> & latest_update);

private:
  int max_rejections_;
  double max_position_trace_m2_;
  std::string mode_{"AUTO"};
};

}  // namespace advanced_localization
