#include "dvo/volume_math.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dvo {

VolumeAdjustment calculate_volume_adjustment(double current, int delta_percent) {
  if (!std::isfinite(current) || current < 0.0 || current > 1.0) {
    throw std::invalid_argument("current volume must be finite and in [0, 1]");
  }
  if (delta_percent == 0 || delta_percent < -100 || delta_percent > 100) {
    throw std::invalid_argument("volume delta must be in [-100, -1] or [1, 100]");
  }

  VolumeAdjustment result;
  result.current = current;
  result.requested = current + static_cast<double>(delta_percent) / 100.0;
  result.target = std::clamp(result.requested, 0.0, 1.0);
  result.clamped = result.requested < 0.0 || result.requested > 1.0;
  result.noop = std::abs(result.target - result.current) <= 1.0e-9;
  return result;
}

}  // namespace dvo
