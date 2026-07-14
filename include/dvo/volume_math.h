#pragma once

namespace dvo {

struct VolumeAdjustment {
  double current{};
  double requested{};
  double target{};
  bool clamped{};
  bool noop{};
};

// Computes a relative change in Windows endpoint-volume slider percentage
// points. current must be finite and in [0, 1], and delta must be non-zero.
[[nodiscard]] VolumeAdjustment calculate_volume_adjustment(double current,
                                                           int delta_percent);

}  // namespace dvo
