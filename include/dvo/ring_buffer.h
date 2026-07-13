#pragma once

#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <span>
#include <vector>

#include "dvo/audio_types.h"

namespace dvo {

struct RingSlice {
  SampleSpan requested;
  SampleSpan actual;
  std::vector<float> samples;
  bool truncated_left{};
  bool truncated_right{};
};

class TimedRingBuffer {
 public:
  explicit TimedRingBuffer(std::size_t capacity_samples);

  void push(std::uint64_t first_sample, std::span<const float> samples);
  void reset(std::uint64_t next_sample = 0);
  [[nodiscard]] RingSlice slice(SampleSpan span) const;
  [[nodiscard]] std::uint64_t head() const;
  [[nodiscard]] std::uint64_t tail() const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t capacity() const { return data_.size(); }

 private:
  mutable std::shared_mutex mutex_;
  std::vector<float> data_;
  std::uint64_t head_{};
  std::uint64_t tail_{};
  bool initialized_{};
};

}  // namespace dvo
