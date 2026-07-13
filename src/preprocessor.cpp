#include "dvo/preprocessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dvo {

BypassPreprocessor::BypassPreprocessor(const AudioConfig& config) : config_(config) {
  if (config_.target_sample_rate != kProcessingSampleRate || config_.frame_ms != 10) {
    throw std::invalid_argument("phase 1 bypass preprocessor requires 16 kHz / 10 ms output");
  }
  input_mono_.reserve(8192);
  output_samples_.reserve(8192);
}

PreprocessResult BypassPreprocessor::process(const AudioPacket& microphone,
                                             const std::optional<AudioPacket>& render) {
  PreprocessResult result;
  result.render_available = render.has_value();
  if (microphone.format.sample_rate == 0 || microphone.format.channels == 0) {
    result.reset_required = true;
    return result;
  }
  if (microphone.discontinuity || (input_rate_ != 0 && input_rate_ != microphone.format.sample_rate)) {
    reset();
    pending_discontinuity_ = true;
    result.reset_required = true;
  }
  if (input_rate_ == 0) configure(microphone.format.sample_rate);
  if (!qpc_origin_valid_) {
    qpc_origin_100ns_ = microphone.qpc_100ns;
    qpc_origin_sample_ = next_output_sample_;
    qpc_origin_valid_ = true;
  }
  append_mono(microphone);
  resample();

  while (output_samples_.size() >= kFrameSamples) {
    NormalizedFrame frame;
    frame.first_sample = next_output_sample_;
    frame.qpc_100ns = qpc_origin_100ns_ +
                      (frame.first_sample - qpc_origin_sample_) * 10000000ULL /
                          kProcessingSampleRate;
    frame.discontinuity = pending_discontinuity_;
    pending_discontinuity_ = false;
    std::copy_n(output_samples_.begin(), kFrameSamples, frame.samples.begin());
    output_samples_.erase(output_samples_.begin(), output_samples_.begin() + kFrameSamples);
    next_output_sample_ += kFrameSamples;
    result.frames.push_back(std::move(frame));
  }
  return result;
}

void BypassPreprocessor::reset() {
  input_rate_ = 0;
  next_input_position_ = 0.0;
  input_base_ = 0;
  input_mono_.clear();
  output_samples_.clear();
  qpc_origin_valid_ = false;
}

void BypassPreprocessor::configure(std::uint32_t input_rate) {
  input_rate_ = input_rate;
  next_input_position_ = static_cast<double>(input_base_);
}

void BypassPreprocessor::append_mono(const AudioPacket& packet) {
  const auto channels = static_cast<std::size_t>(packet.format.channels);
  const auto frames = packet.samples.size() / channels;
  input_mono_.reserve(input_mono_.size() + frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    float sum = 0.0F;
    for (std::size_t channel = 0; channel < channels; ++channel) {
      sum += packet.samples[frame * channels + channel];
    }
    input_mono_.push_back(std::clamp(sum / static_cast<float>(channels), -1.0F, 1.0F));
  }
}

void BypassPreprocessor::resample() {
  if (input_mono_.size() < 2) return;
  const double step = static_cast<double>(input_rate_) / config_.target_sample_rate;
  const auto available_end = input_base_ + input_mono_.size();
  while (next_input_position_ + 1.0 < static_cast<double>(available_end)) {
    const auto lower_absolute = static_cast<std::uint64_t>(std::floor(next_input_position_));
    const auto lower = static_cast<std::size_t>(lower_absolute - input_base_);
    const auto fraction = static_cast<float>(next_input_position_ - std::floor(next_input_position_));
    const float value = input_mono_[lower] +
                        (input_mono_[lower + 1] - input_mono_[lower]) * fraction;
    output_samples_.push_back(value);
    next_input_position_ += step;
  }

  const auto consumed_absolute = static_cast<std::uint64_t>(std::floor(next_input_position_));
  if (consumed_absolute > input_base_) {
    const auto erase_count = std::min(static_cast<std::size_t>(consumed_absolute - input_base_),
                                      input_mono_.size() - 1);
    input_mono_.erase(input_mono_.begin(), input_mono_.begin() +
                      static_cast<std::ptrdiff_t>(erase_count));
    input_base_ += erase_count;
  }
}

}  // namespace dvo
