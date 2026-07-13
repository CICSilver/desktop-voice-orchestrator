#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/config.h"

namespace dvo {

struct PreprocessResult {
  std::vector<NormalizedFrame> frames;
  bool reset_required{};
  bool render_available{};
};

class IAudioPreprocessor {
 public:
  virtual ~IAudioPreprocessor() = default;
  virtual PreprocessResult process(const AudioPacket& microphone,
                                   const std::optional<AudioPacket>& render) = 0;
  virtual void reset() = 0;
};

class BypassPreprocessor final : public IAudioPreprocessor {
 public:
  explicit BypassPreprocessor(const AudioConfig& config);

  PreprocessResult process(const AudioPacket& microphone,
                           const std::optional<AudioPacket>& render) override;
  void reset() override;

 private:
  void configure(std::uint32_t input_rate);
  void append_mono(const AudioPacket& packet);
  void resample();

  AudioConfig config_;
  std::uint32_t input_rate_{};
  double next_input_position_{};
  std::uint64_t input_base_{};
  std::vector<float> input_mono_;
  std::vector<float> output_samples_;
  std::uint64_t next_output_sample_{};
  std::uint64_t qpc_origin_100ns_{};
  std::uint64_t qpc_origin_sample_{};
  bool qpc_origin_valid_{};
  bool pending_discontinuity_{};
};

}  // namespace dvo
