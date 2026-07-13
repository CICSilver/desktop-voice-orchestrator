#pragma once

#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/config.h"
#include "dvo/ring_buffer.h"

namespace dvo {

struct SegmenterResult {
  std::vector<UtteranceCandidate> candidates;
  std::vector<std::string> rejections;
};

class UtteranceSegmenter {
 public:
  UtteranceSegmenter(SegmentationConfig config, TimedRingBuffer& ring);

  void add_vad_interval(VadInterval interval);
  void set_vad_state(bool speech, std::uint64_t at_sample);
  void add_kws_hit(KwsHit hit);
  [[nodiscard]] SegmenterResult advance(std::uint64_t current_sample);
  void reset(bool discontinuity = false);
  void reconfigure(SegmentationConfig config);
  [[nodiscard]] std::size_t pending_count() const { return pending_hits_.size(); }

 private:
  [[nodiscard]] std::optional<UtteranceCandidate> finalize(const KwsHit& hit,
                                                           bool timed_out,
                                                           std::string& rejection);
  [[nodiscard]] std::string make_id() const;
  [[nodiscard]] std::uint64_t samples(std::uint32_t milliseconds) const;

  SegmentationConfig config_;
  TimedRingBuffer& ring_;
  std::deque<VadInterval> vad_intervals_;
  std::deque<KwsHit> pending_hits_;
  bool vad_speech_{};
  std::uint64_t current_speech_start_{};
  std::uint64_t last_speech_end_{};
  bool discontinuity_{};
};

}  // namespace dvo
