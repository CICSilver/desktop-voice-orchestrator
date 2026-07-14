#pragma once

#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/candidate_assembler.h"
#include "dvo/config.h"
#include "dvo/ring_buffer.h"

namespace dvo {

struct UtteranceStart {
  std::string utterance_id;
  SampleSpan wake_span;
  std::vector<SampleSpan> provisional_left_spans;
};

struct SegmenterRejection {
  std::string utterance_id;
  std::string reason;
};

struct SegmenterResult {
  std::vector<UtteranceCandidate> candidates;
  std::vector<SegmenterRejection> rejections;
};

struct SegmenterPlanResult {
  std::vector<CandidateAssemblyRequest> assemblies;
  std::vector<SegmenterRejection> rejections;
};

class UtteranceSegmenter {
 public:
  UtteranceSegmenter(SegmentationConfig config, TimedRingBuffer& ring);

  void add_vad_interval(VadInterval interval);
  void set_vad_state(bool speech, std::uint64_t at_sample);
  [[nodiscard]] std::optional<UtteranceStart> add_kws_hit(KwsHit hit);
  // Runtime path: computes only candidate metadata and source spans. No ring
  // slice or long PCM allocation occurs on the 10 ms processing thread.
  [[nodiscard]] SegmenterPlanResult advance_for_assembly(
      std::uint64_t current_sample);
  // Synchronous compatibility/test path.
  [[nodiscard]] SegmenterResult advance(std::uint64_t current_sample);
  void reset(bool discontinuity = false);
  void reconfigure(SegmentationConfig config);
  [[nodiscard]] std::size_t pending_count() const { return pending_hits_.size(); }

 private:
  struct PendingHit {
    KwsHit hit;
    std::string utterance_id;
  };

  [[nodiscard]] std::optional<CandidateAssemblyRequest> plan(
      const PendingHit& pending, bool timed_out, std::string& rejection);
  [[nodiscard]] std::vector<SampleSpan> provisional_left_spans(const KwsHit& hit) const;
  [[nodiscard]] std::string make_id() const;
  [[nodiscard]] std::uint64_t samples(std::uint32_t milliseconds) const;

  SegmentationConfig config_;
  TimedRingBuffer& ring_;
  std::deque<VadInterval> vad_intervals_;
  std::deque<PendingHit> pending_hits_;
  bool vad_speech_{};
  std::uint64_t current_speech_start_{};
  std::uint64_t last_speech_end_{};
  bool discontinuity_{};
};

}  // namespace dvo
