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

enum class SegmenterActivationState {
  dormant,
  keyword_turn,
  armed_idle,
  followup_turn,
};

enum class ActivationTransitionKind {
  started,
  refreshed,
  expired,
  cancelled,
};

struct ActivationTransition {
  ActivationTransitionKind kind{ActivationTransitionKind::started};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t at_sample{};
  std::uint64_t idle_deadline_sample{};
  std::uint64_t hard_deadline_sample{};
  std::string reason;
};

struct ActivationSnapshot {
  SegmenterActivationState state{SegmenterActivationState::dormant};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t idle_deadline_sample{};
  std::uint64_t hard_deadline_sample{};
  bool playback_hold{};

  [[nodiscard]] bool active() const {
    return state != SegmenterActivationState::dormant;
  }
};

struct UtteranceStart {
  std::string utterance_id;
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
  std::uint64_t stream_start_sample{};
  std::optional<SampleSpan> wake_span;
  std::vector<SampleSpan> backfill_spans;
};

struct SegmenterRejection {
  std::string utterance_id;
  std::string reason;
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
  std::uint64_t end_sample{};
};

struct SegmenterResult {
  std::vector<UtteranceCandidate> candidates;
  std::vector<SegmenterRejection> rejections;
  std::vector<ActivationTransition> activation_events;
};

struct SegmenterPlanResult {
  std::vector<CandidateAssemblyRequest> assemblies;
  std::vector<SegmenterRejection> rejections;
  std::vector<ActivationTransition> activation_events;
};

class UtteranceSegmenter {
 public:
  // The legacy constructor intentionally disables follow-up activation. It
  // preserves the old one-keyword/one-candidate behavior for embedders that
  // have not opted into ActivationConfig yet.
  UtteranceSegmenter(SegmentationConfig config, TimedRingBuffer& ring);
  UtteranceSegmenter(SegmentationConfig config, ActivationConfig activation,
                     TimedRingBuffer& ring);

  void add_vad_interval(VadInterval interval);
  [[nodiscard]] std::optional<UtteranceStart> set_vad_state(
      bool speech, std::uint64_t at_sample, bool allow_followup = true);
  [[nodiscard]] std::optional<UtteranceStart> add_kws_hit(KwsHit hit);
  // Runtime path: computes only candidate metadata and source spans. No ring
  // slice or long PCM allocation occurs on the 10 ms processing thread.
  [[nodiscard]] SegmenterPlanResult advance_for_assembly(
      std::uint64_t current_sample);
  // Synchronous compatibility/test path.
  [[nodiscard]] SegmenterResult advance(std::uint64_t current_sample);

  // Called only after the exact-final transcript parsed successfully and its
  // command plan was accepted by the execution queue.
  [[nodiscard]] bool refresh_activation(const std::string& activation_id,
                                        std::uint32_t turn_index,
                                        std::uint64_t at_sample);
  // A conditionally captured follow-up is executable only after earlier
  // validation has refreshed the idle window far enough to cover its onset.
  [[nodiscard]] bool followup_plan_eligible(
      const std::string& activation_id, std::uint32_t turn_index) const;
  [[nodiscard]] bool abandon_followup(const std::string& activation_id,
                                      std::uint32_t turn_index,
                                      std::uint64_t at_sample,
                                      std::string reason);
  // Extends only the idle window. It neither changes the active turn nor
  // represents a command-driven refresh, and it can never cross the hard cap.
  [[nodiscard]] bool defer_activation_for_playback(
      std::uint64_t duration_samples, std::uint64_t at_sample);
  [[nodiscard]] bool begin_activation_playback(std::uint64_t at_sample);
  [[nodiscard]] bool finish_activation_playback(
      std::uint64_t duration_samples, std::uint64_t at_sample);
  void cancel_activation(std::uint64_t at_sample, std::string reason);
  [[nodiscard]] std::vector<ActivationTransition> take_activation_events();
  [[nodiscard]] const ActivationSnapshot& activation() const {
    return activation_state_;
  }
  [[nodiscard]] std::size_t awaiting_validation_count() const {
    return awaiting_validation_turns_.size();
  }

  void reset(bool discontinuity = false);
  void reconfigure(SegmentationConfig config);
  void reconfigure(SegmentationConfig config, ActivationConfig activation);
  [[nodiscard]] std::size_t pending_count() const { return pending_turns_.size(); }

 private:
  struct AwaitingValidation {
    std::uint32_t turn_index{};
    std::uint64_t trigger_sample{};
    std::uint64_t completion_sample{};
  };

  struct PendingTurn {
    UtteranceOrigin origin{UtteranceOrigin::keyword};
    std::optional<KwsHit> hit;
    std::string utterance_id;
    std::string activation_id;
    std::uint32_t turn_index{};
    std::uint64_t trigger_sample{};
  };

  [[nodiscard]] std::optional<CandidateAssemblyRequest> plan(
      const PendingTurn& pending, bool timed_out, std::string& rejection);
  [[nodiscard]] std::optional<CandidateAssemblyRequest> plan_keyword(
      const PendingTurn& pending, bool timed_out, std::string& rejection);
  [[nodiscard]] std::optional<CandidateAssemblyRequest> plan_followup(
      const PendingTurn& pending, bool timed_out, std::string& rejection);
  [[nodiscard]] std::vector<SampleSpan> keyword_backfill_spans(
      const KwsHit& hit) const;
  [[nodiscard]] UtteranceStart make_start(const PendingTurn& pending) const;
  void finish_turn(const PendingTurn& pending, std::uint64_t at_sample,
                   bool awaiting_validation);
  void expire_if_idle(std::uint64_t current_sample);
  void transition_to_dormant(ActivationTransitionKind kind,
                             std::uint64_t at_sample, std::string reason);
  void queue_transition(ActivationTransitionKind kind,
                        std::uint64_t at_sample, std::string reason,
                        std::optional<std::uint32_t> event_turn = std::nullopt);
  [[nodiscard]] std::vector<ActivationTransition> drain_transitions();
  [[nodiscard]] std::string make_id() const;
  [[nodiscard]] std::string make_activation_id() const;
  [[nodiscard]] std::uint64_t samples(std::uint32_t milliseconds) const;

  SegmentationConfig config_;
  ActivationConfig activation_config_;
  TimedRingBuffer& ring_;
  std::deque<VadInterval> vad_intervals_;
  std::deque<PendingTurn> pending_turns_;
  std::vector<SegmenterRejection> deferred_rejections_;
  std::vector<ActivationTransition> transition_events_;
  std::vector<AwaitingValidation> awaiting_validation_turns_;
  std::string validation_activation_id_;
  std::uint64_t validation_idle_deadline_sample_{};
  std::uint64_t validation_hard_deadline_sample_{};
  ActivationSnapshot activation_state_;
  bool vad_speech_{};
  std::uint64_t current_speech_start_{};
  std::uint64_t last_speech_end_{};
  std::uint64_t last_keyword_sample_{};
  std::string last_keyword_;
  bool discontinuity_{};
};

[[nodiscard]] const char* to_string(SegmenterActivationState state);
[[nodiscard]] const char* to_string(ActivationTransitionKind kind);

}  // namespace dvo
