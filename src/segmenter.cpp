#include "dvo/segmenter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <sstream>
#include <utility>

namespace dvo {
namespace {

std::uint64_t saturating_sub(std::uint64_t value, std::uint64_t amount) {
  return value > amount ? value - amount : 0;
}

std::uint64_t saturating_add(std::uint64_t value, std::uint64_t amount) {
  return amount > std::numeric_limits<std::uint64_t>::max() - value
             ? std::numeric_limits<std::uint64_t>::max()
             : value + amount;
}

}  // namespace

UtteranceSegmenter::UtteranceSegmenter(SegmentationConfig config,
                                       TimedRingBuffer& ring)
    : UtteranceSegmenter(std::move(config),
                         ActivationConfig{.enabled = false}, ring) {}

UtteranceSegmenter::UtteranceSegmenter(SegmentationConfig config,
                                       ActivationConfig activation,
                                       TimedRingBuffer& ring)
    : config_(std::move(config)),
      activation_config_(std::move(activation)),
      ring_(ring) {}

void UtteranceSegmenter::add_vad_interval(VadInterval interval) {
  if (interval.span.empty()) return;
  if (!vad_intervals_.empty() &&
      interval.span.start <= vad_intervals_.back().span.end) {
    vad_intervals_.back().span.end =
        std::max(vad_intervals_.back().span.end, interval.span.end);
  } else {
    vad_intervals_.push_back(interval);
  }
  last_speech_end_ = std::max(last_speech_end_, interval.span.end);
  const auto keep_after =
      saturating_sub(ring_.tail(), samples(config_.max_candidate_ms) * 2);
  while (!vad_intervals_.empty() &&
         vad_intervals_.front().span.end < keep_after) {
    vad_intervals_.pop_front();
  }
}

std::optional<UtteranceStart> UtteranceSegmenter::set_vad_state(
    bool speech, std::uint64_t at_sample, bool allow_followup) {
  if (speech == vad_speech_) return std::nullopt;
  vad_speech_ = speech;
  if (!speech) {
    last_speech_end_ = at_sample;
    return std::nullopt;
  }

  current_speech_start_ = at_sample;
  if (!allow_followup || activation_state_.playback_hold ||
      !activation_config_.enabled ||
      activation_state_.state != SegmenterActivationState::armed_idle) {
    return std::nullopt;
  }

  // VAD is processed before advance() for a frame. Consequently speech that
  // starts exactly on the idle boundary wins the race and belongs to the
  // activation; only a strictly later onset is rejected.
  if (at_sample >= activation_state_.hard_deadline_sample) {
    transition_to_dormant(ActivationTransitionKind::expired, at_sample,
                          "hard_limit");
    return std::nullopt;
  }
  if (at_sample > activation_state_.idle_deadline_sample) {
    // Capture while an earlier turn is still awaiting exact validation. The
    // new turn is conditional: followup_plan_eligible() will admit it only if
    // a deterministic earlier refresh subsequently covers this onset.
    if (awaiting_validation_turns_.empty()) {
      transition_to_dormant(ActivationTransitionKind::expired, at_sample,
                            "idle_timeout");
      return std::nullopt;
    }
  }

  PendingTurn pending;
  pending.origin = UtteranceOrigin::followup;
  pending.utterance_id = make_id();
  pending.activation_id = activation_state_.activation_id;
  pending.turn_index = activation_state_.turn_index + 1;
  pending.trigger_sample = at_sample;
  pending_turns_.push_back(pending);

  activation_state_.state = SegmenterActivationState::followup_turn;
  activation_state_.turn_index = pending.turn_index;
  return make_start(pending);
}

std::optional<UtteranceStart> UtteranceSegmenter::add_kws_hit(KwsHit hit) {
  const auto debounce = samples(500);
  if (!last_keyword_.empty() && hit.keyword == last_keyword_ &&
      hit.detected_at_sample < saturating_add(last_keyword_sample_, debounce)) {
    return std::nullopt;
  }
  last_keyword_ = hit.keyword;
  last_keyword_sample_ = hit.detected_at_sample;

  if (activation_config_.enabled) {
    if (activation_state_.active()) {
      transition_to_dormant(ActivationTransitionKind::cancelled,
                            hit.detected_at_sample, "keyword_rearm");
    }
    for (const auto& pending : pending_turns_) {
      deferred_rejections_.push_back(
          {pending.utterance_id, "superseded by keyword rearm",
           pending.origin, pending.activation_id, pending.turn_index,
           pending.trigger_sample, hit.detected_at_sample});
    }
    pending_turns_.clear();
    awaiting_validation_turns_.clear();

    activation_state_.state = SegmenterActivationState::keyword_turn;
    activation_state_.activation_id = make_activation_id();
    activation_state_.turn_index = 0;
    activation_state_.idle_deadline_sample = 0;
    activation_state_.hard_deadline_sample = saturating_add(
        hit.detected_at_sample, samples(activation_config_.hard_limit_ms));
    validation_activation_id_ = activation_state_.activation_id;
    validation_idle_deadline_sample_ = 0;
    validation_hard_deadline_sample_ =
        activation_state_.hard_deadline_sample;
    queue_transition(ActivationTransitionKind::started,
                     hit.detected_at_sample, "keyword_hit");
  } else if (!pending_turns_.empty()) {
    const auto& previous = pending_turns_.back();
    if (previous.hit && hit.keyword == previous.hit->keyword &&
        hit.detected_at_sample <
            saturating_add(previous.hit->detected_at_sample, debounce)) {
      return std::nullopt;
    }
  }

  PendingTurn pending;
  pending.origin = UtteranceOrigin::keyword;
  pending.hit = std::move(hit);
  pending.utterance_id = make_id();
  pending.activation_id = activation_state_.activation_id;
  pending.turn_index = 0;
  pending.trigger_sample = pending.hit->detected_at_sample;
  pending_turns_.push_back(pending);
  return make_start(pending);
}

SegmenterResult UtteranceSegmenter::advance(std::uint64_t current_sample) {
  auto planned = advance_for_assembly(current_sample);
  SegmenterResult result;
  result.rejections = std::move(planned.rejections);
  result.activation_events = std::move(planned.activation_events);
  for (auto& request : planned.assemblies) {
    auto assembled = assemble_candidate(ring_, std::move(request));
    if (assembled.candidate) {
      result.candidates.push_back(std::move(*assembled.candidate));
    } else {
      result.rejections.push_back(
          {std::move(assembled.utterance_id), std::move(assembled.rejection),
           assembled.origin, std::move(assembled.activation_id),
           assembled.turn_index, assembled.trigger_sample,
           assembled.end_sample});
    }
  }
  return result;
}

SegmenterPlanResult UtteranceSegmenter::advance_for_assembly(
    std::uint64_t current_sample) {
  SegmenterPlanResult result;
  result.rejections = std::move(deferred_rejections_);
  deferred_rejections_.clear();

  while (!pending_turns_.empty()) {
    const auto& pending = pending_turns_.front();
    const auto waited = current_sample > pending.trigger_sample
                            ? current_sample - pending.trigger_sample
                            : 0;
    const bool timed_out = waited >= samples(config_.max_candidate_ms);
    const auto keyword_end = pending.hit ? pending.hit->wake_span.end : 0;
    const auto speech_reference =
        std::max({last_speech_end_, pending.trigger_sample, keyword_end});
    const bool endpoint =
        !vad_speech_ &&
        current_sample >= saturating_add(
                              speech_reference,
                              samples(config_.endpoint_silence_ms));
    if (!timed_out && !endpoint) break;

    std::string rejection;
    auto request = plan(pending, timed_out, rejection);
    const bool awaiting_validation =
        request.has_value() &&
        pending.origin == UtteranceOrigin::followup;
    if (request) {
      result.assemblies.push_back(std::move(*request));
    } else {
      result.rejections.push_back(
          {pending.utterance_id, std::move(rejection), pending.origin,
           pending.activation_id, pending.turn_index, pending.trigger_sample,
           current_sample});
    }
    const auto completed = pending;
    pending_turns_.pop_front();
    finish_turn(completed, current_sample, awaiting_validation);
  }

  expire_if_idle(current_sample);
  result.activation_events = drain_transitions();
  return result;
}

std::optional<CandidateAssemblyRequest> UtteranceSegmenter::plan(
    const PendingTurn& pending, bool timed_out, std::string& rejection) {
  return pending.origin == UtteranceOrigin::keyword
             ? plan_keyword(pending, timed_out, rejection)
             : plan_followup(pending, timed_out, rejection);
}

std::optional<CandidateAssemblyRequest> UtteranceSegmenter::plan_keyword(
    const PendingTurn& pending, bool timed_out, std::string& rejection) {
  if (!pending.hit) {
    rejection = "internal error: keyword turn has no KWS hit";
    return std::nullopt;
  }
  const auto& hit = *pending.hit;
  const auto max_window = samples(config_.max_candidate_ms);
  const auto connection_gap = samples(config_.endpoint_silence_ms);
  const auto search_start = saturating_sub(hit.wake_span.start, max_window);
  const auto search_end = saturating_add(hit.wake_span.end, max_window);

  std::vector<SampleSpan> relevant;
  for (const auto& interval : vad_intervals_) {
    if (interval.span.end >= search_start &&
        interval.span.start <= search_end) {
      relevant.push_back(interval.span);
    }
  }
  if (vad_speech_ && current_speech_start_ < search_end) {
    relevant.push_back({current_speech_start_, ring_.tail()});
  }
  if (relevant.empty()) {
    rejection = "no VAD speech around keyword";
    return std::nullopt;
  }
  std::sort(relevant.begin(), relevant.end(),
            [](const auto& a, const auto& b) { return a.start < b.start; });

  std::uint64_t envelope_start = hit.wake_span.start;
  std::uint64_t envelope_end = hit.wake_span.end;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& span : relevant) {
      if (saturating_add(span.end, connection_gap) >= envelope_start &&
          span.start <= saturating_add(envelope_end, connection_gap)) {
        const auto next_start = std::min(envelope_start, span.start);
        const auto next_end = std::max(envelope_end, span.end);
        changed = changed || next_start != envelope_start ||
                  next_end != envelope_end;
        envelope_start = next_start;
        envelope_end = next_end;
      }
    }
  }

  const auto min_speech = samples(config_.min_command_speech_ms);
  const auto left_speech = hit.wake_span.start > envelope_start
                               ? hit.wake_span.start - envelope_start
                               : 0;
  const auto right_speech = envelope_end > hit.wake_span.end
                                ? envelope_end - hit.wake_span.end
                                : 0;
  const bool has_left = left_speech >= min_speech;
  const bool has_right = right_speech >= min_speech;
  if (!has_left && !has_right) {
    rejection = "keyword has no command speech on either side";
    return std::nullopt;
  }

  UtteranceCandidate candidate;
  candidate.utterance_id = pending.utterance_id;
  candidate.origin = pending.origin;
  candidate.activation_id = pending.activation_id;
  candidate.turn_index = pending.turn_index;
  candidate.trigger_sample = pending.trigger_sample;
  candidate.keyword = hit.keyword;
  candidate.tokens = hit.tokens;
  candidate.wake_span = hit.wake_span;
  candidate.timed_out = timed_out;
  candidate.discontinuity = discontinuity_;
  candidate.position = has_left && has_right
                           ? WakePosition::embedded
                           : (has_left ? WakePosition::suffix
                                       : WakePosition::prefix);

  if (has_left) {
    candidate.source_spans.push_back(
        {saturating_sub(envelope_start, samples(config_.pre_roll_ms)),
         hit.wake_span.start});
  }
  if (has_right) {
    candidate.source_spans.push_back(
        {hit.wake_span.end,
         saturating_add(envelope_end, samples(config_.post_roll_ms))});
  }

  for (const auto span : candidate.source_spans) {
    if (candidate.wake_span && span.overlaps(*candidate.wake_span)) {
      rejection = "internal error: source span overlaps wake span";
      return std::nullopt;
    }
  }
  CandidateAssemblyRequest request;
  request.candidate = std::move(candidate);
  request.join_silence_samples =
      static_cast<std::size_t>(samples(config_.embedded_join_silence_ms));
  return request;
}

std::optional<CandidateAssemblyRequest> UtteranceSegmenter::plan_followup(
    const PendingTurn& pending, bool timed_out, std::string& rejection) {
  const auto max_window = samples(config_.max_candidate_ms);
  const auto connection_gap = samples(config_.endpoint_silence_ms);
  const auto search_end = saturating_add(pending.trigger_sample, max_window);

  std::vector<SampleSpan> relevant;
  for (const auto& interval : vad_intervals_) {
    // A follow-up may use the VAD model's more accurate onset before the
    // frame-level trigger, but it must never reconnect the preceding turn.
    if (interval.span.end >= pending.trigger_sample &&
        interval.span.start <= search_end) {
      relevant.push_back(interval.span);
    }
  }
  if (vad_speech_ && current_speech_start_ <= search_end) {
    relevant.push_back({current_speech_start_, ring_.tail()});
  }
  if (relevant.empty()) {
    rejection = "follow-up has no VAD speech";
    return std::nullopt;
  }
  std::sort(relevant.begin(), relevant.end(),
            [](const auto& a, const auto& b) { return a.start < b.start; });

  auto envelope_start = pending.trigger_sample;
  auto envelope_end = pending.trigger_sample;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& span : relevant) {
      if (saturating_add(span.end, connection_gap) >= envelope_start &&
          span.start <= saturating_add(envelope_end, connection_gap)) {
        const auto next_start = std::min(envelope_start, span.start);
        const auto next_end = std::max(envelope_end, span.end);
        changed = changed || next_start != envelope_start ||
                  next_end != envelope_end;
        envelope_start = next_start;
        envelope_end = next_end;
      }
    }
  }
  if (envelope_end <= envelope_start ||
      envelope_end - envelope_start <
          samples(config_.min_command_speech_ms)) {
    rejection = "follow-up command speech is too short";
    return std::nullopt;
  }

  UtteranceCandidate candidate;
  candidate.utterance_id = pending.utterance_id;
  candidate.origin = pending.origin;
  candidate.activation_id = pending.activation_id;
  candidate.turn_index = pending.turn_index;
  candidate.trigger_sample = pending.trigger_sample;
  candidate.position = std::nullopt;
  candidate.wake_span = std::nullopt;
  candidate.source_spans.push_back(
      {saturating_sub(envelope_start, samples(config_.pre_roll_ms)),
       saturating_add(envelope_end, samples(config_.post_roll_ms))});
  candidate.timed_out = timed_out;
  candidate.discontinuity = discontinuity_;

  CandidateAssemblyRequest request;
  request.candidate = std::move(candidate);
  request.join_silence_samples = 0;
  return request;
}

std::vector<SampleSpan> UtteranceSegmenter::keyword_backfill_spans(
    const KwsHit& hit) const {
  const auto max_window = samples(config_.max_candidate_ms);
  const auto connection_gap = samples(config_.endpoint_silence_ms);
  const auto search_start = saturating_sub(hit.wake_span.start, max_window);
  auto envelope_start = hit.wake_span.start;

  if (vad_speech_ && current_speech_start_ < hit.wake_span.start) {
    envelope_start = std::max(search_start, current_speech_start_);
  }

  for (auto it = vad_intervals_.rbegin(); it != vad_intervals_.rend(); ++it) {
    const auto span = it->span;
    if (span.end > hit.wake_span.start) continue;
    if (span.end < search_start) break;
    if (saturating_add(span.end, connection_gap) < envelope_start) break;
    envelope_start =
        std::max(search_start, std::min(envelope_start, span.start));
  }
  if (hit.wake_span.start <= envelope_start ||
      hit.wake_span.start - envelope_start <
          samples(config_.min_command_speech_ms)) {
    return {};
  }
  return {{saturating_sub(envelope_start, samples(config_.pre_roll_ms)),
           hit.wake_span.start}};
}

UtteranceStart UtteranceSegmenter::make_start(
    const PendingTurn& pending) const {
  UtteranceStart start;
  start.utterance_id = pending.utterance_id;
  start.origin = pending.origin;
  start.activation_id = pending.activation_id;
  start.turn_index = pending.turn_index;
  start.trigger_sample = pending.trigger_sample;
  start.stream_start_sample = pending.trigger_sample;
  if (pending.hit) {
    start.wake_span = pending.hit->wake_span;
    start.stream_start_sample = pending.hit->wake_span.end;
    start.backfill_spans = keyword_backfill_spans(*pending.hit);
  } else if (pending.trigger_sample != 0) {
    const auto backfill_start =
        saturating_sub(pending.trigger_sample, samples(config_.pre_roll_ms));
    if (backfill_start < pending.trigger_sample) {
      start.backfill_spans.push_back(
          {backfill_start, pending.trigger_sample});
    }
  }
  return start;
}

void UtteranceSegmenter::finish_turn(const PendingTurn& pending,
                                     std::uint64_t at_sample,
                                     bool awaiting_validation) {
  if (!activation_config_.enabled ||
      pending.activation_id.empty() ||
      pending.activation_id != activation_state_.activation_id) {
    return;
  }

  activation_state_.turn_index = pending.turn_index;
  if (pending.origin == UtteranceOrigin::followup &&
      awaiting_validation &&
      std::find_if(awaiting_validation_turns_.begin(),
                   awaiting_validation_turns_.end(),
                   [&pending](const AwaitingValidation& value) {
                     return value.turn_index == pending.turn_index;
                   }) == awaiting_validation_turns_.end()) {
    awaiting_validation_turns_.push_back(
        {pending.turn_index, pending.trigger_sample, at_sample});
  }
  if (at_sample >= activation_state_.hard_deadline_sample) {
    transition_to_dormant(ActivationTransitionKind::expired, at_sample,
                          "hard_limit");
    return;
  }

  activation_state_.state = SegmenterActivationState::armed_idle;
  if (pending.origin == UtteranceOrigin::keyword) {
    activation_state_.idle_deadline_sample = std::min(
        activation_state_.hard_deadline_sample,
        saturating_add(at_sample, samples(activation_config_.idle_timeout_ms)));
    validation_idle_deadline_sample_ =
        activation_state_.idle_deadline_sample;
  }
}

void UtteranceSegmenter::expire_if_idle(std::uint64_t current_sample) {
  if (!activation_config_.enabled ||
      activation_state_.state != SegmenterActivationState::armed_idle) {
    return;
  }
  if (current_sample >= activation_state_.hard_deadline_sample) {
    transition_to_dormant(ActivationTransitionKind::expired, current_sample,
                          "hard_limit");
  } else if (!activation_state_.playback_hold &&
             current_sample >= activation_state_.idle_deadline_sample &&
             awaiting_validation_turns_.empty()) {
    transition_to_dormant(ActivationTransitionKind::expired, current_sample,
                          "idle_timeout");
  }
}

bool UtteranceSegmenter::refresh_activation(
    const std::string& activation_id, std::uint32_t turn_index,
    std::uint64_t at_sample) {
  if (!activation_config_.enabled || activation_id.empty() ||
      activation_id != validation_activation_id_ || turn_index == 0) {
    return false;
  }
  if (!followup_plan_eligible(activation_id, turn_index)) return false;
  const auto awaiting = std::find_if(
      awaiting_validation_turns_.begin(), awaiting_validation_turns_.end(),
      [turn_index](const AwaitingValidation& value) {
        return value.turn_index == turn_index;
      });
  if (awaiting == awaiting_validation_turns_.end()) return false;
  // ASR and action-queue latency must not shorten or extend a sample-clock
  // activation window. The deterministic endpoint at which the candidate was
  // closed is the logical refresh instant.
  const auto refresh_sample = awaiting->completion_sample;
  awaiting_validation_turns_.erase(awaiting);
  static_cast<void>(at_sample);
  // A turn that started before the hard cap remains executable even if its
  // exact result arrives later. It resolves validation bookkeeping but cannot
  // reopen an already expired activation.
  if (!activation_state_.active() ||
      refresh_sample >= validation_hard_deadline_sample_) {
    if (awaiting_validation_turns_.empty()) {
      validation_activation_id_.clear();
      validation_idle_deadline_sample_ = 0;
      validation_hard_deadline_sample_ = 0;
    }
    return true;
  }
  activation_state_.idle_deadline_sample = std::min(
      activation_state_.hard_deadline_sample,
      saturating_add(refresh_sample,
                     samples(activation_config_.idle_timeout_ms)));
  validation_idle_deadline_sample_ =
      activation_state_.idle_deadline_sample;
  queue_transition(ActivationTransitionKind::refreshed, refresh_sample,
                   "valid_plan_enqueued", turn_index);
  return true;
}

bool UtteranceSegmenter::followup_plan_eligible(
    const std::string& activation_id, std::uint32_t turn_index) const {
  if (!activation_config_.enabled || activation_id.empty() ||
      activation_id != validation_activation_id_ || turn_index == 0) {
    return false;
  }
  const auto awaiting = std::find_if(
      awaiting_validation_turns_.begin(), awaiting_validation_turns_.end(),
      [turn_index](const AwaitingValidation& value) {
        return value.turn_index == turn_index;
      });
  return awaiting != awaiting_validation_turns_.end() &&
         awaiting->trigger_sample < validation_hard_deadline_sample_ &&
         awaiting->trigger_sample <= validation_idle_deadline_sample_;
}

bool UtteranceSegmenter::abandon_followup(
    const std::string& activation_id, std::uint32_t turn_index,
    std::uint64_t at_sample, std::string reason) {
  if (!activation_config_.enabled || activation_id.empty() ||
      activation_id != validation_activation_id_ || turn_index == 0) {
    return false;
  }
  const auto awaiting = std::find_if(
      awaiting_validation_turns_.begin(), awaiting_validation_turns_.end(),
      [turn_index](const AwaitingValidation& value) {
        return value.turn_index == turn_index;
      });
  if (awaiting == awaiting_validation_turns_.end()) return false;
  awaiting_validation_turns_.erase(awaiting);
  static_cast<void>(reason);
  if (activation_state_.active()) expire_if_idle(at_sample);
  if (awaiting_validation_turns_.empty() && !activation_state_.active()) {
    validation_activation_id_.clear();
    validation_idle_deadline_sample_ = 0;
    validation_hard_deadline_sample_ = 0;
  }
  return true;
}

bool UtteranceSegmenter::defer_activation_for_playback(
    std::uint64_t duration_samples, std::uint64_t at_sample) {
  if (!activation_config_.enabled || !activation_state_.active() ||
      activation_state_.idle_deadline_sample == 0 ||
      at_sample >= activation_state_.hard_deadline_sample) {
    return false;
  }
  activation_state_.idle_deadline_sample = std::min(
      activation_state_.hard_deadline_sample,
      saturating_add(activation_state_.idle_deadline_sample,
                     duration_samples));
  validation_idle_deadline_sample_ =
      activation_state_.idle_deadline_sample;
  return true;
}

bool UtteranceSegmenter::begin_activation_playback(
    std::uint64_t at_sample) {
  if (!activation_config_.enabled || !activation_state_.active() ||
      activation_state_.playback_hold ||
      at_sample >= activation_state_.hard_deadline_sample) {
    return false;
  }
  // As with VAD onset at the endpoint, a playback start delivered exactly on
  // the idle boundary wins. A strictly late start cannot revive an activation
  // unless an earlier follow-up is still awaiting deterministic validation.
  if (activation_state_.state == SegmenterActivationState::armed_idle &&
      at_sample > activation_state_.idle_deadline_sample &&
      awaiting_validation_turns_.empty()) {
    transition_to_dormant(ActivationTransitionKind::expired, at_sample,
                          "idle_timeout");
    return false;
  }
  activation_state_.playback_hold = true;
  return true;
}

bool UtteranceSegmenter::finish_activation_playback(
    std::uint64_t duration_samples, std::uint64_t at_sample) {
  if (!activation_config_.enabled || !activation_state_.active() ||
      !activation_state_.playback_hold) {
    return false;
  }
  activation_state_.playback_hold = false;
  if (at_sample >= activation_state_.hard_deadline_sample) {
    transition_to_dormant(ActivationTransitionKind::expired, at_sample,
                          "hard_limit");
    return true;
  }
  const auto deferred =
      defer_activation_for_playback(duration_samples, at_sample);
  expire_if_idle(at_sample);
  return deferred;
}

void UtteranceSegmenter::cancel_activation(std::uint64_t at_sample,
                                           std::string reason) {
  if (!activation_state_.active()) return;
  transition_to_dormant(ActivationTransitionKind::cancelled, at_sample,
                        std::move(reason));
}

std::vector<ActivationTransition>
UtteranceSegmenter::take_activation_events() {
  return drain_transitions();
}

void UtteranceSegmenter::transition_to_dormant(
    ActivationTransitionKind kind, std::uint64_t at_sample,
    std::string reason) {
  if (!activation_state_.active()) return;
  queue_transition(kind, at_sample, std::move(reason));
  const bool preserve_validation =
      kind == ActivationTransitionKind::expired &&
      !awaiting_validation_turns_.empty();
  if (!preserve_validation) {
    awaiting_validation_turns_.clear();
    validation_activation_id_.clear();
    validation_idle_deadline_sample_ = 0;
    validation_hard_deadline_sample_ = 0;
  }
  activation_state_ = {};
}

void UtteranceSegmenter::queue_transition(ActivationTransitionKind kind,
                                          std::uint64_t at_sample,
                                          std::string reason,
                                          std::optional<std::uint32_t> event_turn) {
  transition_events_.push_back(
      {kind,
       activation_state_.activation_id,
       event_turn.value_or(activation_state_.turn_index),
       at_sample,
       activation_state_.idle_deadline_sample,
       activation_state_.hard_deadline_sample,
       std::move(reason)});
}

std::vector<ActivationTransition>
UtteranceSegmenter::drain_transitions() {
  auto events = std::move(transition_events_);
  transition_events_.clear();
  return events;
}

void UtteranceSegmenter::reset(bool discontinuity) {
  cancel_activation(ring_.tail(), discontinuity ? "timeline_discontinuity"
                                                : "pipeline_reset");
  vad_intervals_.clear();
  pending_turns_.clear();
  deferred_rejections_.clear();
  awaiting_validation_turns_.clear();
  validation_activation_id_.clear();
  validation_idle_deadline_sample_ = 0;
  validation_hard_deadline_sample_ = 0;
  vad_speech_ = false;
  current_speech_start_ = 0;
  last_speech_end_ = 0;
  last_keyword_sample_ = 0;
  last_keyword_.clear();
  // Every pending hit and VAD interval was discarded above, so a future hit
  // starts in a new trusted window. Carrying this flag forward would make all
  // candidates after one device gap permanently non-executable.
  static_cast<void>(discontinuity);
  discontinuity_ = false;
}

void UtteranceSegmenter::reconfigure(SegmentationConfig config) {
  config_ = std::move(config);
  reset(false);
}

void UtteranceSegmenter::reconfigure(SegmentationConfig config,
                                     ActivationConfig activation) {
  cancel_activation(ring_.tail(), "configuration_changed");
  config_ = std::move(config);
  activation_config_ = std::move(activation);
  vad_intervals_.clear();
  pending_turns_.clear();
  deferred_rejections_.clear();
  awaiting_validation_turns_.clear();
  validation_activation_id_.clear();
  validation_idle_deadline_sample_ = 0;
  validation_hard_deadline_sample_ = 0;
  vad_speech_ = false;
  current_speech_start_ = 0;
  last_speech_end_ = 0;
  last_keyword_sample_ = 0;
  last_keyword_.clear();
  discontinuity_ = false;
}

std::string UtteranceSegmenter::make_id() const {
  static std::atomic<std::uint64_t> sequence{0};
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  std::ostringstream stream;
  stream << std::hex
         << std::chrono::duration_cast<std::chrono::microseconds>(now).count()
         << '-' << sequence.fetch_add(1, std::memory_order_relaxed);
  return stream.str();
}

std::string UtteranceSegmenter::make_activation_id() const {
  return "activation-" + make_id();
}

std::uint64_t UtteranceSegmenter::samples(
    std::uint32_t milliseconds) const {
  return static_cast<std::uint64_t>(milliseconds) * kProcessingSampleRate /
         1000;
}

const char* to_string(SegmenterActivationState state) {
  switch (state) {
    case SegmenterActivationState::dormant: return "dormant";
    case SegmenterActivationState::keyword_turn: return "keyword_turn";
    case SegmenterActivationState::armed_idle: return "armed_idle";
    case SegmenterActivationState::followup_turn: return "followup_turn";
  }
  return "unknown";
}

const char* to_string(ActivationTransitionKind kind) {
  switch (kind) {
    case ActivationTransitionKind::started: return "started";
    case ActivationTransitionKind::refreshed: return "refreshed";
    case ActivationTransitionKind::expired: return "expired";
    case ActivationTransitionKind::cancelled: return "cancelled";
  }
  return "unknown";
}

}  // namespace dvo
