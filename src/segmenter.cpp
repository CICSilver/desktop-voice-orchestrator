#include "dvo/segmenter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>

namespace dvo {
namespace {

std::uint64_t saturating_sub(std::uint64_t value, std::uint64_t amount) {
  return value > amount ? value - amount : 0;
}

}  // namespace

UtteranceSegmenter::UtteranceSegmenter(SegmentationConfig config, TimedRingBuffer& ring)
    : config_(std::move(config)), ring_(ring) {}

void UtteranceSegmenter::add_vad_interval(VadInterval interval) {
  if (interval.span.empty()) return;
  if (!vad_intervals_.empty() && interval.span.start <= vad_intervals_.back().span.end) {
    vad_intervals_.back().span.end = std::max(vad_intervals_.back().span.end, interval.span.end);
  } else {
    vad_intervals_.push_back(interval);
  }
  last_speech_end_ = std::max(last_speech_end_, interval.span.end);
  const auto keep_after = saturating_sub(ring_.tail(), samples(config_.max_candidate_ms) * 2);
  while (!vad_intervals_.empty() && vad_intervals_.front().span.end < keep_after) vad_intervals_.pop_front();
}

void UtteranceSegmenter::set_vad_state(bool speech, std::uint64_t at_sample) {
  if (speech == vad_speech_) return;
  vad_speech_ = speech;
  if (speech) {
    current_speech_start_ = at_sample;
  } else {
    last_speech_end_ = at_sample;
  }
}

void UtteranceSegmenter::add_kws_hit(KwsHit hit) {
  if (!pending_hits_.empty()) {
    const auto debounce = samples(500);
    if (hit.detected_at_sample < pending_hits_.back().detected_at_sample + debounce &&
        hit.keyword == pending_hits_.back().keyword) return;
  }
  pending_hits_.push_back(std::move(hit));
}

SegmenterResult UtteranceSegmenter::advance(std::uint64_t current_sample) {
  SegmenterResult result;
  while (!pending_hits_.empty()) {
    const auto& hit = pending_hits_.front();
    const auto waited = current_sample > hit.detected_at_sample ? current_sample - hit.detected_at_sample : 0;
    const bool timed_out = waited >= samples(config_.max_candidate_ms);
    const auto speech_reference = std::max(last_speech_end_, hit.wake_span.end);
    const bool endpoint = !vad_speech_ && current_sample >= speech_reference + samples(config_.endpoint_silence_ms);
    if (!timed_out && !endpoint) break;

    std::string rejection;
    auto candidate = finalize(hit, timed_out, rejection);
    if (candidate) result.candidates.push_back(std::move(*candidate));
    else result.rejections.push_back(std::move(rejection));
    pending_hits_.pop_front();
  }
  return result;
}

std::optional<UtteranceCandidate> UtteranceSegmenter::finalize(const KwsHit& hit,
                                                               bool timed_out,
                                                               std::string& rejection) {
  const auto max_window = samples(config_.max_candidate_ms);
  const auto connection_gap = samples(config_.endpoint_silence_ms);
  const auto search_start = saturating_sub(hit.wake_span.start, max_window);
  const auto search_end = hit.wake_span.end + max_window;

  std::vector<SampleSpan> relevant;
  for (const auto& interval : vad_intervals_) {
    if (interval.span.end >= search_start && interval.span.start <= search_end) relevant.push_back(interval.span);
  }
  if (vad_speech_ && current_speech_start_ < search_end) {
    relevant.push_back({current_speech_start_, ring_.tail()});
  }
  if (relevant.empty()) {
    rejection = "no VAD speech around keyword";
    return std::nullopt;
  }
  std::sort(relevant.begin(), relevant.end(), [](const auto& a, const auto& b) { return a.start < b.start; });

  std::uint64_t envelope_start = hit.wake_span.start;
  std::uint64_t envelope_end = hit.wake_span.end;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& span : relevant) {
      if (span.end + connection_gap >= envelope_start && span.start <= envelope_end + connection_gap) {
        const auto next_start = std::min(envelope_start, span.start);
        const auto next_end = std::max(envelope_end, span.end);
        changed = changed || next_start != envelope_start || next_end != envelope_end;
        envelope_start = next_start;
        envelope_end = next_end;
      }
    }
  }

  const auto min_speech = samples(config_.min_command_speech_ms);
  const auto left_speech = hit.wake_span.start > envelope_start ? hit.wake_span.start - envelope_start : 0;
  const auto right_speech = envelope_end > hit.wake_span.end ? envelope_end - hit.wake_span.end : 0;
  const bool has_left = left_speech >= min_speech;
  const bool has_right = right_speech >= min_speech;
  if (!has_left && !has_right) {
    rejection = "keyword has no command speech on either side";
    return std::nullopt;
  }

  UtteranceCandidate candidate;
  candidate.utterance_id = make_id();
  candidate.keyword = hit.keyword;
  candidate.tokens = hit.tokens;
  candidate.wake_span = hit.wake_span;
  candidate.timed_out = timed_out;
  candidate.discontinuity = discontinuity_;
  candidate.position = has_left && has_right ? WakePosition::embedded
                                             : (has_left ? WakePosition::suffix : WakePosition::prefix);

  if (has_left) {
    candidate.source_spans.push_back({saturating_sub(envelope_start, samples(config_.pre_roll_ms)),
                                      hit.wake_span.start});
  }
  if (has_right) {
    candidate.source_spans.push_back({hit.wake_span.end,
                                      envelope_end + samples(config_.post_roll_ms)});
  }

  for (std::size_t i = 0; i < candidate.source_spans.size(); ++i) {
    auto& span = candidate.source_spans[i];
    if (span.overlaps(candidate.wake_span)) {
      rejection = "internal error: source span overlaps wake span";
      return std::nullopt;
    }
    auto slice = ring_.slice(span);
    candidate.truncated = candidate.truncated || slice.truncated_left || slice.truncated_right;
    span = slice.actual;
    candidate.pcm.insert(candidate.pcm.end(), slice.samples.begin(), slice.samples.end());
    if (i + 1 < candidate.source_spans.size()) {
      candidate.pcm.insert(candidate.pcm.end(), samples(config_.embedded_join_silence_ms), 0.0F);
    }
  }
  if (candidate.pcm.empty()) {
    rejection = "candidate audio is no longer available in the ring buffer";
    return std::nullopt;
  }
  return candidate;
}

void UtteranceSegmenter::reset(bool discontinuity) {
  vad_intervals_.clear();
  pending_hits_.clear();
  vad_speech_ = false;
  current_speech_start_ = 0;
  last_speech_end_ = 0;
  discontinuity_ = discontinuity;
}

void UtteranceSegmenter::reconfigure(SegmentationConfig config) {
  config_ = std::move(config);
  reset(false);
}

std::string UtteranceSegmenter::make_id() const {
  static std::atomic<std::uint64_t> sequence{0};
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  std::ostringstream stream;
  stream << std::hex << std::chrono::duration_cast<std::chrono::microseconds>(now).count()
         << '-' << sequence.fetch_add(1, std::memory_order_relaxed);
  return stream.str();
}

std::uint64_t UtteranceSegmenter::samples(std::uint32_t milliseconds) const {
  return static_cast<std::uint64_t>(milliseconds) * kProcessingSampleRate / 1000;
}

}  // namespace dvo
