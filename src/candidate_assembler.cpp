#include "dvo/candidate_assembler.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace dvo {

CandidateAssemblyResult assemble_candidate(TimedRingBuffer& ring,
                                            CandidateAssemblyRequest request) {
  CandidateAssemblyResult result;
  result.utterance_id = request.candidate.utterance_id;
  result.origin = request.candidate.origin;
  result.activation_id = request.candidate.activation_id;
  result.turn_index = request.candidate.turn_index;
  result.trigger_sample = request.candidate.trigger_sample;
  result.end_sample = !request.candidate.source_spans.empty()
                          ? request.candidate.source_spans.back().end
                          : (request.candidate.wake_span
                                 ? request.candidate.wake_span->end
                                 : request.candidate.trigger_sample);
  result.generation = request.generation;

  auto candidate = std::move(request.candidate);
  candidate.pcm.clear();

  std::uint64_t reserve_samples{};
  for (const auto& span : candidate.source_spans) {
    const auto remaining = std::numeric_limits<std::uint64_t>::max() - reserve_samples;
    reserve_samples += std::min(span.size(), remaining);
  }
  if (candidate.source_spans.size() > 1) {
    const auto joins = static_cast<std::uint64_t>(candidate.source_spans.size() - 1);
    const auto silence = std::min<std::uint64_t>(
        std::numeric_limits<std::uint64_t>::max() / joins,
        static_cast<std::uint64_t>(request.join_silence_samples));
    const auto silence_total = joins * silence;
    reserve_samples += std::min(
        silence_total, std::numeric_limits<std::uint64_t>::max() - reserve_samples);
  }
  if (reserve_samples <= std::numeric_limits<std::size_t>::max()) {
    candidate.pcm.reserve(static_cast<std::size_t>(reserve_samples));
  }

  std::vector<SampleSpan> actual_spans;
  actual_spans.reserve(candidate.source_spans.size());
  for (const auto span : candidate.source_spans) {
    if (candidate.wake_span && span.overlaps(*candidate.wake_span)) {
      result.rejection = "internal error: source span overlaps wake span";
      return result;
    }
    auto slice = ring.slice_cooperative(span);
    candidate.truncated = candidate.truncated || slice.truncated_left ||
                          slice.truncated_right;
    if (slice.samples.empty()) continue;
    if (!actual_spans.empty()) {
      candidate.pcm.insert(candidate.pcm.end(), request.join_silence_samples, 0.0F);
    }
    actual_spans.push_back(slice.actual);
    candidate.pcm.insert(candidate.pcm.end(), slice.samples.begin(), slice.samples.end());
  }
  candidate.source_spans = std::move(actual_spans);
  if (candidate.pcm.empty()) {
    result.rejection = "candidate audio is no longer available in the ring buffer";
    return result;
  }

  result.candidate = std::move(candidate);
  return result;
}

BackfillAssemblyResult assemble_backfill(TimedRingBuffer& ring,
                                         BackfillAssemblyRequest request) {
  BackfillAssemblyResult result;
  result.utterance_id = std::move(request.utterance_id);
  result.origin = request.origin;
  result.activation_id = std::move(request.activation_id);
  result.turn_index = request.turn_index;
  result.trigger_sample = request.trigger_sample;
  result.stream_start_sample = request.stream_start_sample;
  result.recognition_generation = request.recognition_generation;
  result.generation = request.generation;

  bool have_audio{};
  for (const auto span : request.source_spans) {
    auto slice = ring.slice_cooperative(span);
    result.truncated = result.truncated || slice.truncated_left ||
                       slice.truncated_right;
    if (slice.samples.empty()) continue;
    if (!have_audio) {
      result.first_sample = slice.actual.start;
      have_audio = true;
    } else {
      result.pcm.insert(result.pcm.end(), request.join_silence_samples, 0.0F);
    }
    result.pcm.insert(result.pcm.end(), slice.samples.begin(), slice.samples.end());
  }
  if (have_audio && request.trailing_silence_samples != 0) {
    result.pcm.insert(result.pcm.end(), request.trailing_silence_samples, 0.0F);
  }
  // Empty source_spans are valid for prefix commands: recognition begins with
  // the first post-wake frame. Requested spans that expired are also allowed,
  // but remain visible through truncated for diagnostics.
  return result;
}

CandidateAssembler::CandidateAssembler(TimedRingBuffer& ring, std::size_t capacity)
    : ring_(ring), capacity_(capacity), requests_(capacity), completions_(capacity) {
  if (capacity == 0) throw std::invalid_argument("candidate assembler capacity must be positive");
  worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

CandidateAssembler::~CandidateAssembler() { stop(); }

bool CandidateAssembler::try_submit(CandidateAssemblyRequest request) {
  return try_submit_request(AudioAssemblyRequest{std::move(request)});
}

bool CandidateAssembler::try_submit(BackfillAssemblyRequest request) {
  return try_submit_request(AudioAssemblyRequest{std::move(request)});
}

bool CandidateAssembler::try_submit_request(AudioAssemblyRequest request) {
  if (!running_.load(std::memory_order_acquire)) return false;

  auto outstanding = outstanding_.load(std::memory_order_acquire);
  while (outstanding < capacity_) {
    if (outstanding_.compare_exchange_weak(outstanding, outstanding + 1,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
      set_request_generation(request, generation_.load(std::memory_order_acquire));
      if (!requests_.try_push(std::move(request))) {
        outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        return false;
      }
      wake_.release();
      return true;
    }
  }
  return false;
}

bool CandidateAssembler::try_pop(AudioAssemblyResult& result) {
  if (!completions_.try_pop(result)) return false;
  outstanding_.fetch_sub(1, std::memory_order_acq_rel);
  return true;
}

void CandidateAssembler::cancel_pending() {
  generation_.fetch_add(1, std::memory_order_acq_rel);
  wake_.release();
}

void CandidateAssembler::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (worker_.joinable()) {
    worker_.request_stop();
    wake_.release();
    worker_.join();
  }
}

void CandidateAssembler::run(std::stop_token stop) {
  while (!stop.stop_requested()) {
    wake_.acquire();
    if (stop.stop_requested()) break;

    AudioAssemblyRequest request;
    while (requests_.try_pop(request)) {
      AudioAssemblyResult result;
      const auto generation = request_generation(request);
      // Keep only the small request identity before moving the request into
      // the potentially allocating assembly routine. This lets the worker
      // turn allocation/logic failures into an explicit completion instead
      // of terminating the process from an uncaught worker exception.
      auto failure_identity = cancelled_result(request);
      if (generation != generation_.load(std::memory_order_acquire)) {
        result = std::move(failure_identity);
      } else {
        try {
          result = std::visit(
              [this](auto value) -> AudioAssemblyResult {
                using Request = decltype(value);
                if constexpr (std::is_same_v<Request, CandidateAssemblyRequest>) {
                  return assemble_candidate(ring_, std::move(value));
                } else {
                  return assemble_backfill(ring_, std::move(value));
                }
              },
              std::move(request));
          if (generation != generation_.load(std::memory_order_acquire)) {
            // Discard potentially large PCM assembled concurrently with a
            // reset and deliver the prebuilt cancellation identity instead.
            result = std::move(failure_identity);
          }
        } catch (const std::exception& error) {
          result = std::move(failure_identity);
          std::visit(
              [&error](auto& value) {
                value.rejection =
                    std::string{"audio assembly failed: "} + error.what();
              },
              result);
        } catch (...) {
          result = std::move(failure_identity);
          std::visit(
              [](auto& value) {
                value.rejection = "audio assembly failed: unknown exception";
              },
              result);
        }
      }

      while (!completions_.try_push(std::move(result))) {
        if (stop.stop_requested()) return;
        std::this_thread::yield();
      }
    }
  }
}

std::uint64_t CandidateAssembler::request_generation(
    const AudioAssemblyRequest& request) {
  return std::visit([](const auto& value) { return value.generation; }, request);
}

void CandidateAssembler::set_request_generation(AudioAssemblyRequest& request,
                                                std::uint64_t generation) {
  std::visit([generation](auto& value) { value.generation = generation; }, request);
}

AudioAssemblyResult CandidateAssembler::cancelled_result(
    const AudioAssemblyRequest& request) {
  return std::visit(
      [](const auto& value) -> AudioAssemblyResult {
        using Request = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Request, CandidateAssemblyRequest>) {
          CandidateAssemblyResult result;
          result.utterance_id = value.candidate.utterance_id;
          result.origin = value.candidate.origin;
          result.activation_id = value.candidate.activation_id;
          result.turn_index = value.candidate.turn_index;
          result.trigger_sample = value.candidate.trigger_sample;
          result.end_sample = !value.candidate.source_spans.empty()
                                  ? value.candidate.source_spans.back().end
                                  : (value.candidate.wake_span
                                         ? value.candidate.wake_span->end
                                         : value.candidate.trigger_sample);
          result.rejection = "candidate assembly cancelled by pipeline reset";
          result.generation = value.generation;
          return result;
        } else {
          BackfillAssemblyResult result;
          result.utterance_id = value.utterance_id;
          result.origin = value.origin;
          result.activation_id = value.activation_id;
          result.turn_index = value.turn_index;
          result.trigger_sample = value.trigger_sample;
          result.stream_start_sample = value.stream_start_sample;
          result.recognition_generation = value.recognition_generation;
          result.rejection = "backfill assembly cancelled by pipeline reset";
          result.generation = value.generation;
          return result;
        }
      },
      request);
}

}  // namespace dvo
