#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/ring_buffer.h"
#include "dvo/spsc_queue.h"

namespace dvo {

// The segmenter creates only this small metadata request on the 10 ms audio
// thread. Ring-buffer slicing and the potentially 18-second PCM append happen
// later on CandidateAssembler's worker.
struct CandidateAssemblyRequest {
  UtteranceCandidate candidate;
  std::size_t join_silence_samples{};
  std::uint64_t generation{};
};

struct CandidateAssemblyResult {
  std::string utterance_id;
  std::optional<UtteranceCandidate> candidate;
  std::string rejection;
  std::uint64_t generation{};
};

struct BackfillAssemblyRequest {
  std::string utterance_id;
  std::vector<SampleSpan> source_spans;
  std::size_t join_silence_samples{};
  std::size_t trailing_silence_samples{};
  std::uint64_t wake_end_sample{};
  std::uint64_t recognition_generation{};
  std::uint64_t generation{};
};

struct BackfillAssemblyResult {
  std::string utterance_id;
  std::vector<float> pcm;
  std::uint64_t first_sample{};
  std::uint64_t wake_end_sample{};
  std::uint64_t recognition_generation{};
  bool truncated{};
  std::string rejection;
  std::uint64_t generation{};
};

using AudioAssemblyRequest =
    std::variant<CandidateAssemblyRequest, BackfillAssemblyRequest>;
using AudioAssemblyResult =
    std::variant<CandidateAssemblyResult, BackfillAssemblyResult>;

[[nodiscard]] CandidateAssemblyResult assemble_candidate(
    TimedRingBuffer& ring, CandidateAssemblyRequest request);
[[nodiscard]] BackfillAssemblyResult assemble_backfill(
    TimedRingBuffer& ring, BackfillAssemblyRequest request);

// Single producer (audio processing thread), single worker, single completion
// consumer (audio processing thread). Outstanding work, including completed
// results not yet consumed, is bounded by capacity so neither queue can grow.
class CandidateAssembler {
 public:
  CandidateAssembler(TimedRingBuffer& ring, std::size_t capacity);
  ~CandidateAssembler();

  CandidateAssembler(const CandidateAssembler&) = delete;
  CandidateAssembler& operator=(const CandidateAssembler&) = delete;

  [[nodiscard]] bool try_submit(CandidateAssemblyRequest request);
  [[nodiscard]] bool try_submit(BackfillAssemblyRequest request);
  [[nodiscard]] bool try_pop(AudioAssemblyResult& result);
  void cancel_pending();
  void stop();

  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  [[nodiscard]] std::size_t outstanding() const {
    return outstanding_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::size_t queued() const { return requests_.size(); }

 private:
  void run(std::stop_token stop);
  [[nodiscard]] bool try_submit_request(AudioAssemblyRequest request);
  [[nodiscard]] static std::uint64_t request_generation(
      const AudioAssemblyRequest& request);
  static void set_request_generation(AudioAssemblyRequest& request,
                                     std::uint64_t generation);
  [[nodiscard]] static AudioAssemblyResult cancelled_result(
      const AudioAssemblyRequest& request);

  TimedRingBuffer& ring_;
  const std::size_t capacity_;
  SpscQueue<AudioAssemblyRequest> requests_;
  SpscQueue<AudioAssemblyResult> completions_;
  std::counting_semaphore<> wake_{0};
  std::jthread worker_;
  std::atomic<std::size_t> outstanding_{};
  std::atomic<std::uint64_t> generation_{1};
  std::atomic<bool> running_{true};
};

}  // namespace dvo
