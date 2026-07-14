#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "dvo/audio_types.h"

namespace dvo {

// Audio buffers are prepared by the caller and moved into the bounded request
// queue by shared ownership. Submitting a request therefore does not copy PCM
// or wait for the recognizer worker.
using SharedPcm = std::shared_ptr<const std::vector<float>>;

struct StreamingRecognizerConfig {
  bool enabled{true};
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  std::filesystem::path tokens;
  std::string provider{"cpu"};
  std::string decoding_method{"greedy_search"};
  std::int32_t num_threads{2};
  std::int32_t feature_dim{80};
  std::int32_t max_active_paths{4};
  std::uint32_t sample_rate{kProcessingSampleRate};
  std::size_t queue_capacity{256};
  std::size_t max_active_streams{2};
  std::uint32_t max_pending_audio_ms{60000};
};

struct RecognitionBegin {
  std::string utterance_id;
  std::uint64_t generation{};
  std::string source{"live"};
  std::uint64_t left_backfill_first_sample{};
  SharedPcm left_backfill;
  std::uint32_t sample_rate{kProcessingSampleRate};
};

struct RecognitionChunk {
  std::string utterance_id;
  std::uint64_t generation{};
  std::uint64_t first_sample{};
  SharedPcm pcm;
  std::uint32_t sample_rate{kProcessingSampleRate};
};

struct RecognitionFinalize {
  std::string utterance_id;
  std::uint64_t generation{};
  // Finalization can be submitted without a provisional stream (for example
  // after overload recovery), so the execution source must travel with it.
  std::string source{"live"};
  std::shared_ptr<const UtteranceCandidate> candidate;
};

struct RecognitionCancel {
  std::string utterance_id;
  std::uint64_t generation{};
  std::string reason{"cancelled"};
};

struct AsrHypothesis {
  std::string text;
  std::vector<std::string> tokens;
  std::vector<float> token_timestamps_seconds;
  std::string json;
};

enum class RecognitionResultKind {
  started,
  partial,
  final,
  cancelled,
  error,
};

enum class StreamingRecognizerState {
  loading,
  ready,
  unavailable,
  stopped,
};

struct RecognitionResult {
  RecognitionResultKind kind{RecognitionResultKind::started};
  std::uint64_t sequence{};
  std::string utterance_id;
  std::uint64_t generation{};
  std::uint64_t revision{};
  std::string source{"live"};
  AsrHypothesis hypothesis;
  std::uint64_t audio_start_sample{};
  std::uint64_t audio_end_sample{};
  std::size_t decoded_sample_count{};
  // Queueing plus inference time for the request which produced this result.
  double latency_ms{};
  // Time spent inside the decoder. For partials this is cumulative for the
  // provisional stream; for exact-final it covers only the authoritative pass.
  double inference_ms{};
  double rtf{};
  bool exact_final{};
  std::shared_ptr<const UtteranceCandidate> candidate;
  std::string detail;
};

enum class RecognitionSubmitStatus {
  accepted,
  queue_full,
  pending_audio_full,
  unavailable,
  cancelled_generation,
  invalid_request,
  stopped,
};

struct StreamingRecognizerStats {
  std::uint64_t submitted{};
  std::uint64_t dropped_queue_full{};
  std::uint64_t dropped_pending_audio_full{};
  std::uint64_t rejected_active_stream_limit{};
  std::uint64_t rejected_unavailable{};
  std::uint64_t rejected_stale_generation{};
  std::uint64_t suppressed_stale_results{};
  std::uint64_t partial_results{};
  std::uint64_t final_results{};
  std::uint64_t cancelled_results{};
  std::uint64_t errors{};
  std::uint64_t callback_errors{};
};

class IStreamingRecognizer {
 public:
  virtual ~IStreamingRecognizer() = default;

  // All try_* methods are non-blocking. There must be one lifecycle-request
  // producer; cancel_generation() is lock-free and may be called by any thread.
  virtual RecognitionSubmitStatus try_begin(RecognitionBegin request) noexcept = 0;
  virtual RecognitionSubmitStatus try_accept(RecognitionChunk request) noexcept = 0;
  virtual RecognitionSubmitStatus try_finalize(RecognitionFinalize request) noexcept = 0;
  virtual RecognitionSubmitStatus try_cancel(RecognitionCancel request) noexcept = 0;

  // Generation values are monotonic. Cancelling generation N invalidates N and
  // every older generation, including work already being decoded by the worker.
  virtual void cancel_generation(std::uint64_t generation) noexcept = 0;

  [[nodiscard]] virtual bool available() const noexcept = 0;
  [[nodiscard]] virtual StreamingRecognizerState state() const noexcept = 0;
  [[nodiscard]] virtual std::string status() const = 0;
  [[nodiscard]] virtual std::size_t queue_depth() const noexcept = 0;
  [[nodiscard]] virtual std::size_t pending_requests() const noexcept = 0;
  [[nodiscard]] virtual std::size_t pending_audio_samples() const noexcept = 0;
  // Intended for replay/benchmark/control threads, never the real-time audio
  // producer or the recognizer result callback.
  [[nodiscard]] virtual bool wait_until_idle(std::chrono::milliseconds timeout) = 0;
  [[nodiscard]] virtual bool wait_until_loaded(std::chrono::milliseconds timeout) = 0;
  [[nodiscard]] virtual StreamingRecognizerStats stats() const noexcept = 0;
};

// This small adapter seam keeps the worker lifecycle independently testable and
// allows the project to keep a single sherpa-onnx/ONNX Runtime dependency.
class IOnlineAsrSession {
 public:
  virtual ~IOnlineAsrSession() = default;
  virtual AsrHypothesis accept(std::span<const float> samples,
                               std::uint32_t sample_rate) = 0;
  virtual AsrHypothesis finish() = 0;
};

class IOnlineAsrEngine {
 public:
  virtual ~IOnlineAsrEngine() = default;
  [[nodiscard]] virtual bool available() const noexcept = 0;
  [[nodiscard]] virtual std::string status() const = 0;
  virtual std::unique_ptr<IOnlineAsrSession> create_session() = 0;
};

using RecognitionResultCallback = std::function<void(RecognitionResult)>;

// Passing an engine is intended for deterministic tests. Production callers
// omit it and receive the sherpa-onnx Online Paraformer implementation.
std::unique_ptr<IStreamingRecognizer> create_streaming_recognizer(
    StreamingRecognizerConfig config, RecognitionResultCallback callback,
    std::unique_ptr<IOnlineAsrEngine> engine = {});

[[nodiscard]] const char* to_string(RecognitionResultKind kind) noexcept;
[[nodiscard]] const char* to_string(RecognitionSubmitStatus status) noexcept;
[[nodiscard]] const char* to_string(StreamingRecognizerState state) noexcept;

}  // namespace dvo
