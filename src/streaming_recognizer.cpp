#include "dvo/streaming_recognizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "dvo/spsc_queue.h"

#ifndef DVO_HAS_SHERPA
#define DVO_HAS_SHERPA 0
#endif

#if DVO_HAS_SHERPA
#include <sherpa-onnx/c-api/c-api.h>
#endif

namespace dvo {
namespace {

class UnavailableAsrEngine final : public IOnlineAsrEngine {
 public:
  explicit UnavailableAsrEngine(std::string reason) : reason_(std::move(reason)) {}

  [[nodiscard]] bool available() const noexcept override { return false; }
  [[nodiscard]] std::string status() const override { return reason_; }
  std::unique_ptr<IOnlineAsrSession> create_session() override { return {}; }

 private:
  std::string reason_;
};

#if DVO_HAS_SHERPA

class SherpaOnlineAsrSession final : public IOnlineAsrSession {
 public:
  explicit SherpaOnlineAsrSession(const SherpaOnnxOnlineRecognizer* recognizer)
      : recognizer_(recognizer), stream_(SherpaOnnxCreateOnlineStream(recognizer)) {
    if (!stream_) throw std::runtime_error("SherpaOnnxCreateOnlineStream failed");
  }

  ~SherpaOnlineAsrSession() override {
    if (stream_) SherpaOnnxDestroyOnlineStream(stream_);
  }

  AsrHypothesis accept(std::span<const float> samples,
                       std::uint32_t sample_rate) override {
    if (finished_) throw std::logic_error("cannot append audio after input finished");
    if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");

    std::size_t offset{};
    constexpr auto kMaxChunk = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    while (offset < samples.size()) {
      const auto count = std::min(kMaxChunk, samples.size() - offset);
      SherpaOnnxOnlineStreamAcceptWaveform(
          stream_, static_cast<std::int32_t>(sample_rate), samples.data() + offset,
          static_cast<std::int32_t>(count));
      offset += count;
      decode_ready();
    }
    return snapshot();
  }

  AsrHypothesis finish() override {
    if (!finished_) {
      SherpaOnnxOnlineStreamInputFinished(stream_);
      finished_ = true;
    }
    decode_ready();
    return snapshot();
  }

 private:
  void decode_ready() {
    while (SherpaOnnxIsOnlineStreamReady(recognizer_, stream_)) {
      SherpaOnnxDecodeOnlineStream(recognizer_, stream_);
    }
  }

  [[nodiscard]] AsrHypothesis snapshot() const {
    AsrHypothesis value;
    const auto* result = SherpaOnnxGetOnlineStreamResult(recognizer_, stream_);
    if (!result) return value;

    if (result->text) value.text = result->text;
    if (result->json) value.json = result->json;
    if (result->count > 0) {
      value.tokens.reserve(static_cast<std::size_t>(result->count));
      if (result->timestamps) {
        value.token_timestamps_seconds.reserve(static_cast<std::size_t>(result->count));
      }
      for (std::int32_t i = 0; i < result->count; ++i) {
        value.tokens.emplace_back(result->tokens_arr && result->tokens_arr[i]
                                      ? result->tokens_arr[i]
                                      : "");
        if (result->timestamps) {
          value.token_timestamps_seconds.push_back(result->timestamps[i]);
        }
      }
    }
    SherpaOnnxDestroyOnlineRecognizerResult(result);
    return value;
  }

  const SherpaOnnxOnlineRecognizer* recognizer_{};
  const SherpaOnnxOnlineStream* stream_{};
  bool finished_{};
};

class SherpaOnlineParaformerEngine final : public IOnlineAsrEngine {
 public:
  explicit SherpaOnlineParaformerEngine(const StreamingRecognizerConfig& config)
      : encoder_(config.encoder.string()),
        decoder_(config.decoder.string()),
        tokens_(config.tokens.string()),
        provider_(config.provider),
        decoding_method_(config.decoding_method) {
    SherpaOnnxOnlineRecognizerConfig c{};
    c.feat_config.sample_rate = static_cast<std::int32_t>(config.sample_rate);
    c.feat_config.feature_dim = config.feature_dim;
    c.model_config.paraformer.encoder = encoder_.c_str();
    c.model_config.paraformer.decoder = decoder_.c_str();
    c.model_config.tokens = tokens_.c_str();
    c.model_config.provider = provider_.c_str();
    c.model_config.num_threads = config.num_threads;
    c.decoding_method = decoding_method_.c_str();
    c.max_active_paths = config.max_active_paths;
    c.enable_endpoint = 0;

    recognizer_ = SherpaOnnxCreateOnlineRecognizer(&c);
    if (!recognizer_) throw std::runtime_error("SherpaOnnxCreateOnlineRecognizer failed");
  }

  ~SherpaOnlineParaformerEngine() override {
    if (recognizer_) SherpaOnnxDestroyOnlineRecognizer(recognizer_);
  }

  [[nodiscard]] bool available() const noexcept override { return recognizer_ != nullptr; }
  [[nodiscard]] std::string status() const override {
    return "sherpa-onnx Online Paraformer ready";
  }

  std::unique_ptr<IOnlineAsrSession> create_session() override {
    return std::make_unique<SherpaOnlineAsrSession>(recognizer_);
  }

 private:
  std::string encoder_;
  std::string decoder_;
  std::string tokens_;
  std::string provider_;
  std::string decoding_method_;
  const SherpaOnnxOnlineRecognizer* recognizer_{};
};

#endif

using RecognitionRequest =
    std::variant<RecognitionBegin, RecognitionChunk, RecognitionFinalize,
                 RecognitionCancel>;

using RecognitionClock = std::chrono::steady_clock;

struct QueuedRecognitionRequest {
  RecognitionRequest value;
  RecognitionClock::time_point submitted_at;
  std::size_t audio_samples{};
};

struct ActiveRecognition {
  std::uint64_t generation{};
  std::string source{"live"};
  std::unique_ptr<IOnlineAsrSession> session;
  AsrHypothesis last_hypothesis;
  std::uint64_t audio_start_sample{};
  std::uint64_t audio_end_sample{};
  std::size_t decoded_sample_count{};
  std::uint64_t revision{};
  double inference_ms{};
  double decoded_audio_seconds{};
  bool has_audio_span{};
};

struct AtomicStats {
  std::atomic<std::uint64_t> submitted{};
  std::atomic<std::uint64_t> dropped_queue_full{};
  std::atomic<std::uint64_t> dropped_pending_audio_full{};
  std::atomic<std::uint64_t> rejected_active_stream_limit{};
  std::atomic<std::uint64_t> rejected_unavailable{};
  std::atomic<std::uint64_t> rejected_stale_generation{};
  std::atomic<std::uint64_t> suppressed_stale_results{};
  std::atomic<std::uint64_t> partial_results{};
  std::atomic<std::uint64_t> final_results{};
  std::atomic<std::uint64_t> cancelled_results{};
  std::atomic<std::uint64_t> errors{};
  std::atomic<std::uint64_t> callback_errors{};
};

using EngineLoader = std::function<std::unique_ptr<IOnlineAsrEngine>()>;

class StreamingRecognizerWorker final : public IStreamingRecognizer {
 public:
  StreamingRecognizerWorker(StreamingRecognizerConfig config,
                            RecognitionResultCallback callback,
                            std::unique_ptr<IOnlineAsrEngine> engine,
                            EngineLoader engine_loader)
      : config_(std::move(config)),
        callback_(std::move(callback)),
        engine_(std::move(engine)),
        engine_loader_(std::move(engine_loader)),
        queue_(std::max<std::size_t>(1, config_.queue_capacity)),
        max_pending_audio_samples_(audio_limit_samples(config_)) {
    if (engine_loader_) {
      set_state(StreamingRecognizerState::loading,
                "loading sherpa-onnx Online Paraformer");
    } else if (engine_ && engine_->available()) {
      set_state(StreamingRecognizerState::ready, engine_->status());
    } else {
      set_state(StreamingRecognizerState::unavailable,
                engine_ ? engine_->status() : "ASR engine is not configured");
    }
    worker_ = std::jthread([this](std::stop_token stop) { worker_loop(stop); });
  }

  ~StreamingRecognizerWorker() override {
    accepting_.store(false, std::memory_order_release);
    set_state(StreamingRecognizerState::stopped, "ASR stopped");
    worker_.request_stop();
    wake_.release();
    if (worker_.joinable()) worker_.join();
  }

  RecognitionSubmitStatus try_begin(RecognitionBegin request) noexcept override {
    if (request.utterance_id.empty() || request.generation == 0 ||
        request.sample_rate == 0) {
      return RecognitionSubmitStatus::invalid_request;
    }
    return submit(RecognitionRequest{std::in_place_type<RecognitionBegin>,
                                     std::move(request)});
  }

  RecognitionSubmitStatus try_accept(RecognitionChunk request) noexcept override {
    if (request.utterance_id.empty() || request.generation == 0 ||
        request.sample_rate == 0 || !request.pcm || request.pcm->empty()) {
      return RecognitionSubmitStatus::invalid_request;
    }
    return submit(RecognitionRequest{std::in_place_type<RecognitionChunk>,
                                     std::move(request)});
  }

  RecognitionSubmitStatus try_finalize(RecognitionFinalize request) noexcept override {
    if (request.utterance_id.empty() || request.generation == 0 ||
        request.source.empty() ||
        !request.candidate || request.candidate->pcm.empty() ||
        request.candidate->sample_rate == 0 ||
        (!request.candidate->utterance_id.empty() &&
         request.candidate->utterance_id != request.utterance_id)) {
      return RecognitionSubmitStatus::invalid_request;
    }
    return submit(RecognitionRequest{std::in_place_type<RecognitionFinalize>,
                                     std::move(request)});
  }

  RecognitionSubmitStatus try_cancel(RecognitionCancel request) noexcept override {
    if (request.utterance_id.empty() || request.generation == 0) {
      return RecognitionSubmitStatus::invalid_request;
    }
    return submit(RecognitionRequest{std::in_place_type<RecognitionCancel>,
                                     std::move(request)});
  }

  void cancel_generation(std::uint64_t generation) noexcept override {
    auto current = cancelled_through_.load(std::memory_order_relaxed);
    while (generation > current &&
           !cancelled_through_.compare_exchange_weak(
               current, generation, std::memory_order_release,
               std::memory_order_relaxed)) {
    }
    wake_.release();
  }

  [[nodiscard]] bool available() const noexcept override {
    return state() == StreamingRecognizerState::ready;
  }
  [[nodiscard]] StreamingRecognizerState state() const noexcept override {
    return state_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::string status() const override {
    std::scoped_lock lock(state_mutex_);
    return status_;
  }
  [[nodiscard]] std::size_t queue_depth() const noexcept override {
    return queue_.size();
  }

  [[nodiscard]] std::size_t pending_requests() const noexcept override {
    return pending_requests_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t pending_audio_samples() const noexcept override {
    return pending_audio_samples_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool wait_until_idle(std::chrono::milliseconds timeout) override {
    std::unique_lock lock(idle_mutex_);
    return idle_cv_.wait_for(lock, timeout, [this] {
      return pending_requests_.load(std::memory_order_acquire) == 0;
    });
  }

  [[nodiscard]] bool wait_until_loaded(std::chrono::milliseconds timeout) override {
    std::unique_lock lock(state_mutex_);
    return state_cv_.wait_for(lock, timeout, [this] {
      return state_.load(std::memory_order_acquire) !=
             StreamingRecognizerState::loading;
    });
  }

  [[nodiscard]] StreamingRecognizerStats stats() const noexcept override {
    StreamingRecognizerStats value;
    value.submitted = stats_.submitted.load(std::memory_order_relaxed);
    value.dropped_queue_full =
        stats_.dropped_queue_full.load(std::memory_order_relaxed);
    value.dropped_pending_audio_full =
        stats_.dropped_pending_audio_full.load(std::memory_order_relaxed);
    value.rejected_active_stream_limit =
        stats_.rejected_active_stream_limit.load(std::memory_order_relaxed);
    value.rejected_unavailable =
        stats_.rejected_unavailable.load(std::memory_order_relaxed);
    value.rejected_stale_generation =
        stats_.rejected_stale_generation.load(std::memory_order_relaxed);
    value.suppressed_stale_results =
        stats_.suppressed_stale_results.load(std::memory_order_relaxed);
    value.partial_results = stats_.partial_results.load(std::memory_order_relaxed);
    value.final_results = stats_.final_results.load(std::memory_order_relaxed);
    value.cancelled_results =
        stats_.cancelled_results.load(std::memory_order_relaxed);
    value.errors = stats_.errors.load(std::memory_order_relaxed);
    value.callback_errors = stats_.callback_errors.load(std::memory_order_relaxed);
    return value;
  }

 private:
  [[nodiscard]] static std::size_t audio_limit_samples(
      const StreamingRecognizerConfig& config) noexcept {
    const auto samples = static_cast<long double>(config.max_pending_audio_ms) *
                         static_cast<long double>(config.sample_rate) / 1000.0L;
    const auto maximum = static_cast<long double>(
        std::numeric_limits<std::size_t>::max());
    return samples >= maximum ? std::numeric_limits<std::size_t>::max()
                              : static_cast<std::size_t>(std::ceil(samples));
  }

  [[nodiscard]] std::size_t request_audio_samples(
      const RecognitionRequest& request) const noexcept {
    return std::visit(
        [this](const auto& value) -> std::size_t {
          using T = std::decay_t<decltype(value)>;
          std::size_t count{};
          std::uint32_t sample_rate{config_.sample_rate};
          if constexpr (std::is_same_v<T, RecognitionBegin>) {
            count = value.backfill ? value.backfill->size() : 0;
            sample_rate = value.sample_rate;
          } else if constexpr (std::is_same_v<T, RecognitionChunk>) {
            count = value.pcm ? value.pcm->size() : 0;
            sample_rate = value.sample_rate;
          } else if constexpr (std::is_same_v<T, RecognitionFinalize>) {
            count = value.candidate ? value.candidate->pcm.size() : 0;
            sample_rate = value.candidate ? value.candidate->sample_rate
                                          : config_.sample_rate;
          }
          if (count == 0 || sample_rate == 0) return 0;
          const auto scaled = static_cast<long double>(count) *
                              static_cast<long double>(config_.sample_rate) /
                              static_cast<long double>(sample_rate);
          const auto maximum = static_cast<long double>(
              std::numeric_limits<std::size_t>::max());
          return scaled >= maximum ? std::numeric_limits<std::size_t>::max()
                                   : static_cast<std::size_t>(std::ceil(scaled));
        },
        request);
  }

  [[nodiscard]] bool generation_is_stale(std::uint64_t generation) const noexcept {
    return generation <= cancelled_through_.load(std::memory_order_acquire);
  }

  RecognitionSubmitStatus submit(RecognitionRequest request) noexcept {
    if (!accepting_.load(std::memory_order_acquire)) {
      return RecognitionSubmitStatus::stopped;
    }
    const auto current_state = state();
    if (current_state == StreamingRecognizerState::stopped) {
      return RecognitionSubmitStatus::stopped;
    }
    if (current_state == StreamingRecognizerState::unavailable) {
      stats_.rejected_unavailable.fetch_add(1, std::memory_order_relaxed);
      return RecognitionSubmitStatus::unavailable;
    }

    const auto generation = std::visit(
        [](const auto& value) { return value.generation; }, request);
    if (generation_is_stale(generation)) {
      stats_.rejected_stale_generation.fetch_add(1, std::memory_order_relaxed);
      return RecognitionSubmitStatus::cancelled_generation;
    }
    const auto audio_samples = request_audio_samples(request);
    if (audio_samples > max_pending_audio_samples_) {
      stats_.dropped_pending_audio_full.fetch_add(1, std::memory_order_relaxed);
      return RecognitionSubmitStatus::pending_audio_full;
    }
    const auto previous_audio =
        pending_audio_samples_.fetch_add(audio_samples, std::memory_order_acq_rel);
    if (previous_audio > max_pending_audio_samples_ - audio_samples) {
      pending_audio_samples_.fetch_sub(audio_samples, std::memory_order_acq_rel);
      stats_.dropped_pending_audio_full.fetch_add(1, std::memory_order_relaxed);
      return RecognitionSubmitStatus::pending_audio_full;
    }

    QueuedRecognitionRequest queued{std::move(request), RecognitionClock::now(),
                                    audio_samples};
    pending_requests_.fetch_add(1, std::memory_order_acq_rel);
    if (!queue_.try_push(std::move(queued))) {
      complete_request(audio_samples);
      stats_.dropped_queue_full.fetch_add(1, std::memory_order_relaxed);
      return RecognitionSubmitStatus::queue_full;
    }
    stats_.submitted.fetch_add(1, std::memory_order_relaxed);
    wake_.release();
    return RecognitionSubmitStatus::accepted;
  }

  void worker_loop(std::stop_token stop) noexcept {
    initialize_engine();
    while (!stop.stop_requested()) {
      wake_.acquire();
      if (stop.stop_requested()) break;

      if (state() == StreamingRecognizerState::unavailable) {
        QueuedRecognitionRequest unavailable;
        while (queue_.try_pop(unavailable)) {
          emit_unavailable(unavailable);
          complete_request(unavailable.audio_samples);
        }
        continue;
      }

      cancel_stale_active();
      QueuedRecognitionRequest queued;
      while (!stop.stop_requested() && queue_.try_pop(queued)) {
        if (request_is_stale(queued.value)) {
          stats_.rejected_stale_generation.fetch_add(1, std::memory_order_relaxed);
          complete_request(queued.audio_samples);
          continue;
        }
        std::visit(
            [this, submitted_at = queued.submitted_at](auto& value) {
              handle(value, submitted_at);
            },
            queued.value);
        complete_request(queued.audio_samples);
        cancel_stale_active();
      }
    }
    QueuedRecognitionRequest abandoned;
    while (queue_.try_pop(abandoned)) complete_request(abandoned.audio_samples);
    active_.clear();
  }

  [[nodiscard]] bool request_is_stale(const RecognitionRequest& request) const {
    return std::visit(
        [this](const auto& value) { return generation_is_stale(value.generation); },
        request);
  }

  void set_state(StreamingRecognizerState value, std::string status) noexcept {
    try {
      std::scoped_lock lock(state_mutex_);
      status_ = std::move(status);
      state_.store(value, std::memory_order_release);
      state_cv_.notify_all();
    } catch (...) {
      state_.store(value, std::memory_order_release);
      state_cv_.notify_all();
    }
  }

  void initialize_engine() noexcept {
    if (!engine_loader_) return;
    try {
      auto loaded = engine_loader_();
      engine_loader_ = {};
      if (!accepting_.load(std::memory_order_acquire)) return;
      engine_ = std::move(loaded);
      if (engine_ && engine_->available()) {
        set_state(StreamingRecognizerState::ready, engine_->status());
      } else {
        set_state(StreamingRecognizerState::unavailable,
                  engine_ ? engine_->status()
                          : "sherpa-onnx Online Paraformer loader returned null");
      }
    } catch (const std::exception& error) {
      engine_loader_ = {};
      set_state(StreamingRecognizerState::unavailable,
                std::string{"failed to initialize sherpa-onnx Online Paraformer: "} +
                    error.what());
    } catch (...) {
      engine_loader_ = {};
      set_state(StreamingRecognizerState::unavailable,
                "failed to initialize sherpa-onnx Online Paraformer");
    }
  }

  void emit_unavailable(const QueuedRecognitionRequest& queued) noexcept {
    std::string utterance_id;
    std::string source{"live"};
    std::uint64_t generation{};
    bool exact_final_attempt{};
    std::shared_ptr<const UtteranceCandidate> candidate;
    std::visit(
        [&](const auto& request) {
          utterance_id = request.utterance_id;
          generation = request.generation;
          using T = std::decay_t<decltype(request)>;
          if constexpr (std::is_same_v<T, RecognitionBegin> ||
                        std::is_same_v<T, RecognitionFinalize>) {
            source = request.source;
          }
          exact_final_attempt = std::is_same_v<T, RecognitionFinalize>;
          if constexpr (std::is_same_v<T, RecognitionFinalize>) {
            candidate = request.candidate;
          }
        },
        queued.value);
    emit_error(utterance_id, generation, source, status(), 0,
               latency_since(queued.submitted_at), exact_final_attempt,
               std::move(candidate));
  }

  [[nodiscard]] static double latency_since(
      RecognitionClock::time_point submitted_at) noexcept {
    return std::chrono::duration<double, std::milli>(
               RecognitionClock::now() - submitted_at)
        .count();
  }

  void complete_request(std::size_t audio_samples) noexcept {
    pending_audio_samples_.fetch_sub(audio_samples, std::memory_order_acq_rel);
    if (pending_requests_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::scoped_lock lock(idle_mutex_);
      idle_cv_.notify_all();
    }
  }

  void handle(RecognitionBegin& request,
              RecognitionClock::time_point submitted_at) noexcept {
    try {
      if (const auto existing = active_.find(request.utterance_id);
          existing != active_.end()) {
        emit_cancelled(existing->first, existing->second, "superseded by begin");
        active_.erase(existing);
      }
      if (active_.size() >= config_.max_active_streams) {
        stats_.rejected_active_stream_limit.fetch_add(1,
                                                       std::memory_order_relaxed);
        emit_error(request.utterance_id, request.generation, request.source,
                   "ASR active stream limit reached", 0,
                   latency_since(submitted_at));
        return;
      }

      ActiveRecognition value;
      value.generation = request.generation;
      value.source = request.source;
      value.session = engine_->create_session();
      if (!value.session) throw std::runtime_error("ASR session creation failed");

      auto [it, inserted] = active_.emplace(request.utterance_id, std::move(value));
      if (!inserted) throw std::runtime_error("failed to register ASR session");

      RecognitionResult started;
      started.kind = RecognitionResultKind::started;
      started.utterance_id = request.utterance_id;
      started.generation = request.generation;
      started.source = request.source;
      started.latency_ms = latency_since(submitted_at);
      emit(std::move(started));

      if (request.backfill && !request.backfill->empty()) {
        feed(request.utterance_id, it->second, request.backfill_first_sample,
             *request.backfill, request.sample_rate, submitted_at);
      }
    } catch (const std::exception& error) {
      active_.erase(request.utterance_id);
      emit_error(request.utterance_id, request.generation, request.source,
                 error.what(), 0, latency_since(submitted_at));
    } catch (...) {
      active_.erase(request.utterance_id);
      emit_error(request.utterance_id, request.generation, request.source,
                 "unknown ASR begin error", 0, latency_since(submitted_at));
    }
  }

  void handle(RecognitionChunk& request,
              RecognitionClock::time_point submitted_at) noexcept {
    auto it = active_.find(request.utterance_id);
    if (it == active_.end() || it->second.generation != request.generation) return;
    try {
      feed(request.utterance_id, it->second, request.first_sample, *request.pcm,
           request.sample_rate, submitted_at);
    } catch (const std::exception& error) {
      const auto source = it->second.source;
      const auto revision = it->second.revision + 1;
      active_.erase(it);
      emit_error(request.utterance_id, request.generation, source, error.what(),
                 revision, latency_since(submitted_at));
    } catch (...) {
      const auto source = it->second.source;
      const auto revision = it->second.revision + 1;
      active_.erase(it);
      emit_error(request.utterance_id, request.generation, source,
                 "unknown ASR chunk error", revision,
                 latency_since(submitted_at));
    }
  }

  void handle(RecognitionFinalize& request,
              RecognitionClock::time_point submitted_at) noexcept {
    std::string source = request.source;
    std::uint64_t final_revision{1};
    if (const auto it = active_.find(request.utterance_id); it != active_.end()) {
      if (it->second.generation == request.generation) {
        final_revision = it->second.revision + 1;
        // Never reuse provisional decoder state for exact-final. A different
        // generation with the same ID is unrelated and must remain active.
        active_.erase(it);
      }
    }

    try {
      const auto started = std::chrono::steady_clock::now();
      auto session = engine_->create_session();
      if (!session) throw std::runtime_error("exact-final ASR session creation failed");
      (void)session->accept(request.candidate->pcm, request.candidate->sample_rate);
      auto hypothesis = session->finish();
      const auto inference_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();

      RecognitionResult result;
      result.kind = RecognitionResultKind::final;
      result.utterance_id = request.utterance_id;
      result.generation = request.generation;
      result.revision = final_revision;
      result.source = std::move(source);
      result.hypothesis = std::move(hypothesis);
      result.decoded_sample_count = request.candidate->pcm.size();
      result.latency_ms = latency_since(submitted_at);
      result.inference_ms = inference_ms;
      const auto audio_ms = static_cast<double>(request.candidate->pcm.size()) * 1000.0 /
                            static_cast<double>(request.candidate->sample_rate);
      result.rtf = audio_ms > 0.0 ? inference_ms / audio_ms : 0.0;
      result.exact_final_attempt = true;
      result.exact_final = true;
      result.candidate = request.candidate;
      if (!request.candidate->source_spans.empty()) {
        result.audio_start_sample = request.candidate->source_spans.front().start;
        result.audio_end_sample = request.candidate->source_spans.front().end;
        for (const auto& span : request.candidate->source_spans) {
          result.audio_start_sample = std::min(result.audio_start_sample, span.start);
          result.audio_end_sample = std::max(result.audio_end_sample, span.end);
        }
      }
      emit(std::move(result));
    } catch (const std::exception& error) {
      emit_error(request.utterance_id, request.generation, source, error.what(),
                 final_revision, latency_since(submitted_at), true,
                 request.candidate);
    } catch (...) {
      emit_error(request.utterance_id, request.generation, source,
                 "unknown exact-final ASR error", final_revision,
                 latency_since(submitted_at), true, request.candidate);
    }
  }

  void handle(RecognitionCancel& request,
              RecognitionClock::time_point submitted_at) noexcept {
    const auto it = active_.find(request.utterance_id);
    if (it == active_.end() || it->second.generation != request.generation) return;
    emit_cancelled(it->first, it->second, request.reason,
                   latency_since(submitted_at));
    active_.erase(it);
  }

  void feed(const std::string& utterance_id, ActiveRecognition& active,
            std::uint64_t first_sample, const std::vector<float>& pcm,
            std::uint32_t sample_rate,
            RecognitionClock::time_point submitted_at) {
    if (pcm.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    auto hypothesis = active.session->accept(pcm, sample_rate);
    active.inference_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const auto available_to_end =
        std::numeric_limits<std::uint64_t>::max() - first_sample;
    const auto pcm_size = static_cast<std::uint64_t>(std::min<std::size_t>(
        pcm.size(), static_cast<std::size_t>(available_to_end)));
    const auto end_sample = first_sample + pcm_size;
    if (!active.has_audio_span) {
      active.audio_start_sample = first_sample;
      active.audio_end_sample = end_sample;
      active.has_audio_span = true;
    } else {
      active.audio_start_sample = std::min(active.audio_start_sample, first_sample);
      active.audio_end_sample = std::max(active.audio_end_sample, end_sample);
    }
    active.decoded_sample_count += pcm.size();
    active.decoded_audio_seconds +=
        static_cast<double>(pcm.size()) / static_cast<double>(sample_rate);

    if (hypothesis.text != active.last_hypothesis.text) {
      active.last_hypothesis = hypothesis;
      RecognitionResult result;
      result.kind = RecognitionResultKind::partial;
      result.utterance_id = utterance_id;
      result.generation = active.generation;
      result.revision = ++active.revision;
      result.source = active.source;
      result.hypothesis = std::move(hypothesis);
      result.audio_start_sample = active.audio_start_sample;
      result.audio_end_sample = active.audio_end_sample;
      result.decoded_sample_count = active.decoded_sample_count;
      result.latency_ms = latency_since(submitted_at);
      result.inference_ms = active.inference_ms;
      const auto audio_ms = active.decoded_audio_seconds * 1000.0;
      result.rtf = audio_ms > 0.0 ? active.inference_ms / audio_ms : 0.0;
      emit(std::move(result));
    }
  }

  void cancel_stale_active() noexcept {
    for (auto it = active_.begin(); it != active_.end();) {
      if (generation_is_stale(it->second.generation)) {
        emit_cancelled(it->first, it->second, "generation cancelled");
        it = active_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void emit_cancelled(const std::string& utterance_id,
                      const ActiveRecognition& active,
                      const std::string& reason,
                      double latency_ms = 0.0) noexcept {
    RecognitionResult result;
    result.kind = RecognitionResultKind::cancelled;
    result.utterance_id = utterance_id;
    result.generation = active.generation;
    result.revision = active.revision + 1;
    result.source = active.source;
    result.audio_start_sample = active.audio_start_sample;
    result.audio_end_sample = active.audio_end_sample;
    result.decoded_sample_count = active.decoded_sample_count;
    result.latency_ms = latency_ms;
    result.inference_ms = active.inference_ms;
    const auto audio_ms = active.decoded_audio_seconds * 1000.0;
    result.rtf = audio_ms > 0.0 ? active.inference_ms / audio_ms : 0.0;
    result.detail = reason;
    emit(std::move(result));
  }

  void emit_error(const std::string& utterance_id, std::uint64_t generation,
                  const std::string& source, std::string detail,
                  std::uint64_t revision, double latency_ms,
                  bool exact_final_attempt = false,
                  std::shared_ptr<const UtteranceCandidate> candidate = {}) noexcept {
    RecognitionResult result;
    result.kind = RecognitionResultKind::error;
    result.utterance_id = utterance_id;
    result.generation = generation;
    result.revision = revision;
    result.source = source;
    result.latency_ms = latency_ms;
    result.exact_final_attempt = exact_final_attempt;
    result.candidate = std::move(candidate);
    if (result.candidate && !result.candidate->source_spans.empty()) {
      result.audio_start_sample = result.candidate->source_spans.front().start;
      result.audio_end_sample = result.candidate->source_spans.back().end;
    }
    result.detail = std::move(detail);
    emit(std::move(result));
  }

  void emit(RecognitionResult result) noexcept {
    if (result.kind != RecognitionResultKind::cancelled &&
        generation_is_stale(result.generation)) {
      stats_.suppressed_stale_results.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    result.sequence = next_result_sequence_++;
    switch (result.kind) {
      case RecognitionResultKind::partial:
        stats_.partial_results.fetch_add(1, std::memory_order_relaxed);
        break;
      case RecognitionResultKind::final:
        stats_.final_results.fetch_add(1, std::memory_order_relaxed);
        break;
      case RecognitionResultKind::cancelled:
        stats_.cancelled_results.fetch_add(1, std::memory_order_relaxed);
        break;
      case RecognitionResultKind::error:
        stats_.errors.fetch_add(1, std::memory_order_relaxed);
        break;
      case RecognitionResultKind::started:
        break;
    }

    if (!callback_) return;
    try {
      callback_(std::move(result));
    } catch (...) {
      stats_.callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
  }

  StreamingRecognizerConfig config_;
  RecognitionResultCallback callback_;
  std::unique_ptr<IOnlineAsrEngine> engine_;
  EngineLoader engine_loader_;
  std::atomic<StreamingRecognizerState> state_{
      StreamingRecognizerState::unavailable};
  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  std::string status_{"ASR engine is not configured"};
  SpscQueue<QueuedRecognitionRequest> queue_;
  const std::size_t max_pending_audio_samples_{};
  std::counting_semaphore<> wake_{0};
  std::atomic<bool> accepting_{true};
  std::atomic<std::uint64_t> cancelled_through_{};
  std::atomic<std::size_t> pending_requests_{};
  std::atomic<std::size_t> pending_audio_samples_{};
  std::mutex idle_mutex_;
  std::condition_variable idle_cv_;
  AtomicStats stats_;
  std::unordered_map<std::string, ActiveRecognition> active_;
  std::uint64_t next_result_sequence_{1};
  std::jthread worker_;
};

[[nodiscard]] std::string validate_common_config(
    const StreamingRecognizerConfig& config) {
  if (config.queue_capacity == 0) return "ASR queue_capacity must be greater than 0";
  if (config.max_active_streams == 0) {
    return "ASR max_active_streams must be greater than 0";
  }
  if (config.max_pending_audio_ms == 0) {
    return "ASR max_pending_audio_ms must be greater than 0";
  }
  if (config.sample_rate == 0) return "ASR sample_rate must be greater than 0";
  if (config.feature_dim <= 0) return "ASR feature_dim must be greater than 0";
  if (config.num_threads <= 0) return "ASR num_threads must be greater than 0";
  if (config.max_active_paths <= 0) {
    return "ASR max_active_paths must be greater than 0";
  }
  if (config.provider.empty()) return "ASR provider must not be empty";
  if (config.decoding_method.empty()) return "ASR decoding_method must not be empty";
  return {};
}

[[nodiscard]] std::string missing_model_reason(
    const StreamingRecognizerConfig& config) {
  for (const auto* entry : {&config.encoder, &config.decoder, &config.tokens}) {
    std::error_code error;
    if (entry->empty() || !std::filesystem::is_regular_file(*entry, error) || error) {
      return "ASR model/config file missing: " + entry->string();
    }
  }
  return {};
}

}  // namespace

std::unique_ptr<IStreamingRecognizer> create_streaming_recognizer(
    StreamingRecognizerConfig config, RecognitionResultCallback callback,
    std::unique_ptr<IOnlineAsrEngine> engine) {
  EngineLoader engine_loader;
  if (!config.enabled) {
    engine = std::make_unique<UnavailableAsrEngine>(
        "ASR disabled by configuration");
  } else if (const auto invalid = validate_common_config(config); !invalid.empty()) {
    engine = std::make_unique<UnavailableAsrEngine>(invalid);
  } else if (!engine) {
    if (const auto missing = missing_model_reason(config); !missing.empty()) {
      engine = std::make_unique<UnavailableAsrEngine>(missing);
    } else {
#if DVO_HAS_SHERPA
      auto loader_config = config;
      engine_loader = [config = std::move(loader_config)]() {
        return std::make_unique<SherpaOnlineParaformerEngine>(config);
      };
#else
      engine = std::make_unique<UnavailableAsrEngine>(
          "built without sherpa-onnx");
#endif
    }
  }

  return std::make_unique<StreamingRecognizerWorker>(
      std::move(config), std::move(callback), std::move(engine),
      std::move(engine_loader));
}

std::unique_ptr<IOnlineAsrEngine> create_online_asr_engine(
    const StreamingRecognizerConfig& config) {
  if (const auto invalid = validate_common_config(config); !invalid.empty()) {
    throw std::invalid_argument(invalid);
  }
  if (const auto missing = missing_model_reason(config); !missing.empty()) {
    throw std::runtime_error(missing);
  }
#if DVO_HAS_SHERPA
  return std::make_unique<SherpaOnlineParaformerEngine>(config);
#else
  throw std::runtime_error("built without sherpa-onnx");
#endif
}

const char* to_string(RecognitionResultKind kind) noexcept {
  switch (kind) {
    case RecognitionResultKind::started: return "started";
    case RecognitionResultKind::partial: return "partial";
    case RecognitionResultKind::final: return "final";
    case RecognitionResultKind::cancelled: return "cancelled";
    case RecognitionResultKind::error: return "error";
  }
  return "unknown";
}

const char* to_string(RecognitionSubmitStatus status) noexcept {
  switch (status) {
    case RecognitionSubmitStatus::accepted: return "accepted";
    case RecognitionSubmitStatus::queue_full: return "queue_full";
    case RecognitionSubmitStatus::pending_audio_full: return "pending_audio_full";
    case RecognitionSubmitStatus::unavailable: return "unavailable";
    case RecognitionSubmitStatus::cancelled_generation:
      return "cancelled_generation";
    case RecognitionSubmitStatus::invalid_request: return "invalid_request";
    case RecognitionSubmitStatus::stopped: return "stopped";
  }
  return "unknown";
}

const char* to_string(StreamingRecognizerState state) noexcept {
  switch (state) {
    case StreamingRecognizerState::loading: return "loading";
    case StreamingRecognizerState::ready: return "ready";
    case StreamingRecognizerState::unavailable: return "unavailable";
    case StreamingRecognizerState::stopped: return "stopped";
  }
  return "unknown";
}

}  // namespace dvo
