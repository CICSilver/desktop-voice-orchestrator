#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dvo/streaming_recognizer.h"

namespace {

using namespace std::chrono_literals;

struct FakeEngineState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::vector<float>> sessions;
  bool block_accept{};
  bool accept_entered{};
  bool release_accept{};
  bool block_finish{};
  bool finish_entered{};
  bool release_finish{};
  std::chrono::milliseconds decode_delay{};
};

class FakeSession final : public dvo::IOnlineAsrSession {
 public:
  FakeSession(std::shared_ptr<FakeEngineState> state, std::size_t index)
      : state_(std::move(state)), index_(index) {}

  dvo::AsrHypothesis accept(std::span<const float> samples,
                            std::uint32_t) override {
    std::unique_lock lock(state_->mutex);
    auto& received = state_->sessions.at(index_);
    received.insert(received.end(), samples.begin(), samples.end());
    if (state_->block_accept) {
      state_->accept_entered = true;
      state_->cv.notify_all();
      state_->cv.wait(lock, [this] { return state_->release_accept; });
    }
    const auto delay = state_->decode_delay;
    lock.unlock();
    if (delay.count() > 0) std::this_thread::sleep_for(delay);
    lock.lock();
    return hypothesis("partial", received);
  }

  dvo::AsrHypothesis finish() override {
    std::unique_lock lock(state_->mutex);
    if (state_->block_finish) {
      state_->finish_entered = true;
      state_->cv.notify_all();
      state_->cv.wait(lock, [this] { return state_->release_finish; });
    }
    const auto delay = state_->decode_delay;
    lock.unlock();
    if (delay.count() > 0) std::this_thread::sleep_for(delay);
    lock.lock();
    return hypothesis("final", state_->sessions.at(index_));
  }

 private:
  static dvo::AsrHypothesis hypothesis(const char* prefix,
                                       const std::vector<float>& samples) {
    const auto sum = std::accumulate(samples.begin(), samples.end(), 0.0F);
    std::ostringstream text;
    text << prefix << ':' << samples.size() << ':' << sum;
    dvo::AsrHypothesis value;
    value.text = text.str();
    return value;
  }

  std::shared_ptr<FakeEngineState> state_;
  std::size_t index_{};
};

class FakeEngine final : public dvo::IOnlineAsrEngine {
 public:
  explicit FakeEngine(std::shared_ptr<FakeEngineState> state)
      : state_(std::move(state)) {}

  [[nodiscard]] bool available() const noexcept override { return true; }
  [[nodiscard]] std::string status() const override { return "fake ready"; }

  std::unique_ptr<dvo::IOnlineAsrSession> create_session() override {
    std::lock_guard lock(state_->mutex);
    const auto index = state_->sessions.size();
    state_->sessions.emplace_back();
    return std::make_unique<FakeSession>(state_, index);
  }

 private:
  std::shared_ptr<FakeEngineState> state_;
};

struct ResultCollector {
  void push(dvo::RecognitionResult result) {
    std::lock_guard lock(mutex);
    results.push_back(std::move(result));
    cv.notify_all();
  }

  bool wait_for_kind(dvo::RecognitionResultKind kind,
                     std::chrono::milliseconds timeout = 2s) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, timeout, [&] {
      return std::any_of(results.begin(), results.end(),
                         [kind](const auto& result) { return result.kind == kind; });
    });
  }

  std::vector<dvo::RecognitionResult> snapshot() {
    std::lock_guard lock(mutex);
    return results;
  }

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<dvo::RecognitionResult> results;
};

std::shared_ptr<const std::vector<float>> pcm(std::initializer_list<float> values) {
  return std::make_shared<const std::vector<float>>(values);
}

dvo::StreamingRecognizerConfig fake_config(std::size_t capacity = 16) {
  dvo::StreamingRecognizerConfig config;
  config.queue_capacity = capacity;
  return config;
}

}  // namespace

TEST_CASE("streaming recognizer emits partials and exact-final redecodes candidate PCM") {
  auto state = std::make_shared<FakeEngineState>();
  state->decode_delay = 1ms;
  ResultCollector collector;
  auto recognizer = dvo::create_streaming_recognizer(
      fake_config(), [&collector](dvo::RecognitionResult result) {
        collector.push(std::move(result));
      },
      std::make_unique<FakeEngine>(state));

  dvo::RecognitionBegin begin;
  begin.utterance_id = "utterance-1";
  begin.generation = 1;
  begin.left_backfill_first_sample = 100;
  begin.left_backfill = pcm({1.0F, 2.0F});
  REQUIRE(recognizer->try_begin(std::move(begin)) ==
          dvo::RecognitionSubmitStatus::accepted);

  dvo::RecognitionChunk chunk;
  chunk.utterance_id = "utterance-1";
  chunk.generation = 1;
  chunk.first_sample = 200;
  chunk.pcm = pcm({3.0F, 4.0F, 5.0F});
  REQUIRE(recognizer->try_accept(std::move(chunk)) ==
          dvo::RecognitionSubmitStatus::accepted);

  auto candidate = std::make_shared<dvo::UtteranceCandidate>();
  candidate->utterance_id = "utterance-1";
  candidate->pcm = {9.0F, 9.0F, 9.0F, 9.0F};
  candidate->source_spans = {{300, 304}};
  dvo::RecognitionFinalize finalize;
  finalize.utterance_id = "utterance-1";
  finalize.generation = 1;
  finalize.candidate = candidate;
  REQUIRE(recognizer->try_finalize(std::move(finalize)) ==
          dvo::RecognitionSubmitStatus::accepted);

  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::final));
  const auto results = collector.snapshot();
  const auto final = std::find_if(results.begin(), results.end(), [](const auto& result) {
    return result.kind == dvo::RecognitionResultKind::final;
  });
  REQUIRE(final != results.end());
  CHECK(final->hypothesis.text == "final:4:36");
  CHECK(final->exact_final);
  CHECK(final->candidate == candidate);
  CHECK(final->decoded_sample_count == 4);
  CHECK(final->revision >= 2);
  CHECK(final->latency_ms >= final->inference_ms);
  CHECK(final->inference_ms > 0.0);
  CHECK(final->rtf > 0.0);
  CHECK(final->audio_start_sample == 300);
  CHECK(final->audio_end_sample == 304);
  CHECK(std::is_sorted(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.sequence < rhs.sequence;
  }));

  std::lock_guard lock(state->mutex);
  REQUIRE(state->sessions.size() == 2);
  CHECK((state->sessions[0] ==
         std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F}));
  CHECK((state->sessions[1] == std::vector<float>{9.0F, 9.0F, 9.0F, 9.0F}));
}

TEST_CASE("generation cancellation rejects old work and cancels an active stream") {
  auto state = std::make_shared<FakeEngineState>();
  ResultCollector collector;
  auto recognizer = dvo::create_streaming_recognizer(
      fake_config(), [&collector](dvo::RecognitionResult result) {
        collector.push(std::move(result));
      },
      std::make_unique<FakeEngine>(state));

  recognizer->cancel_generation(7);
  dvo::RecognitionBegin stale;
  stale.utterance_id = "stale";
  stale.generation = 7;
  CHECK(recognizer->try_begin(std::move(stale)) ==
        dvo::RecognitionSubmitStatus::cancelled_generation);

  dvo::RecognitionBegin current;
  current.utterance_id = "current";
  current.generation = 8;
  REQUIRE(recognizer->try_begin(std::move(current)) ==
          dvo::RecognitionSubmitStatus::accepted);
  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::started));

  recognizer->cancel_generation(8);
  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::cancelled));
  const auto results = collector.snapshot();
  CHECK(std::none_of(results.begin(), results.end(), [](const auto& result) {
    return result.kind == dvo::RecognitionResultKind::final;
  }));
  CHECK(recognizer->stats().rejected_stale_generation >= 1);
}

TEST_CASE("generation cancellation suppresses an in-flight exact-final result") {
  auto state = std::make_shared<FakeEngineState>();
  state->block_finish = true;
  ResultCollector collector;
  auto recognizer = dvo::create_streaming_recognizer(
      fake_config(), [&collector](dvo::RecognitionResult result) {
        collector.push(std::move(result));
      },
      std::make_unique<FakeEngine>(state));

  auto candidate = std::make_shared<dvo::UtteranceCandidate>();
  candidate->utterance_id = "in-flight";
  candidate->pcm = {1.0F, 2.0F};
  dvo::RecognitionFinalize finalize;
  finalize.utterance_id = "in-flight";
  finalize.generation = 2;
  finalize.candidate = candidate;
  REQUIRE(recognizer->try_finalize(std::move(finalize)) ==
          dvo::RecognitionSubmitStatus::accepted);

  {
    std::unique_lock lock(state->mutex);
    REQUIRE(state->cv.wait_for(lock, 2s, [&] { return state->finish_entered; }));
  }
  CHECK(recognizer->pending_requests() == 1);
  CHECK(recognizer->pending_audio_samples() == 2);
  CHECK_FALSE(recognizer->wait_until_idle(10ms));
  recognizer->cancel_generation(2);
  {
    std::lock_guard lock(state->mutex);
    state->release_finish = true;
    state->cv.notify_all();
  }
  REQUIRE(recognizer->wait_until_idle(2s));
  CHECK(recognizer->pending_requests() == 0);
  CHECK(recognizer->pending_audio_samples() == 0);

  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (recognizer->stats().suppressed_stale_results == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  CHECK(recognizer->stats().suppressed_stale_results == 1);
  const auto results = collector.snapshot();
  CHECK(std::none_of(results.begin(), results.end(), [](const auto& result) {
    return result.kind == dvo::RecognitionResultKind::final;
  }));
}

TEST_CASE("bounded recognizer queue reports overflow without waiting for worker") {
  auto state = std::make_shared<FakeEngineState>();
  state->block_accept = true;
  auto recognizer = dvo::create_streaming_recognizer(
      fake_config(1), [](dvo::RecognitionResult) {},
      std::make_unique<FakeEngine>(state));

  dvo::RecognitionBegin begin;
  begin.utterance_id = "blocked";
  begin.generation = 1;
  begin.left_backfill = pcm({1.0F});
  REQUIRE(recognizer->try_begin(std::move(begin)) ==
          dvo::RecognitionSubmitStatus::accepted);
  {
    std::unique_lock lock(state->mutex);
    REQUIRE(state->cv.wait_for(lock, 2s, [&] { return state->accept_entered; }));
  }

  dvo::RecognitionChunk first;
  first.utterance_id = "blocked";
  first.generation = 1;
  first.pcm = pcm({2.0F});
  REQUIRE(recognizer->try_accept(std::move(first)) ==
          dvo::RecognitionSubmitStatus::accepted);

  dvo::RecognitionChunk overflow;
  overflow.utterance_id = "blocked";
  overflow.generation = 1;
  overflow.pcm = pcm({3.0F});
  CHECK(recognizer->try_accept(std::move(overflow)) ==
        dvo::RecognitionSubmitStatus::queue_full);
  CHECK(recognizer->stats().dropped_queue_full == 1);

  {
    std::lock_guard lock(state->mutex);
    state->release_accept = true;
    state->cv.notify_all();
  }
  REQUIRE(recognizer->wait_until_idle(2s));
}

TEST_CASE("exact-final keeps replay source without a provisional stream") {
  auto state = std::make_shared<FakeEngineState>();
  ResultCollector collector;
  auto recognizer = dvo::create_streaming_recognizer(
      fake_config(), [&collector](dvo::RecognitionResult result) {
        collector.push(std::move(result));
      },
      std::make_unique<FakeEngine>(state));

  auto candidate = std::make_shared<dvo::UtteranceCandidate>();
  candidate->utterance_id = "replay-final";
  candidate->pcm = {1.0F, 2.0F, 3.0F};
  dvo::RecognitionFinalize finalize;
  finalize.utterance_id = candidate->utterance_id;
  finalize.generation = 1;
  finalize.source = "replay";
  finalize.candidate = candidate;
  REQUIRE(recognizer->try_finalize(std::move(finalize)) ==
          dvo::RecognitionSubmitStatus::accepted);
  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::final));

  const auto results = collector.snapshot();
  const auto final = std::find_if(results.begin(), results.end(), [](const auto& result) {
    return result.kind == dvo::RecognitionResultKind::final;
  });
  REQUIRE(final != results.end());
  CHECK(final->source == "replay");
  CHECK(final->candidate == candidate);
  CHECK(final->exact_final);
}

TEST_CASE("recognizer enforces active-stream and pending-audio bounds") {
  auto state = std::make_shared<FakeEngineState>();
  ResultCollector collector;
  auto config = fake_config();
  config.max_active_streams = 1;
  config.max_pending_audio_ms = 1;  // 16 samples at 16 kHz.
  auto recognizer = dvo::create_streaming_recognizer(
      config, [&collector](dvo::RecognitionResult result) {
        collector.push(std::move(result));
      },
      std::make_unique<FakeEngine>(state));

  dvo::RecognitionBegin first;
  first.utterance_id = "first";
  first.generation = 1;
  REQUIRE(recognizer->try_begin(std::move(first)) ==
          dvo::RecognitionSubmitStatus::accepted);
  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::started));

  dvo::RecognitionBegin second;
  second.utterance_id = "second";
  second.generation = 1;
  REQUIRE(recognizer->try_begin(std::move(second)) ==
          dvo::RecognitionSubmitStatus::accepted);
  REQUIRE(collector.wait_for_kind(dvo::RecognitionResultKind::error));
  CHECK(recognizer->stats().rejected_active_stream_limit == 1);

  dvo::RecognitionChunk oversized;
  oversized.utterance_id = "first";
  oversized.generation = 1;
  oversized.pcm = std::make_shared<const std::vector<float>>(17, 0.0F);
  CHECK(recognizer->try_accept(std::move(oversized)) ==
        dvo::RecognitionSubmitStatus::pending_audio_full);
  CHECK(recognizer->stats().dropped_pending_audio_full == 1);
  CHECK(recognizer->pending_audio_samples() == 0);
}

TEST_CASE("missing Online Paraformer files leave recognizer stably unavailable") {
  dvo::StreamingRecognizerConfig config;
  config.encoder = "missing-asr-encoder.onnx";
  config.decoder = "missing-asr-decoder.onnx";
  config.tokens = "missing-asr-tokens.txt";
  auto recognizer = dvo::create_streaming_recognizer(config, {});

  CHECK_FALSE(recognizer->available());
  CHECK(recognizer->status().find("missing") != std::string::npos);
  dvo::RecognitionBegin begin;
  begin.utterance_id = "unavailable";
  begin.generation = 1;
  CHECK(recognizer->try_begin(std::move(begin)) ==
        dvo::RecognitionSubmitStatus::unavailable);
  CHECK(recognizer->stats().rejected_unavailable == 1);
  CHECK(recognizer->wait_until_idle(10ms));
}

TEST_CASE("downloaded sherpa Online Paraformer model initializes") {
#if !DVO_HAS_SHERPA
  SKIP("core-only build does not include sherpa-onnx");
#else
  const auto model = std::filesystem::path(DVO_PROJECT_ROOT) /
                     "models/sherpa-onnx-streaming-paraformer-zh";
  dvo::StreamingRecognizerConfig config;
  config.encoder = model / "encoder.int8.onnx";
  config.decoder = model / "decoder.int8.onnx";
  config.tokens = model / "tokens.txt";
  config.num_threads = 1;
  if (!std::filesystem::is_regular_file(config.encoder) ||
      !std::filesystem::is_regular_file(config.decoder) ||
      !std::filesystem::is_regular_file(config.tokens)) {
    SKIP("run fetch_models to enable the Online Paraformer integration test");
  }

  auto recognizer = dvo::create_streaming_recognizer(config, {});
  CHECK((recognizer->state() == dvo::StreamingRecognizerState::loading ||
         recognizer->state() == dvo::StreamingRecognizerState::ready));
  REQUIRE(recognizer->wait_until_loaded(std::chrono::seconds(60)));
  REQUIRE(recognizer->available());
  CHECK(recognizer->state() == dvo::StreamingRecognizerState::ready);
  CHECK(recognizer->status().find("Online Paraformer ready") != std::string::npos);
#endif
}
