#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dvo/command_parser.h"
#include "dvo/wake_probe.h"

namespace {

using namespace std::chrono_literals;

class ScriptedEngine final : public dvo::IOfflineAsrEngine {
 public:
  explicit ScriptedEngine(std::vector<std::string> texts) : texts_(std::move(texts)) {}
  [[nodiscard]] std::string name() const override { return "scripted"; }
  dvo::AsrHypothesis decode(std::span<const float>, std::uint32_t) override {
    dvo::AsrHypothesis value;
    value.text = next_ < texts_.size() ? texts_[next_++] : std::string{};
    return value;
  }

 private:
  std::vector<std::string> texts_;
  std::size_t next_{};
};

bool wait_for_result(dvo::WakeProbe& probe, dvo::WakeProbeResult& result) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (probe.try_pop(result)) return true;
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

}  // namespace

TEST_CASE("wake word is found in recognized text exactly or as an edge near miss") {
  const auto exact = dvo::find_wake_in_text("播放音乐，小助手。", "小助手");
  REQUIRE(exact);
  CHECK(exact->exact);
  CHECK(exact->begin == 4);
  CHECK(exact->end == 7);
  CHECK(exact->length == 7);

  const auto prefix_miss = dvo::find_wake_in_text("小叔手增加音量", "@小助手");
  REQUIRE(prefix_miss);
  CHECK_FALSE(prefix_miss->exact);
  CHECK(prefix_miss->begin == 0);
  CHECK(prefix_miss->end == 3);

  const auto suffix_short = dvo::find_wake_in_text("暂停音乐小助", "小助手");
  REQUIRE(suffix_short);
  CHECK(suffix_short->begin == 4);
  CHECK(suffix_short->end == 6);

  // Fillers do not count as characters.
  const auto filler = dvo::find_wake_in_text("啊小助手暂停音乐", "小助手");
  REQUIRE(filler);
  CHECK(filler->begin == 0);

  CHECK_FALSE(dvo::find_wake_in_text("播放小叔手音乐", "小助手"));
  CHECK_FALSE(dvo::find_wake_in_text("今天天气怎么样", "小助手"));
  CHECK_FALSE(dvo::find_wake_in_text("暂停音乐小说", "小助手"));
}

TEST_CASE("wake span is estimated from the character position") {
  const dvo::SampleSpan speech{16000, 16000 + 7 * 1600};
  const auto prefix = dvo::estimate_wake_span(speech, 0, 3, 7);
  CHECK(prefix.start == 16000);
  CHECK(prefix.end == 16000 + 3 * 1600);
  const auto suffix = dvo::estimate_wake_span(speech, 4, 7, 7);
  CHECK(suffix.start == 16000 + 4 * 1600);
  CHECK(suffix.end == speech.end);
  CHECK(dvo::estimate_wake_span(speech, 0, 0, 0).start == speech.start);
}

TEST_CASE("wake probe decodes ring audio and reports matches with a wake span") {
  dvo::TimedRingBuffer ring(16000 * 5);
  const std::vector<float> audio(16000 * 3, 0.1F);
  ring.push(0, audio);
  auto engine = std::make_unique<ScriptedEngine>(
      std::vector<std::string>{"小助手，暂停音乐", "今天天气怎么样"});
  dvo::WakeProbe probe([&engine] { return std::move(engine); }, ring, {"小助手"});

  REQUIRE(probe.try_submit({1, {1600, 1600 + 7 * 1600}, {0, 16000 * 2}}));
  dvo::WakeProbeResult matched;
  REQUIRE(wait_for_result(probe, matched));
  CHECK(matched.id == 1);
  CHECK(matched.matched);
  CHECK(matched.exact);
  CHECK(matched.keyword == "小助手");
  REQUIRE(matched.wake_span);
  CHECK(matched.wake_span->start == 1600);
  CHECK(matched.wake_span->end == 1600 + 3 * 1600);
  CHECK(matched.error.empty());

  REQUIRE(probe.try_submit({2, {0, 16000}, {0, 16000}}));
  dvo::WakeProbeResult unmatched;
  REQUIRE(wait_for_result(probe, unmatched));
  CHECK(unmatched.id == 2);
  CHECK_FALSE(unmatched.matched);
  CHECK_FALSE(unmatched.wake_span);

  // Audio that already left the ring is reported, not decoded.
  REQUIRE(probe.try_submit({3, {16000 * 10, 16000 * 11}, {16000 * 10, 16000 * 11}}));
  dvo::WakeProbeResult missing;
  REQUIRE(wait_for_result(probe, missing));
  CHECK_FALSE(missing.matched);
  CHECK_FALSE(missing.error.empty());
  CHECK(probe.idle());
}

TEST_CASE("wake probe queue is bounded and clear drops pending work") {
  dvo::TimedRingBuffer ring(16000);
  // The loader blocks until released, so submissions stay queued.
  std::atomic<bool> release{};
  dvo::WakeProbe probe(
      [&release]() -> std::unique_ptr<dvo::IOfflineAsrEngine> {
        while (!release.load()) std::this_thread::sleep_for(1ms);
        return std::make_unique<ScriptedEngine>(std::vector<std::string>{"小助手"});
      },
      ring, {"小助手"}, 2);
  CHECK(probe.try_submit({1, {0, 100}, {0, 100}}));
  CHECK(probe.try_submit({2, {0, 100}, {0, 100}}));
  CHECK_FALSE(probe.try_submit({3, {0, 100}, {0, 100}}));
  CHECK_FALSE(probe.idle());
  probe.clear();
  CHECK(probe.idle());
  release.store(true);
  std::this_thread::sleep_for(20ms);
  dvo::WakeProbeResult result;
  CHECK_FALSE(probe.try_pop(result));
}
