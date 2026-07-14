#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "dvo/segmenter.h"

namespace {

dvo::SegmentationConfig config() {
  dvo::SegmentationConfig c;
  c.wake_guard_ms = 0;
  c.pre_roll_ms = 0;
  c.post_roll_ms = 0;
  c.endpoint_silence_ms = 100;
  c.max_candidate_ms = 5000;
  c.min_command_speech_ms = 150;
  c.embedded_join_silence_ms = 150;
  return c;
}

dvo::KwsHit hit(dvo::SampleSpan wake) {
  return {"小助手", {"x", "iǎo"}, {wake.start, wake.end - 1}, wake, wake.end};
}

std::vector<float> audio(std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) values[i] = static_cast<float>(i % 97) / 97.0F;
  return values;
}

dvo::SegmenterResult run(dvo::VadInterval speech, dvo::SampleSpan wake) {
  dvo::TimedRingBuffer ring(16000 * 10);
  auto values = audio(16000 * 10);
  ring.push(0, values);
  dvo::UtteranceSegmenter segmenter(config(), ring);
  segmenter.add_vad_interval(speech);
  segmenter.set_vad_state(false, speech.span.end);
  segmenter.add_kws_hit(hit(wake));
  return segmenter.advance(std::max(speech.span.end, wake.end) + 1600);
}

}  // namespace

TEST_CASE("segmenter emits prefix candidate without the wake interval") {
  const auto result = run({{1000, 7000}}, {1000, 2500});
  REQUIRE(result.candidates.size() == 1);
  const auto& candidate = result.candidates.front();
  REQUIRE(candidate.position == dvo::WakePosition::prefix);
  REQUIRE(candidate.source_spans == std::vector<dvo::SampleSpan>{{2500, 7000}});
  REQUIRE_FALSE(candidate.source_spans.front().overlaps(candidate.wake_span));
}

TEST_CASE("segmenter emits suffix candidate") {
  const auto result = run({{1000, 7000}}, {5500, 7000});
  REQUIRE(result.candidates.size() == 1);
  REQUIRE(result.candidates.front().position == dvo::WakePosition::suffix);
  REQUIRE(result.candidates.front().source_spans == std::vector<dvo::SampleSpan>{{1000, 5500}});
}

TEST_CASE("segmenter emits embedded candidate with join silence") {
  const auto result = run({{1000, 9000}}, {4000, 5500});
  REQUIRE(result.candidates.size() == 1);
  const auto& candidate = result.candidates.front();
  REQUIRE(candidate.position == dvo::WakePosition::embedded);
  REQUIRE(candidate.source_spans.size() == 2);
  REQUIRE(candidate.pcm.size() == (3000 + 3500 + 2400));
  REQUIRE(std::all_of(candidate.pcm.begin() + 3000, candidate.pcm.begin() + 5400,
                      [](float value) { return value == 0.0F; }));
  REQUIRE_FALSE(candidate.source_spans[0].overlaps(candidate.wake_span));
  REQUIRE_FALSE(candidate.source_spans[1].overlaps(candidate.wake_span));
}

TEST_CASE("segmenter rejects wake-only speech") {
  const auto result = run({{1000, 2500}}, {1000, 2500});
  REQUIRE(result.candidates.empty());
  REQUIRE(result.rejections.size() == 1);
}

TEST_CASE("runtime segmenter path plans spans without copying candidate PCM") {
  dvo::TimedRingBuffer ring(16000 * 10);
  auto values = audio(16000 * 10);
  ring.push(0, values);
  dvo::UtteranceSegmenter segmenter(config(), ring);
  segmenter.add_vad_interval({{1000, 9000}});
  segmenter.set_vad_state(false, 9000);
  segmenter.add_kws_hit(hit({4000, 5500}));

  auto result = segmenter.advance_for_assembly(10600);
  REQUIRE(result.assemblies.size() == 1);
  CHECK(result.assemblies.front().candidate.pcm.empty());
  CHECK(result.assemblies.front().candidate.source_spans ==
        std::vector<dvo::SampleSpan>{{1000, 4000}, {5500, 9000}});
  CHECK(result.assemblies.front().join_silence_samples == 2400);
}

TEST_CASE("provisional ASR backfill includes active VAD speech before the wake") {
  dvo::TimedRingBuffer ring(16000 * 10);
  ring.push(0, audio(16000 * 10));
  dvo::UtteranceSegmenter segmenter(config(), ring);

  // A completed phrase is close enough to the still-active phrase to be one
  // command envelope. The active phrase crosses the KWS hit and therefore has
  // not yet produced a completed VadInterval.
  segmenter.add_vad_interval({{500, 1000}});
  segmenter.set_vad_state(true, 2000);
  const auto start = segmenter.add_kws_hit(hit({4000, 5500}));

  REQUIRE(start.has_value());
  CHECK(start->provisional_left_spans ==
        std::vector<dvo::SampleSpan>{{500, 4000}});
  CHECK_FALSE(start->provisional_left_spans.front().overlaps(start->wake_span));
}

TEST_CASE("segmenter recovers trust after a discontinuity reset") {
  dvo::TimedRingBuffer ring(16000 * 10);
  ring.push(0, audio(16000 * 10));
  dvo::UtteranceSegmenter segmenter(config(), ring);

  segmenter.reset(true);
  segmenter.add_vad_interval({{1000, 7000}});
  segmenter.set_vad_state(false, 7000);
  REQUIRE(segmenter.add_kws_hit(hit({1000, 2500})).has_value());

  const auto result = segmenter.advance(8600);
  REQUIRE(result.candidates.size() == 1);
  CHECK_FALSE(result.candidates.front().discontinuity);
}
