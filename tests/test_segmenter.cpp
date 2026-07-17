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
  static_cast<void>(segmenter.set_vad_state(false, speech.span.end));
  static_cast<void>(segmenter.add_kws_hit(hit(wake)));
  return segmenter.advance(std::max(speech.span.end, wake.end) + 1600);
}

}  // namespace

TEST_CASE("segmenter emits prefix candidate without the wake interval") {
  const auto result = run({{1000, 7000}}, {1000, 2500});
  REQUIRE(result.candidates.size() == 1);
  const auto& candidate = result.candidates.front();
  REQUIRE(candidate.position == dvo::WakePosition::prefix);
  REQUIRE(candidate.source_spans == std::vector<dvo::SampleSpan>{{2500, 7000}});
  REQUIRE(candidate.wake_span.has_value());
  REQUIRE_FALSE(candidate.source_spans.front().overlaps(*candidate.wake_span));
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
  REQUIRE(candidate.wake_span.has_value());
  REQUIRE_FALSE(candidate.source_spans[0].overlaps(*candidate.wake_span));
  REQUIRE_FALSE(candidate.source_spans[1].overlaps(*candidate.wake_span));
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
  static_cast<void>(segmenter.set_vad_state(false, 9000));
  static_cast<void>(segmenter.add_kws_hit(hit({4000, 5500})));

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
  static_cast<void>(segmenter.set_vad_state(true, 2000));
  const auto start = segmenter.add_kws_hit(hit({4000, 5500}));

  REQUIRE(start.has_value());
  CHECK(start->backfill_spans ==
        std::vector<dvo::SampleSpan>{{500, 4000}});
  REQUIRE(start->wake_span.has_value());
  CHECK_FALSE(start->backfill_spans.front().overlaps(*start->wake_span));
}

TEST_CASE("segmenter recovers trust after a discontinuity reset") {
  dvo::TimedRingBuffer ring(16000 * 10);
  ring.push(0, audio(16000 * 10));
  dvo::UtteranceSegmenter segmenter(config(), ring);

  segmenter.reset(true);
  segmenter.add_vad_interval({{1000, 7000}});
  static_cast<void>(segmenter.set_vad_state(false, 7000));
  REQUIRE(segmenter.add_kws_hit(hit({1000, 2500})).has_value());

  const auto result = segmenter.advance(8600);
  REQUIRE(result.candidates.size() == 1);
  CHECK_FALSE(result.candidates.front().discontinuity);
}

namespace {

dvo::SegmentationConfig activation_segmentation() {
  auto value = config();
  value.endpoint_silence_ms = 900;
  value.min_command_speech_ms = 100;
  value.pre_roll_ms = 0;
  value.post_roll_ms = 0;
  return value;
}

dvo::ActivationConfig activation_config() {
  dvo::ActivationConfig value;
  value.enabled = true;
  value.idle_timeout_ms = 6000;
  value.hard_limit_ms = 20000;
  return value;
}

constexpr std::uint64_t ms(std::uint64_t value) { return value * 16; }

struct ActivationFixture {
  dvo::TimedRingBuffer ring{16000 * 10};
  dvo::UtteranceSegmenter segmenter{activation_segmentation(),
                                    activation_config(), ring};

  ActivationFixture() { ring.push(0, audio(16000 * 10)); }

  dvo::SegmenterResult finish_keyword(std::uint64_t speech_end = ms(500)) {
    segmenter.add_vad_interval({{ms(100), speech_end}});
    static_cast<void>(segmenter.set_vad_state(false, speech_end));
    REQUIRE(segmenter.add_kws_hit(hit({ms(100), ms(200)})).has_value());
    return segmenter.advance(speech_end + ms(900));
  }
};

}  // namespace

TEST_CASE("activation merges speech resuming at or before the endpoint") {
  for (const auto gap_ms : {890ULL, 900ULL}) {
    dvo::TimedRingBuffer ring(16000 * 10);
    ring.push(0, audio(16000 * 10));
    dvo::UtteranceSegmenter segmenter(activation_segmentation(),
                                      activation_config(), ring);
    const auto first_end = ms(500);
    const auto second_start = first_end + ms(gap_ms);
    const auto second_end = second_start + ms(300);
    segmenter.add_vad_interval({{ms(100), first_end}});
    static_cast<void>(segmenter.set_vad_state(false, first_end));
    REQUIRE(segmenter.add_kws_hit(hit({ms(100), ms(200)})).has_value());

    // Speech onset is delivered before the endpoint check for this frame.
    CHECK_FALSE(segmenter.set_vad_state(true, second_start).has_value());
    static_cast<void>(segmenter.set_vad_state(false, second_end));
    segmenter.add_vad_interval({{second_start, second_end}});
    const auto result = segmenter.advance(second_end + ms(900));

    CAPTURE(gap_ms);
    REQUIRE(result.candidates.size() == 1);
    CHECK(result.candidates.front().origin == dvo::UtteranceOrigin::keyword);
    CHECK(result.candidates.front().turn_index == 0);
    REQUIRE_FALSE(result.candidates.front().source_spans.empty());
    CHECK(result.candidates.front().source_spans.back().end == second_end);
    CHECK(segmenter.activation().state ==
          dvo::SegmenterActivationState::armed_idle);
  }
}

TEST_CASE("activation splits speech that starts after the endpoint") {
  ActivationFixture fixture;
  const auto first = fixture.finish_keyword();
  REQUIRE(first.candidates.size() == 1);
  REQUIRE_FALSE(first.candidates.front().source_spans.empty());
  CHECK(first.candidates.front().source_spans.back().end == ms(500));
  REQUIRE(fixture.segmenter.activation().state ==
          dvo::SegmenterActivationState::armed_idle);

  const auto followup_start = ms(500) + ms(910);
  const auto start = fixture.segmenter.set_vad_state(true, followup_start);
  REQUIRE(start.has_value());
  CHECK(start->origin == dvo::UtteranceOrigin::followup);
  CHECK(start->turn_index == 1);
  CHECK_FALSE(start->wake_span.has_value());

  const auto followup_end = followup_start + ms(300);
  static_cast<void>(fixture.segmenter.set_vad_state(false, followup_end));
  fixture.segmenter.add_vad_interval({{followup_start, followup_end}});
  const auto second = fixture.segmenter.advance(followup_end + ms(900));
  REQUIRE(second.candidates.size() == 1);
  CHECK(second.candidates.front().origin == dvo::UtteranceOrigin::followup);
  CHECK(second.candidates.front().turn_index == 1);
  CHECK_FALSE(second.candidates.front().wake_span.has_value());
  CHECK_FALSE(second.candidates.front().position.has_value());
  REQUIRE_FALSE(second.candidates.front().source_spans.empty());
  CHECK(second.candidates.front().source_spans.back().end == followup_end);
}

TEST_CASE("prefix suffix and embedded keyword turns all admit follow-ups") {
  struct KeywordShape {
    dvo::SampleSpan wake;
    dvo::WakePosition position;
  };
  const std::vector<KeywordShape> shapes{
      {{ms(100), ms(200)}, dvo::WakePosition::prefix},
      {{ms(400), ms(500)}, dvo::WakePosition::suffix},
      {{ms(250), ms(350)}, dvo::WakePosition::embedded},
  };

  for (const auto& shape : shapes) {
    dvo::TimedRingBuffer ring(16000 * 10);
    ring.push(0, audio(16000 * 10));
    dvo::UtteranceSegmenter segmenter(activation_segmentation(),
                                      activation_config(), ring);
    segmenter.add_vad_interval({{ms(100), ms(500)}});
    static_cast<void>(segmenter.set_vad_state(false, ms(500)));
    REQUIRE(segmenter.add_kws_hit(hit(shape.wake)).has_value());

    const auto keyword = segmenter.advance(ms(1400));
    CAPTURE(to_string(shape.position));
    REQUIRE(keyword.candidates.size() == 1);
    CHECK(keyword.candidates.front().position == shape.position);
    REQUIRE(segmenter.activation().state ==
            dvo::SegmenterActivationState::armed_idle);

    const auto followup = segmenter.set_vad_state(true, ms(1410));
    REQUIRE(followup.has_value());
    CHECK(followup->origin == dvo::UtteranceOrigin::followup);
    CHECK(followup->activation_id ==
          keyword.candidates.front().activation_id);
    CHECK(followup->turn_index == 1);
  }
}

TEST_CASE("wake-only utterance arms follow-up activation") {
  dvo::TimedRingBuffer ring(16000 * 10);
  ring.push(0, audio(16000 * 10));
  dvo::UtteranceSegmenter segmenter(activation_segmentation(),
                                    activation_config(), ring);
  segmenter.add_vad_interval({{ms(100), ms(200)}});
  static_cast<void>(segmenter.set_vad_state(false, ms(200)));
  REQUIRE(segmenter.add_kws_hit(hit({ms(100), ms(200)})).has_value());

  const auto result = segmenter.advance(ms(1100));
  CHECK(result.candidates.empty());
  REQUIRE(result.rejections.size() == 1);
  CHECK(segmenter.activation().state ==
        dvo::SegmenterActivationState::armed_idle);
  CHECK(segmenter.activation().idle_deadline_sample == ms(7100));

  const auto followup = segmenter.set_vad_state(true, ms(1200));
  REQUIRE(followup.has_value());
  CHECK(followup->origin == dvo::UtteranceOrigin::followup);
}

TEST_CASE("activation expires on the deterministic sample-clock deadline") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto deadline = fixture.segmenter.activation().idle_deadline_sample;

  const auto before = fixture.segmenter.advance_for_assembly(deadline - 1);
  CHECK(before.activation_events.empty());
  CHECK(fixture.segmenter.activation().active());

  const auto at_deadline = fixture.segmenter.advance_for_assembly(deadline);
  REQUIRE(at_deadline.activation_events.size() == 1);
  CHECK(at_deadline.activation_events.front().kind ==
        dvo::ActivationTransitionKind::expired);
  CHECK(at_deadline.activation_events.front().reason == "idle_timeout");
  CHECK_FALSE(fixture.segmenter.activation().active());
}

TEST_CASE("activation hard deadline is absolute") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto hard_deadline = fixture.segmenter.activation().hard_deadline_sample;

  const auto result = fixture.segmenter.advance_for_assembly(hard_deadline);
  REQUIRE(result.activation_events.size() == 1);
  CHECK(result.activation_events.front().kind ==
        dvo::ActivationTransitionKind::expired);
  CHECK(result.activation_events.front().reason == "hard_limit");
  CHECK_FALSE(fixture.segmenter.activation().active());
  CHECK_FALSE(fixture.segmenter.defer_activation_for_playback(ms(1000),
                                                               hard_deadline));
}

TEST_CASE("speech beginning exactly at hard deadline is not admitted") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto hard_deadline = fixture.segmenter.activation().hard_deadline_sample;

  CHECK_FALSE(fixture.segmenter.set_vad_state(true, hard_deadline).has_value());
  CHECK(fixture.segmenter.pending_count() == 0);
  CHECK_FALSE(fixture.segmenter.activation().active());
  const auto events = fixture.segmenter.take_activation_events();
  REQUIRE(events.size() == 1);
  CHECK(events.front().kind == dvo::ActivationTransitionKind::expired);
  CHECK(events.front().reason == "hard_limit");
}

TEST_CASE("speech begun before the hard deadline can finish and validate after it") {
  dvo::TimedRingBuffer ring(16000 * 30);
  ring.push(0, audio(16000 * 10));
  dvo::UtteranceSegmenter segmenter(activation_segmentation(),
                                    activation_config(), ring);
  segmenter.add_vad_interval({{ms(100), ms(500)}});
  static_cast<void>(segmenter.set_vad_state(false, ms(500)));
  REQUIRE(segmenter.add_kws_hit(hit({ms(100), ms(200)})).has_value());
  REQUIRE(segmenter.advance(ms(1400)).candidates.size() == 1);

  const auto close_and_refresh = [&](std::uint64_t start_sample,
                                     std::uint64_t end_sample) {
    const auto start = segmenter.set_vad_state(true, start_sample);
    REQUIRE(start.has_value());
    static_cast<void>(segmenter.set_vad_state(false, end_sample));
    segmenter.add_vad_interval({{start_sample, end_sample}});
    const auto completed = segmenter.advance(end_sample + ms(900));
    REQUIRE(completed.candidates.size() == 1);
    REQUIRE(segmenter.refresh_activation(start->activation_id,
                                         start->turn_index, end_sample));
  };

  // Two accepted turns carry the idle window up to, but never beyond, hard.
  close_and_refresh(ms(7300), ms(7500));
  ring.push(ms(10000), audio(16000 * 10));
  close_and_refresh(ms(14300), ms(14500));
  const auto hard_deadline = segmenter.activation().hard_deadline_sample;
  REQUIRE(segmenter.activation().idle_deadline_sample == hard_deadline);

  ring.push(ms(20000), audio(16000 * 5));
  const auto final_start = segmenter.set_vad_state(true, hard_deadline - 1);
  REQUIRE(final_start.has_value());
  const auto final_end = hard_deadline + ms(300);
  static_cast<void>(segmenter.set_vad_state(false, final_end));
  segmenter.add_vad_interval({{hard_deadline - 1, final_end}});
  const auto completed = segmenter.advance(final_end + ms(900));
  REQUIRE(completed.candidates.size() == 1);
  CHECK_FALSE(segmenter.activation().active());
  CHECK(segmenter.followup_plan_eligible(final_start->activation_id,
                                         final_start->turn_index));
  REQUIRE(segmenter.refresh_activation(final_start->activation_id,
                                       final_start->turn_index, final_end));
  CHECK_FALSE(segmenter.activation().active());
}

TEST_CASE("valid follow-up plan refreshes idle but not hard deadline") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto hard_deadline = fixture.segmenter.activation().hard_deadline_sample;
  const auto followup_start = ms(1500);
  const auto start = fixture.segmenter.set_vad_state(true, followup_start);
  REQUIRE(start.has_value());
  static_cast<void>(fixture.segmenter.set_vad_state(false, ms(1800)));
  fixture.segmenter.add_vad_interval({{followup_start, ms(1800)}});
  const auto result = fixture.segmenter.advance(ms(2700));
  REQUIRE(result.candidates.size() == 1);

  REQUIRE(fixture.segmenter.refresh_activation(
      start->activation_id, start->turn_index, ms(1800)));
  CHECK(fixture.segmenter.activation().idle_deadline_sample == ms(8700));
  CHECK(fixture.segmenter.activation().hard_deadline_sample == hard_deadline);
  const auto events = fixture.segmenter.take_activation_events();
  REQUIRE(events.size() == 1);
  CHECK(events.front().kind == dvo::ActivationTransitionKind::refreshed);
  CHECK(events.front().at_sample == ms(2700));
}

TEST_CASE("follow-up begun before idle deadline remains refreshable after it") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto old_deadline = fixture.segmenter.activation().idle_deadline_sample;
  const auto followup_start = old_deadline - ms(100);
  const auto start = fixture.segmenter.set_vad_state(true, followup_start);
  REQUIRE(start.has_value());
  const auto followup_end = old_deadline + ms(400);
  static_cast<void>(fixture.segmenter.set_vad_state(false, followup_end));
  fixture.segmenter.add_vad_interval({{followup_start, followup_end}});

  const auto completed = fixture.segmenter.advance(followup_end + ms(900));
  REQUIRE(completed.candidates.size() == 1);
  CHECK(fixture.segmenter.awaiting_validation_count() == 1);
  CHECK(fixture.segmenter.activation().active());

  const auto while_validating =
      fixture.segmenter.advance_for_assembly(followup_end + ms(1200));
  CHECK(while_validating.activation_events.empty());
  CHECK(fixture.segmenter.activation().active());
  REQUIRE(fixture.segmenter.refresh_activation(
      start->activation_id, start->turn_index, followup_end + ms(1200)));
  CHECK(fixture.segmenter.awaiting_validation_count() == 0);
  CHECK(fixture.segmenter.activation().idle_deadline_sample ==
        followup_end + ms(6900));
}

TEST_CASE("rejected in-flight follow-up expires after its old idle deadline") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto old_deadline = fixture.segmenter.activation().idle_deadline_sample;
  const auto followup_start = old_deadline - ms(100);
  const auto start = fixture.segmenter.set_vad_state(true, followup_start);
  REQUIRE(start.has_value());
  const auto followup_end = old_deadline + ms(400);
  static_cast<void>(fixture.segmenter.set_vad_state(false, followup_end));
  fixture.segmenter.add_vad_interval({{followup_start, followup_end}});
  const auto completed = fixture.segmenter.advance(followup_end + ms(900));
  REQUIRE(completed.candidates.size() == 1);
  REQUIRE(fixture.segmenter.awaiting_validation_count() == 1);

  REQUIRE(fixture.segmenter.abandon_followup(
      start->activation_id, start->turn_index, followup_end + ms(1200),
      "parse_rejected"));
  CHECK_FALSE(fixture.segmenter.activation().active());
  const auto events = fixture.segmenter.take_activation_events();
  REQUIRE(events.size() == 1);
  CHECK(events.front().kind == dvo::ActivationTransitionKind::expired);
  CHECK(events.front().reason == "idle_timeout");
}

TEST_CASE("conditional follow-up becomes eligible after causal refresh") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto old_deadline = fixture.segmenter.activation().idle_deadline_sample;

  const auto first_start_sample = old_deadline - ms(100);
  const auto first = fixture.segmenter.set_vad_state(true, first_start_sample);
  REQUIRE(first.has_value());
  const auto first_end = old_deadline + ms(100);
  static_cast<void>(fixture.segmenter.set_vad_state(false, first_end));
  fixture.segmenter.add_vad_interval({{first_start_sample, first_end}});
  REQUIRE(fixture.segmenter.advance(first_end + ms(900)).candidates.size() == 1);

  const auto second_start_sample = old_deadline + ms(1100);
  const auto second = fixture.segmenter.set_vad_state(true, second_start_sample);
  REQUIRE(second.has_value());
  const auto second_end = second_start_sample + ms(300);
  static_cast<void>(fixture.segmenter.set_vad_state(false, second_end));
  fixture.segmenter.add_vad_interval({{second_start_sample, second_end}});
  REQUIRE(fixture.segmenter.advance(second_end + ms(900)).candidates.size() == 1);
  CHECK_FALSE(fixture.segmenter.followup_plan_eligible(
      second->activation_id, second->turn_index));

  // Exact ASR is FIFO. Resolving turn 1 from its deterministic candidate end
  // expands the window far enough to causally admit turn 2, independent of
  // how quickly replay reached this point.
  REQUIRE(fixture.segmenter.refresh_activation(
      first->activation_id, first->turn_index, first_end));
  CHECK(fixture.segmenter.followup_plan_eligible(
      second->activation_id, second->turn_index));
}

TEST_CASE("conditional follow-up stays ineligible when predecessor rejects") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto old_deadline = fixture.segmenter.activation().idle_deadline_sample;
  const auto first_start_sample = old_deadline - ms(100);
  const auto first = fixture.segmenter.set_vad_state(true, first_start_sample);
  REQUIRE(first.has_value());
  const auto first_end = old_deadline + ms(100);
  static_cast<void>(fixture.segmenter.set_vad_state(false, first_end));
  fixture.segmenter.add_vad_interval({{first_start_sample, first_end}});
  REQUIRE(fixture.segmenter.advance(first_end + ms(900)).candidates.size() == 1);

  const auto second_start_sample = old_deadline + ms(1100);
  const auto second = fixture.segmenter.set_vad_state(true, second_start_sample);
  REQUIRE(second.has_value());
  const auto second_end = second_start_sample + ms(300);
  static_cast<void>(fixture.segmenter.set_vad_state(false, second_end));
  fixture.segmenter.add_vad_interval({{second_start_sample, second_end}});
  REQUIRE(fixture.segmenter.advance(second_end + ms(900)).candidates.size() == 1);

  REQUIRE(fixture.segmenter.abandon_followup(
      first->activation_id, first->turn_index, first_end, "parse_rejected"));
  CHECK_FALSE(fixture.segmenter.followup_plan_eligible(
      second->activation_id, second->turn_index));
  REQUIRE(fixture.segmenter.abandon_followup(
      second->activation_id, second->turn_index, second_end, "not_eligible"));
  CHECK_FALSE(fixture.segmenter.activation().active());
}

TEST_CASE("causal refresh must actually cover conditional follow-up onset") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto old_deadline = fixture.segmenter.activation().idle_deadline_sample;
  const auto first_start_sample = old_deadline - ms(100);
  const auto first = fixture.segmenter.set_vad_state(true, first_start_sample);
  REQUIRE(first.has_value());
  const auto first_end = old_deadline + ms(100);
  static_cast<void>(fixture.segmenter.set_vad_state(false, first_end));
  fixture.segmenter.add_vad_interval({{first_start_sample, first_end}});
  REQUIRE(fixture.segmenter.advance(first_end + ms(900)).candidates.size() == 1);

  fixture.ring.push(ms(10000), audio(16000 * 10));
  // The first turn closes 900 ms after speech and refreshes from that
  // deterministic endpoint. Start strictly beyond the resulting window.
  const auto second_start_sample = old_deadline + ms(7100);
  const auto second = fixture.segmenter.set_vad_state(true, second_start_sample);
  REQUIRE(second.has_value());
  const auto second_end = second_start_sample + ms(300);
  static_cast<void>(fixture.segmenter.set_vad_state(false, second_end));
  fixture.segmenter.add_vad_interval({{second_start_sample, second_end}});
  REQUIRE(fixture.segmenter.advance(second_end + ms(900)).candidates.size() == 1);

  REQUIRE(fixture.segmenter.refresh_activation(
      first->activation_id, first->turn_index, first_end));
  CHECK_FALSE(fixture.segmenter.followup_plan_eligible(
      second->activation_id, second->turn_index));
}

TEST_CASE("playback gate suppresses follow-up and defers only the idle window") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto idle_before = fixture.segmenter.activation().idle_deadline_sample;
  const auto hard_before = fixture.segmenter.activation().hard_deadline_sample;

  CHECK_FALSE(fixture.segmenter
                  .set_vad_state(true, ms(1500), false)
                  .has_value());
  CHECK(fixture.segmenter.pending_count() == 0);
  static_cast<void>(fixture.segmenter.set_vad_state(false, ms(1600), false));

  REQUIRE(fixture.segmenter.defer_activation_for_playback(ms(2000),
                                                           ms(1600)));
  CHECK(fixture.segmenter.activation().idle_deadline_sample ==
        idle_before + ms(2000));
  CHECK(fixture.segmenter.activation().hard_deadline_sample == hard_before);
  REQUIRE(fixture.segmenter.defer_activation_for_playback(ms(60000),
                                                           ms(1700)));
  CHECK(fixture.segmenter.activation().idle_deadline_sample == hard_before);
  CHECK(fixture.segmenter.take_activation_events().empty());
}

TEST_CASE("playback hold survives old idle and applies actual duration") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto idle_before = fixture.segmenter.activation().idle_deadline_sample;
  const auto hard_before = fixture.segmenter.activation().hard_deadline_sample;

  REQUIRE(fixture.segmenter.begin_activation_playback(idle_before - ms(100)));
  CHECK(fixture.segmenter.activation().playback_hold);
  CHECK_FALSE(fixture.segmenter.set_vad_state(true, idle_before).has_value());
  static_cast<void>(fixture.segmenter.set_vad_state(false,
                                                     idle_before + ms(100)));
  CHECK(fixture.segmenter.pending_count() == 0);
  const auto during_playback =
      fixture.segmenter.advance_for_assembly(idle_before + ms(1000));
  CHECK(during_playback.activation_events.empty());
  CHECK(fixture.segmenter.activation().active());

  REQUIRE(fixture.segmenter.finish_activation_playback(
      ms(2000), idle_before + ms(1000)));
  CHECK_FALSE(fixture.segmenter.activation().playback_hold);
  CHECK(fixture.segmenter.activation().idle_deadline_sample ==
        idle_before + ms(2000));
  CHECK(fixture.segmenter.activation().hard_deadline_sample == hard_before);
  CHECK(fixture.segmenter.activation().active());
}

TEST_CASE("a later plan refresh preserves playback idle deferral") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());

  const auto followup_start = ms(1500);
  const auto start = fixture.segmenter.set_vad_state(true, followup_start);
  REQUIRE(start.has_value());
  const auto followup_end = ms(1800);
  static_cast<void>(fixture.segmenter.set_vad_state(false, followup_end));
  fixture.segmenter.add_vad_interval({{followup_start, followup_end}});
  const auto completed = fixture.segmenter.advance(followup_end + ms(900));
  REQUIRE(completed.candidates.size() == 1);

  REQUIRE(fixture.segmenter.begin_activation_playback(ms(2800)));
  REQUIRE(fixture.segmenter.finish_activation_playback(ms(4000), ms(6800)));
  const auto deferred_idle = fixture.segmenter.activation().idle_deadline_sample;
  REQUIRE(deferred_idle == ms(11400));

  REQUIRE(fixture.segmenter.refresh_activation(
      start->activation_id, start->turn_index, ms(7000)));
  CHECK(fixture.segmenter.activation().idle_deadline_sample == deferred_idle);

  const auto next = fixture.segmenter.set_vad_state(true, ms(10000));
  REQUIRE(next.has_value());
  CHECK(next->activation_id == start->activation_id);
  CHECK(next->turn_index == start->turn_index + 1);
}

TEST_CASE("playback hold never suspends hard deadline") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto hard_deadline = fixture.segmenter.activation().hard_deadline_sample;
  REQUIRE(fixture.segmenter.begin_activation_playback(ms(1500)));

  const auto expired = fixture.segmenter.advance_for_assembly(hard_deadline);
  REQUIRE(expired.activation_events.size() == 1);
  CHECK(expired.activation_events.front().kind ==
        dvo::ActivationTransitionKind::expired);
  CHECK(expired.activation_events.front().reason == "hard_limit");
  CHECK_FALSE(fixture.segmenter.activation().active());
  CHECK_FALSE(fixture.segmenter.finish_activation_playback(ms(1000),
                                                            hard_deadline));
}

TEST_CASE("a repeated keyword replaces an in-progress follow-up activation") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  const auto followup = fixture.segmenter.set_vad_state(true, ms(1500));
  REQUIRE(followup.has_value());
  const auto old_activation = followup->activation_id;

  const auto rearmed = fixture.segmenter.add_kws_hit(
      hit({ms(1600), ms(1700)}));
  REQUIRE(rearmed.has_value());
  CHECK(rearmed->origin == dvo::UtteranceOrigin::keyword);
  CHECK(rearmed->activation_id != old_activation);
  CHECK(fixture.segmenter.pending_count() == 1);

  const auto transitions = fixture.segmenter.take_activation_events();
  REQUIRE(transitions.size() == 2);
  CHECK(transitions[0].kind == dvo::ActivationTransitionKind::cancelled);
  CHECK(transitions[0].reason == "keyword_rearm");
  CHECK(transitions[1].kind == dvo::ActivationTransitionKind::started);

  const auto result = fixture.segmenter.advance_for_assembly(ms(3000));
  REQUIRE(result.rejections.size() == 1);
  CHECK(result.rejections.front().utterance_id == followup->utterance_id);
}

TEST_CASE("reconfigure cancels activation and applies new timers") {
  ActivationFixture fixture;
  static_cast<void>(fixture.finish_keyword());
  auto updated = activation_config();
  updated.idle_timeout_ms = 5000;
  fixture.segmenter.reconfigure(activation_segmentation(), updated);

  const auto cancelled = fixture.segmenter.take_activation_events();
  REQUIRE(cancelled.size() == 1);
  CHECK(cancelled.front().kind == dvo::ActivationTransitionKind::cancelled);
  CHECK(cancelled.front().reason == "configuration_changed");
  CHECK_FALSE(fixture.segmenter.activation().active());
}
