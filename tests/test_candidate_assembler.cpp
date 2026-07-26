#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>
#include <vector>

#include "dvo/candidate_assembler.h"

namespace {

dvo::CandidateAssemblyRequest request(std::string id,
                                      std::vector<dvo::SampleSpan> spans) {
  dvo::CandidateAssemblyRequest value;
  value.candidate.utterance_id = std::move(id);
  value.candidate.wake_span = {400, 600};
  value.candidate.source_spans = std::move(spans);
  value.join_silence_samples = 3;
  return value;
}

bool wait_for(dvo::CandidateAssembler& assembler,
              dvo::AudioAssemblyResult& result) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (assembler.try_pop(result)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

TEST_CASE("candidate assembly slices outside the segmenter and inserts join silence") {
  dvo::TimedRingBuffer ring(2000);
  std::vector<float> audio(1000);
  for (std::size_t i = 0; i < audio.size(); ++i) {
    audio[i] = static_cast<float>(i);
  }
  ring.push(0, audio);

  auto result = dvo::assemble_candidate(
      ring, request("embedded", {{100, 400}, {600, 900}}));
  REQUIRE(result.candidate.has_value());
  CHECK(result.candidate->source_spans ==
        std::vector<dvo::SampleSpan>{{100, 400}, {600, 900}});
  REQUIRE(result.candidate->pcm.size() == 603);
  CHECK(result.candidate->pcm[299] == 399.0F);
  CHECK(result.candidate->pcm[300] == 0.0F);
  CHECK(result.candidate->pcm[301] == 0.0F);
  CHECK(result.candidate->pcm[302] == 0.0F);
  CHECK(result.candidate->pcm[303] == 600.0F);
}

TEST_CASE("candidate assembly permits continuous source audio across the wake span") {
  dvo::TimedRingBuffer ring(2000);
  ring.push(0, std::vector<float>(1000, 0.25F));

  auto result = dvo::assemble_candidate(
      ring, request("continuous", {{200, 800}}));
  REQUIRE(result.candidate.has_value());
  CHECK(result.rejection.empty());
  CHECK(result.candidate->source_spans ==
        std::vector<dvo::SampleSpan>{{200, 800}});
  CHECK(result.candidate->pcm.size() == 600);
}

TEST_CASE("candidate assembler bounds all outstanding work until completion is consumed") {
  dvo::TimedRingBuffer ring(2000);
  ring.push(0, std::vector<float>(1000, 0.25F));
  dvo::CandidateAssembler assembler(ring, 1);

  REQUIRE(assembler.try_submit(request("first", {{600, 900}})));
  CHECK_FALSE(assembler.try_submit(request("overflow", {{600, 900}})));
  CHECK(assembler.outstanding() == 1);

  dvo::AudioAssemblyResult raw_first;
  REQUIRE(wait_for(assembler, raw_first));
  REQUIRE(std::holds_alternative<dvo::CandidateAssemblyResult>(raw_first));
  auto first = std::get<dvo::CandidateAssemblyResult>(std::move(raw_first));
  REQUIRE(first.candidate.has_value());
  CHECK(first.utterance_id == "first");
  CHECK(assembler.outstanding() == 0);

  REQUIRE(assembler.try_submit(request("second", {{600, 900}})));
  dvo::AudioAssemblyResult raw_second;
  REQUIRE(wait_for(assembler, raw_second));
  REQUIRE(std::holds_alternative<dvo::CandidateAssemblyResult>(raw_second));
  auto second = std::get<dvo::CandidateAssemblyResult>(std::move(raw_second));
  REQUIRE(second.candidate.has_value());
  CHECK(second.utterance_id == "second");
}

TEST_CASE("backfill assembly runs on the bounded worker and preserves timing metadata") {
  dvo::TimedRingBuffer ring(2000);
  std::vector<float> audio(1000);
  for (std::size_t i = 0; i < audio.size(); ++i) {
    audio[i] = static_cast<float>(i);
  }
  ring.push(0, audio);

  dvo::CandidateAssembler assembler(ring, 2);
  dvo::BackfillAssemblyRequest request;
  request.utterance_id = "backfill";
  request.origin = dvo::UtteranceOrigin::followup;
  request.activation_id = "activation-backfill";
  request.turn_index = 3;
  request.trigger_sample = 321;
  request.source_spans = {{100, 200}, {300, 400}};
  request.join_silence_samples = 2;
  request.trailing_silence_samples = 3;
  request.stream_start_sample = 456;
  request.recognition_generation = 7;
  REQUIRE(assembler.try_submit(std::move(request)));

  dvo::AudioAssemblyResult raw;
  REQUIRE(wait_for(assembler, raw));
  REQUIRE(std::holds_alternative<dvo::BackfillAssemblyResult>(raw));
  auto result = std::get<dvo::BackfillAssemblyResult>(std::move(raw));
  CHECK(result.utterance_id == "backfill");
  CHECK(result.origin == dvo::UtteranceOrigin::followup);
  CHECK(result.activation_id == "activation-backfill");
  CHECK(result.turn_index == 3);
  CHECK(result.trigger_sample == 321);
  CHECK(result.first_sample == 100);
  CHECK(result.stream_start_sample == 456);
  CHECK(result.recognition_generation == 7);
  REQUIRE(result.pcm.size() == 205);
  CHECK(result.pcm[99] == 199.0F);
  CHECK(result.pcm[100] == 0.0F);
  CHECK(result.pcm[101] == 0.0F);
  CHECK(result.pcm[102] == 300.0F);
  CHECK(result.pcm.back() == 0.0F);
}

TEST_CASE("candidate assembly rejects audio overwritten by the ring") {
  dvo::TimedRingBuffer ring(100);
  ring.push(1000, std::vector<float>(100, 0.5F));
  auto result = dvo::assemble_candidate(
      ring, request("expired", {{600, 900}}));
  CHECK_FALSE(result.candidate.has_value());
  CHECK(result.utterance_id == "expired");
  CHECK(result.rejection ==
        "candidate audio is no longer available in the ring buffer");
}
