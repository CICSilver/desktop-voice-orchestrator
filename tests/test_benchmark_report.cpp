#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "dvo/benchmark_report.h"

namespace {

nlohmann::json event(std::string type, nlohmann::json payload) {
  return {{"type", std::move(type)}, {"payload", std::move(payload)}};
}

nlohmann::json candidate(std::string id, std::uint64_t offset = 0) {
  return event("candidate",
               {{"utterance_id", std::move(id)},
                {"position", "prefix"},
                {"wake_span", {{"start", 1000 + offset},
                               {"end", 1200 + offset}}},
                {"source_spans",
                 nlohmann::json::array(
                     {{{"start", 1200 + offset}, {"end", 2000 + offset}}})}});
}

}  // namespace

TEST_CASE("benchmark report compares boundaries timing plans and dry-run actions") {
  const auto plan = event(
      "command_plan",
      {{"utterance_id", "utterance-1"},
       {"normalized_text", "增加音量"},
       {"execution_mode", "live"},
       {"actions", nlohmann::json::array({{{"sequence", 0},
                                             {"type", "volume_up"},
                                             {"volume_delta_percent", 5}}})}});
  const std::vector<nlohmann::json> recorded{
      event("kws_hit", {{"keyword", "你好"}}), candidate("old"),
      candidate("old"),
      event("asr_final", {{"latency_ms", 200.0},
                           {"inference_ms", 80.0}, {"rtf", 0.08}}),
      plan};

  auto replayed_plan = plan;
  replayed_plan["payload"]["execution_mode"] = "dry_run";
  const std::vector<nlohmann::json> replayed{
      event("kws_hit", {{"keyword", "你好"}}), candidate("new", 80),
      event("asr_final", {{"latency_ms", 120.0},
                           {"inference_ms", 50.0}, {"rtf", 0.05}}),
      replayed_plan,
      event("action_dry_run", {{"utterance_id", "new"},
                                {"action_id", "action-1"},
                                {"sequence", 0}, {"type", "volume_up"},
                                {"status", "dry_run"},
                                {"requested_volume_delta_percent", 5},
                                {"message", "benchmark dry-run"}})};

  const auto report = dvo::build_benchmark_comparison(recorded, replayed, 160);
  CHECK(report["event_sequence"]["exact"].get<bool>());
  CHECK(report["candidates"]["recorded_count"] == 1);
  CHECK(report["candidates"]["replayed_count"] == 1);
  CHECK(report["candidates"]["all_within_one_frame"].get<bool>());
  CHECK(report["candidates"]["details"][0]["max_boundary_error_samples"] == 80);
  CHECK(report["asr_timing"]["replayed"]["latency_ms"]["p95"] ==
        Catch::Approx(120.0));
  CHECK(report["asr_timing"]["replayed"]["rtf"]["p50"] ==
        Catch::Approx(0.05));
  CHECK(report["command_plans"]["action_sequence_accuracy"] ==
        Catch::Approx(1.0));
  REQUIRE(report["dry_run"]["terminal_count"] == 1);
  CHECK(report["dry_run"]["all_terminal_actions_dry_run"].get<bool>());
  CHECK(report["dry_run"]["actions"][0]["type"] == "volume_up");
}

TEST_CASE("benchmark event reader preserves valid NDJSON around malformed lines") {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("dvo-benchmark-events-" + std::to_string(suffix) + ".ndjson");
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << event("kws_hit", nlohmann::json::object()).dump() << '\n';
    output << "{not-json}\n";
    output << event("candidate_rejected", {{"reason", "wake only"}}).dump()
           << '\n';
  }

  std::vector<std::string> errors;
  const auto events = dvo::read_benchmark_events(path, &errors);
  REQUIRE(events.size() == 2);
  CHECK(events[0]["type"] == "kws_hit");
  REQUIRE(errors.size() == 1);
  CHECK(errors[0].starts_with("line 2:"));
  std::filesystem::remove(path);
}

TEST_CASE("benchmark report accepts nullable follow-up and media action fields") {
  const std::vector<nlohmann::json> replayed{
      event("candidate",
            {{"utterance_id", "followup-1"},
             {"position", nullptr},
             {"wake_span", nullptr},
             {"source_spans",
              nlohmann::json::array(
                  {{{"start", 1000}, {"end", 2000}}})}}),
      event("command_plan",
            {{"utterance_id", "followup-1"},
             {"normalized_text", "播放音乐"},
             {"execution_mode", "dry_run"},
             {"actions",
              nlohmann::json::array(
                  {{{"sequence", 1},
                    {"type", "media.play"},
                    {"volume_delta_percent", nullptr}}})}}),
      event("action_dry_run",
            {{"utterance_id", "followup-1"},
             {"action_id", "action-1"},
             {"sequence", 1},
             {"type", "media.play"},
             {"status", "dry_run"},
             {"requested_volume_delta_percent", nullptr},
             {"message", "benchmark dry-run"}})};

  const auto report = dvo::build_benchmark_comparison({}, replayed, 160);
  CHECK(report["candidates"]["replayed_count"] == 1);
  CHECK(report["command_plans"]["replayed"][0]["actions"][0]
              ["volume_delta_percent"]
                  .is_null());
  CHECK(report["dry_run"]["actions"][0]["requested_volume_delta_percent"]
            .is_null());

  // Pairing a recorded follow-up candidate with a replayed one compares their
  // null wake positions instead of throwing.
  const auto paired = dvo::build_benchmark_comparison(replayed, replayed, 160);
  CHECK(paired["candidates"]["paired_count"] == 1);
  CHECK(paired["candidates"]["details"][0]["position_equal"] == true);
}

TEST_CASE("replayed utterance summary folds hits, candidates, ASR and outcomes") {
  constexpr std::uint64_t origin = 160000;  // replay clock starts 10 s in
  auto hit = event("kws_hit", {{"keyword", "小助手"},
                               {"detector", "microphone_fallback"},
                               {"detected_at_sample", origin + 33120},
                               {"wake_span", {{"start", origin + 23840},
                                              {"end", origin + 32480}}}});
  auto suppressed_hit = event("kws_hit", {{"keyword", "小助手"},
                                          {"detector", "aec"},
                                          {"detected_at_sample", origin + 80000},
                                          {"wake_span", {{"start", origin + 70000},
                                                         {"end", origin + 79000}}}});
  auto suppression = event("kws_suppressed", {{"reason", "announcement_playback_guard"},
                                              {"detector", "aec"}});
  suppression["timestamp_sample"] = origin + 80000;
  const std::vector<nlohmann::json> events{
      hit,
      event("recognition_audio_fallback", {{"activation_id", "act-1"}}),
      event("candidate", {{"utterance_id", "u1"},
                          {"origin", "keyword"},
                          {"activation_id", "act-1"},
                          {"position", "prefix"},
                          {"boundary_source", "vad"},
                          {"source_spans", nlohmann::json::array(
                                               {{{"start", origin + 16000},
                                                 {"end", origin + 64000}}})}}),
      event("asr_final", {{"utterance_id", "u1"}, {"text", "小助手暂停音"},
                          {"latency_ms", 104.5}}),
      event("parse_debug", {{"utterance_id", "u1"}, {"matched", true},
                            {"error", nullptr}}),
      event("command_plan", {{"utterance_id", "u1"},
                             {"normalized_text", "暂停音乐"},
                             {"actions", nlohmann::json::array(
                                             {{{"sequence", 1},
                                               {"type", "media.pause"},
                                               {"volume_delta_percent", nullptr}}})}}),
      suppressed_hit,
      suppression,
      event("candidate", {{"utterance_id", "u2"},
                          {"origin", "followup"},
                          {"activation_id", "act-2"},
                          {"position", nullptr},
                          {"source_spans", nlohmann::json::array(
                                               {{{"start", origin + 96000},
                                                 {"end", origin + 128000}}})}}),
      event("asr_final", {{"utterance_id", "u2"}, {"text", "暂停音乐老板"}}),
      event("parse_debug", {{"utterance_id", "u2"}, {"stage", "parser"},
                            {"error", {{"code", "unknown_text"},
                                       {"message", "unknown"}}}}),
      event("command_rejected", {{"utterance_id", "u2"},
                                 {"reason", "unknown_text: unknown"}}),
      event("candidate_rejected", {{"utterance_id", "u3"},
                                   {"origin", "followup"},
                                   {"reason", "superseded by keyword rearm"}})};

  const auto summary = dvo::summarize_replayed_utterances(events, origin);
  const auto& hits = summary["keyword_hits"];
  REQUIRE(hits.size() == 2);
  CHECK(hits[0]["detector"] == "microphone_fallback");
  CHECK(hits[0]["wake_end_s"].get<double>() == Catch::Approx(2.03));
  CHECK(hits[0]["suppressed"].is_null());
  CHECK(hits[1]["suppressed"] == "announcement_playback_guard");

  const auto& utterances = summary["utterances"];
  REQUIRE(utterances.size() == 3);
  CHECK(utterances[0]["utterance_id"] == "u1");
  CHECK(utterances[0]["audio_source"] == "microphone");
  CHECK(utterances[0]["candidate"]["start_s"].get<double>() == Catch::Approx(1.0));
  CHECK(utterances[0]["candidate"]["end_s"].get<double>() == Catch::Approx(4.0));
  CHECK(utterances[0]["outcome"] == "plan");
  CHECK(utterances[0]["asr_text"] == "小助手暂停音");
  CHECK(utterances[0]["actions"][0]["type"] == "media.pause");
  CHECK(utterances[1]["outcome"] == "rejected");
  CHECK(utterances[1]["audio_source"] == "aec");
  CHECK(utterances[1]["parse_error"]["code"] == "unknown_text");
  // Utterances without a candidate sort last.
  CHECK(utterances[2]["outcome"] == "candidate_rejected");
  CHECK(utterances[2]["candidate"].is_null());
}
