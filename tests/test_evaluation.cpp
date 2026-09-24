#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>

#include "dvo/evaluation.h"

namespace {

nlohmann::json take(std::string id, std::string kind, std::string text,
                    std::string position, nlohmann::json actions, double start_s,
                    double end_s, std::string status = "ok") {
  return {{"id", std::move(id)},       {"kind", std::move(kind)},
          {"status", std::move(status)}, {"prompt", text},
          {"text", text},              {"wake_position", std::move(position)},
          {"expected_actions", std::move(actions)},
          {"start_ms", start_s * 1000.0}, {"end_ms", end_s * 1000.0}};
}

nlohmann::json action(std::string type) { return {{"type", std::move(type)}}; }

nlohmann::json labels_document() {
  return {{"schema", "dvo-eval-labels"},
          {"schema_version", 1},
          {"collection_id", "c1"},
          {"group", {{"id", "quiet-near"}, {"title", "安静 · 近距离"},
                     {"condition", "quiet"}, {"distance_m", 0.5}}},
          {"speaker", "A"},
          {"wake_word", "小助手"},
          {"complete", true},
          {"takes",
           nlohmann::json::array(
               {take("t1", "command", "小助手，暂停音乐", "prefix",
                     nlohmann::json::array({action("media.pause")}), 1.0, 5.0),
                take("t2", "command", "播放音乐，小助手", "suffix",
                     nlohmann::json::array({action("media.play")}), 6.0, 10.0),
                take("t3", "negative", "今天天气怎么样", "none",
                     nlohmann::json::array(), 12.0, 16.0),
                take("t4", "command", "小助手，降低音量百分之十", "prefix",
                     nlohmann::json::array({{{"type", "audio.volume.adjust"},
                                             {"volume_delta_percent", -10}}}),
                     17.0, 21.0),
                take("t5", "command", "小助手，播放音乐", "prefix",
                     nlohmann::json::array({action("media.play")}), 22.0, 24.0,
                     "discarded")})}};
}

nlohmann::json utterance(std::string id, double start, double end, std::string text,
                         std::string outcome, nlohmann::json actions) {
  return {{"utterance_id", std::move(id)},
          {"origin", "keyword"},
          {"position", nullptr},
          {"activation_id", "a"},
          {"audio_source", "aec"},
          {"candidate", {{"start_s", start}, {"end_s", end}}},
          {"asr_text", std::move(text)},
          {"outcome", std::move(outcome)},
          {"normalized_text", nullptr},
          {"actions", std::move(actions)},
          {"rejection", nullptr}};
}

nlohmann::json hit(double wake_end_s) {
  return {{"keyword", "小助手"}, {"detector", "aec"}, {"wake_start_s", wake_end_s - 0.5},
          {"wake_end_s", wake_end_s}, {"detected_s", wake_end_s + 0.04},
          {"suppressed", nullptr}};
}

}  // namespace

TEST_CASE("text coverage reports characters missing at either edge") {
  const auto tail = dvo::text_coverage("小助手，暂停音乐", "小助手暂停音");
  CHECK(tail.expected_chars == 7);
  CHECK(tail.matched_chars == 6);
  CHECK(tail.head_missing == 0);
  CHECK(tail.tail_missing == 1);
  CHECK(tail.missing_tail == "乐");

  const auto head = dvo::text_coverage("播放音乐，小助手", "放音乐小助手");
  CHECK(head.head_missing == 1);
  CHECK(head.missing_head == "播");
  CHECK(head.tail_missing == 0);

  const auto exact = dvo::text_coverage("小助手，暂停音乐。", "小助手 暂停音乐");
  CHECK(exact.matched_chars == exact.expected_chars);
  CHECK(exact.head_missing == 0);
  CHECK(exact.tail_missing == 0);

  const auto inner = dvo::text_coverage("暂停音乐", "暂音乐");
  CHECK(inner.matched_chars == 3);
  CHECK(inner.head_missing == 0);
  CHECK(inner.tail_missing == 0);

  const auto empty = dvo::text_coverage("暂停音乐", "");
  CHECK(empty.matched_chars == 0);
  CHECK(empty.tail_missing == 4);
}

TEST_CASE("evaluation labels are validated") {
  const auto labels = dvo::parse_evaluation_labels(labels_document());
  REQUIRE(labels.takes.size() == 5);
  CHECK(labels.condition == "quiet");
  CHECK(labels.distance_m == Catch::Approx(0.5));
  CHECK(labels.takes[0].start_s == Catch::Approx(1.0));
  CHECK(labels.takes[3].expected_actions[0].volume_delta_percent == -10);

  auto bad_kind = labels_document();
  bad_kind["takes"][0]["kind"] = "chat";
  CHECK_THROWS_AS(dvo::parse_evaluation_labels(bad_kind), std::invalid_argument);

  auto no_actions = labels_document();
  no_actions["takes"][0]["expected_actions"] = nlohmann::json::array();
  CHECK_THROWS_AS(dvo::parse_evaluation_labels(no_actions), std::invalid_argument);

  auto no_wake_position = labels_document();
  no_wake_position["takes"][0]["wake_position"] = "none";
  CHECK_THROWS_AS(dvo::parse_evaluation_labels(no_wake_position), std::invalid_argument);

  auto wrong_schema = labels_document();
  wrong_schema["schema"] = "other";
  CHECK_THROWS_AS(dvo::parse_evaluation_labels(wrong_schema), std::invalid_argument);
}

TEST_CASE("labeled session scoring attributes wake, command and truncation failures") {
  const auto labels = dvo::parse_evaluation_labels(labels_document());
  const nlohmann::json summary{
      {"keyword_hits", nlohmann::json::array({hit(1.8), hit(9.5), hit(13.0), hit(23.0)})},
      {"utterances",
       nlohmann::json::array(
           {utterance("u1", 1.2, 4.0, "小助手暂停音", "plan",
                      nlohmann::json::array({{{"type", "media.pause"},
                                              {"volume_delta_percent", nullptr}}})),
            utterance("u2", 6.6, 9.9, "放音乐小助手", "plan",
                      nlohmann::json::array({{{"type", "media.play"},
                                              {"volume_delta_percent", nullptr}}})),
            utterance("u5", 22.2, 23.5, "小助手播放音乐", "plan",
                      nlohmann::json::array({{{"type", "media.play"},
                                              {"volume_delta_percent", nullptr}}}))})}};

  dvo::SessionReference reference;
  reference.takes.resize(labels.takes.size());
  for (auto& take : reference.takes) take.available = true;
  reference.takes[0].kws_microphone = true;
  reference.takes[0].asr_processed = "小助手暂停音乐";
  reference.takes[1].kws_microphone = true;
  reference.takes[3].kws_microphone = true;  // heard offline, missed live
  // Extending u1's end recovers "乐"; extending u2's start recovers "播".
  reference.candidates["u1"] = {std::nullopt, std::nullopt, "小助手暂停音",
                                "小助手暂停音", "小助手暂停音乐"};
  reference.candidates["u2"] = {-4.0, -40.0, "放音乐小助手", "播放音乐小助手",
                                "放音乐小助手"};

  const auto report = dvo::evaluate_labeled_session(labels, summary, reference);
  const auto& takes = report["takes"];
  REQUIRE(takes.size() == 5);

  CHECK(takes[0]["verdict"] == "correct");
  CHECK(takes[0]["wake"]["verdict"] == "detected");
  CHECK(takes[0]["truncation"]["verdict"] == "tail_cut");
  CHECK(takes[0]["truncation"]["pipeline"]["missing_tail"] == "乐");

  CHECK(takes[1]["verdict"] == "correct");
  CHECK(takes[1]["truncation"]["verdict"] == "head_cut");

  CHECK(takes[2]["kind"] == "negative");
  CHECK(takes[2]["verdict"] == "false_wake");

  CHECK(takes[3]["verdict"] == "no_wake");
  CHECK(takes[3]["wake"]["verdict"] == "missed_pipeline");

  // Discarded retakes keep their window but are not scored.
  CHECK(takes[4]["status"] == "discarded");
  CHECK_FALSE(takes[4].contains("verdict"));

  const auto& commands = report["summary"]["commands"];
  CHECK(commands["total"] == 3);
  CHECK(commands["wake_detected"] == 2);
  CHECK(commands["wake_missed_pipeline"] == 1);
  CHECK(commands["correct"] == 2);
  CHECK(commands["no_wake"] == 1);
  CHECK(commands["tail_cut"] == 1);
  CHECK(commands["head_cut"] == 1);
  CHECK(report["summary"]["by_position"]["prefix"]["total"] == 2);
  CHECK(report["summary"]["by_position"]["prefix"]["wake_detected"] == 1);
  CHECK(report["summary"]["by_position"]["suffix"]["correct"] == 1);
  CHECK(report["summary"]["negatives"]["false_wakes"] == 1);
  CHECK(report["summary"]["unassigned_keyword_hits"] == 0);
}

TEST_CASE("evaluation reports aggregate by condition and format a summary") {
  const auto labels = dvo::parse_evaluation_labels(labels_document());
  dvo::SessionReference reference;
  reference.takes.resize(labels.takes.size());
  const nlohmann::json empty{{"keyword_hits", nlohmann::json::array()},
                             {"utterances", nlohmann::json::array()}};
  auto quiet = dvo::evaluate_labeled_session(labels, empty, reference);
  auto music_document = labels_document();
  music_document["group"]["condition"] = "music";
  music_document["group"]["distance_m"] = 2.0;
  auto music = dvo::evaluate_labeled_session(
      dvo::parse_evaluation_labels(music_document), empty, reference);

  const std::vector<nlohmann::json> reports{quiet, music};
  const auto aggregate = dvo::aggregate_evaluations(reports);
  REQUIRE(aggregate["groups"].size() == 2);
  CHECK(aggregate["overall"]["sessions"] == 2);
  CHECK(aggregate["overall"]["commands"]["total"] == 6);
  CHECK(aggregate["overall"]["commands"]["wake_missed_model"] == 6);

  const auto text = dvo::format_evaluation_summary(reports, aggregate);
  CHECK(text.find("评估汇总") != std::string::npos);
  CHECK(text.find("播放音乐 · 2.0 米") != std::string::npos);
  CHECK(text.find("t4") != std::string::npos);
}

TEST_CASE("missing characters that extension does not recover are model drops") {
  auto document = labels_document();
  document["takes"] = nlohmann::json::array(
      {take("t1", "command", "小助手，暂停音乐", "prefix",
            nlohmann::json::array({action("media.pause")}), 1.0, 5.0)});
  const auto labels = dvo::parse_evaluation_labels(document);
  const nlohmann::json summary{
      {"keyword_hits", nlohmann::json::array({hit(1.8)})},
      {"utterances",
       nlohmann::json::array({utterance("u1", 1.2, 4.0, "小助手暂停音", "plan",
                                        nlohmann::json::array({{{"type", "media.pause"},
                                                                {"volume_delta_percent",
                                                                 nullptr}}}))})}};
  dvo::SessionReference reference;
  reference.takes.resize(1);
  reference.takes[0].available = true;
  // The whole-window decode hears "乐" but the candidate re-decodes do not:
  // that is recognizer instability, not a segmentation cut.
  reference.takes[0].asr_processed = "小助手暂停音乐";
  reference.candidates["u1"] = {-30.0, -30.0, "小助手暂停音", "小助手暂停音",
                                "小助手暂停音"};

  const auto report = dvo::evaluate_labeled_session(labels, summary, reference);
  CHECK(report["takes"][0]["truncation"]["verdict"] == "asr_drop");
  CHECK(report["summary"]["commands"]["asr_drop"] == 1);
  CHECK(report["summary"]["commands"]["tail_cut"] == 0);
  CHECK(report["takes"][0]["utterances"][0]["reference"]["text_tail_extended"] ==
        "小助手暂停音");
}
