#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "dvo/command_parser.h"

namespace {

dvo::CommandParseContext context(std::string id = "utterance-1") {
  dvo::CommandParseContext value;
  value.runtime_session_id = "parser-test-runtime";
  value.utterance_id = std::move(id);
  value.origin = dvo::UtteranceOrigin::followup;
  value.activation_id = "activation-1";
  value.turn_index = 2;
  value.trigger_sample = 15000;
  value.wake_position = "prefix";
  value.config_revision = 11;
  value.recognition_generation = 7;
  value.final_revision = 3;
  value.timestamp_sample = 16000;
  return value;
}

std::string chinese_number(int value) {
  static constexpr std::array<const char*, 10> digit{
      "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
  if (value < 10) return digit[static_cast<std::size_t>(value)];
  if (value == 10) return "十";
  if (value < 20) return std::string("十") + digit[static_cast<std::size_t>(value - 10)];
  return "二十";
}

}  // namespace

TEST_CASE("Chinese command parser maps the four default rules") {
  const dvo::CommandParser parser;

  const auto play = parser.parse("播放音乐", context("play"));
  REQUIRE(play.ok());
  REQUIRE(play.plan->actions.size() == 1);
  CHECK(play.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK_FALSE(play.plan->actions[0].volume_delta_percent.has_value());

  const auto open = parser.parse("打开音乐", context("open"));
  REQUIRE(open.ok());
  CHECK(open.plan->actions[0].type == dvo::ActionType::media_play);

  const auto pause = parser.parse("暂停音乐", context("pause"));
  REQUIRE(pause.ok());
  CHECK(pause.plan->actions[0].type == dvo::ActionType::media_pause);

  const auto increase = parser.parse("增加音量", context("increase"));
  REQUIRE(increase.ok());
  CHECK(increase.plan->actions[0].type == dvo::ActionType::master_volume_adjust);
  CHECK(increase.plan->actions[0].volume_delta_percent == 5);
  CHECK(increase.plan->normalized_text == "增加音量5%");

  const auto decrease = parser.parse("降低音量", context("decrease"));
  REQUIRE(decrease.ok());
  CHECK(decrease.plan->actions[0].volume_delta_percent == -5);
  CHECK(decrease.plan->normalized_text == "降低音量5%");
}

TEST_CASE("Volume amounts accept deterministic Chinese and Arabic forms from 1 to 20") {
  const dvo::CommandParser parser;
  for (int value = 1; value <= 20; ++value) {
    CAPTURE(value);
    const auto chinese = parser.parse("增加音量百分之" + chinese_number(value),
                                      context("zh-" + std::to_string(value)));
    REQUIRE(chinese.ok());
    CHECK(chinese.plan->actions[0].volume_delta_percent == value);

    const auto arabic = parser.parse("降低音量" + std::to_string(value) + "%",
                                     context("ar-" + std::to_string(value)));
    REQUIRE(arabic.ok());
    CHECK(arabic.plan->actions[0].volume_delta_percent == -value);
  }

  const auto liang = parser.parse("增加音量百分之两", context("liang"));
  REQUIRE(liang.ok());
  CHECK(liang.plan->actions[0].volume_delta_percent == 2);

  const auto chinese_suffix = parser.parse("降低音量二十%", context("zh-suffix"));
  REQUIRE(chinese_suffix.ok());
  CHECK(chinese_suffix.plan->actions[0].volume_delta_percent == -20);
}

TEST_CASE("Normalization handles whitespace punctuation and full-width percentages") {
  const dvo::CommandParser parser;
  const auto result = parser.parse("  播放音乐；然后，增加音量５％。  ", context());
  REQUIRE(result.ok());
  REQUIRE(result.plan->actions.size() == 2);
  CHECK(result.plan->normalized_text == "播放音乐;增加音量5%");
  CHECK(result.plan->actions[1].volume_delta_percent == 5);
}

TEST_CASE("Parser rejects leading and consecutive separators as empty clauses") {
  const dvo::CommandParser parser;
  for (const auto* text : {"，播放音乐", ",播放音乐", "播放音乐，，暂停音乐",
                           "播放音乐,;暂停音乐", "播放音乐；然后，，暂停音乐",
                           "播放音乐。。"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, context("empty-clause"));
    REQUIRE_FALSE(result.ok());
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == "empty_clause");
  }

  CHECK(parser.parse("播放音乐。", context("single-terminal")).ok());
}

TEST_CASE("Commands can be adjacent or separated by explicit connectors") {
  const dvo::CommandParser parser;
  const auto result = parser.parse(
      "播放音乐暂停音乐，然后增加音量百分之两；接着降低音量5%，并且播放音乐",
      context());
  REQUIRE(result.ok());
  REQUIRE(result.plan->actions.size() == 5);
  CHECK(result.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(result.plan->actions[1].type == dvo::ActionType::media_pause);
  CHECK(result.plan->actions[2].volume_delta_percent == 2);
  CHECK(result.plan->actions[3].volume_delta_percent == -5);
  CHECK(result.plan->actions[4].type == dvo::ActionType::media_play);
  for (std::size_t i = 0; i < result.plan->actions.size(); ++i) {
    CHECK(result.plan->actions[i].sequence == i + 1);
  }
}

TEST_CASE("Wake text, plus separators and one-character ASR omissions remain executable") {
  const dvo::CommandParser parser;
  auto wake_context = context("continuous-wake");
  wake_context.origin = dvo::UtteranceOrigin::keyword;
  wake_context.wake_word = "@小助手";

  const auto continuous = parser.parse("小助手播放音乐", wake_context);
  REQUIRE(continuous.ok());
  REQUIRE(continuous.plan->actions.size() == 1);
  CHECK(continuous.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(continuous.plan->raw_text == "小助手播放音乐");
  CHECK(continuous.plan->normalized_text == "播放音乐");

  auto followup_wake_context = wake_context;
  followup_wake_context.origin = dvo::UtteranceOrigin::followup;
  const auto repeated_wake =
      parser.parse("小助手播放音乐", followup_wake_context);
  REQUIRE(repeated_wake.ok());
  REQUIRE(repeated_wake.plan->actions.size() == 1);
  CHECK(repeated_wake.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(repeated_wake.plan->normalized_text == "播放音乐");

  const auto tolerant = parser.parse("小助手，播放音+暂停音", wake_context);
  REQUIRE(tolerant.ok());
  REQUIRE(tolerant.plan->actions.size() == 2);
  CHECK(tolerant.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(tolerant.plan->actions[1].type == dvo::ActionType::media_pause);
  CHECK(tolerant.plan->normalized_text == "播放音乐;暂停音乐");

  const auto salvaged = parser.parse("小助手然后后再暂停音", wake_context);
  REQUIRE(salvaged.ok());
  REQUIRE(salvaged.plan->actions.size() == 1);
  CHECK(salvaged.plan->actions[0].type == dvo::ActionType::media_pause);

  const auto truncated_second =
      parser.parse("小助手播放音乐然后再暂", wake_context);
  REQUIRE(truncated_second.ok());
  REQUIRE(truncated_second.plan->actions.size() == 2);
  CHECK(truncated_second.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(truncated_second.plan->actions[1].type == dvo::ActionType::media_pause);
  CHECK(truncated_second.plan->normalized_text == "播放音乐;暂停音乐");

  const auto polite_suffix_wake =
      parser.parse("帮我打开音乐小助手", wake_context);
  REQUIRE(polite_suffix_wake.ok());
  REQUIRE(polite_suffix_wake.plan->actions.size() == 1);
  CHECK(polite_suffix_wake.plan->actions[0].type ==
        dvo::ActionType::media_play);
  CHECK(polite_suffix_wake.plan->normalized_text == "打开音乐");

  const auto polite_prefix_wake =
      parser.parse("小助手，请帮我播放音乐", wake_context);
  REQUIRE(polite_prefix_wake.ok());
  CHECK(polite_prefix_wake.plan->actions[0].type ==
        dvo::ActionType::media_play);

  const auto pause_confusion =
      parser.parse("再连音乐小助手两", wake_context);
  REQUIRE(pause_confusion.ok());
  CHECK(pause_confusion.plan->actions[0].type ==
        dvo::ActionType::media_pause);
  CHECK(pause_confusion.plan->normalized_text == "暂停音乐");

  const auto alternate_pause_confusion =
      parser.parse("再听音乐小助手", wake_context);
  REQUIRE(alternate_pause_confusion.ok());
  CHECK(alternate_pause_confusion.plan->actions[0].type ==
        dvo::ActionType::media_pause);

  CHECK_FALSE(parser.parse("再连音乐", wake_context).ok());
  CHECK_FALSE(parser.parse("再连音乐小助手两次", wake_context).ok());
}

TEST_CASE("Natural multi-command connectors are accepted") {
  const dvo::CommandParser parser;
  for (const auto* text : {"播放音乐然后再暂停音乐", "播放音乐以及暂停音乐",
                           "播放音乐和暂停音乐", "播放音乐还有暂停音乐"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, context("natural-connectors"));
    REQUIRE(result.ok());
    REQUIRE(result.plan->actions.size() == 2);
  }
}

TEST_CASE("Parser rejects the whole sentence when any residual text is unknown") {
  const dvo::CommandParser parser;
  for (const auto* text : {"帮我未知命令", "播放音乐吧", "播放音乐然后未知命令",
                           "播放音乐暂停音乐谢谢", "播放，音乐", "增加音量五",
                           "播放音乐并暂停音乐"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, context());
    CHECK_FALSE(result.ok());
    CHECK_FALSE(result.plan.has_value());
    REQUIRE(result.error.has_value());
  }
}

TEST_CASE("Parser rejects invalid ambiguous and excessive volume amounts") {
  const dvo::CommandParser parser;
  for (const auto* text : {"增加音量0%", "增加音量百分之零", "增加音量21%",
                           "降低音量百分之二十一", "增加音量05%", "增加音量百分之",
                           "增加音量百分之5%", "降低音量到5%"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, context());
    CHECK_FALSE(result.ok());
    CHECK_FALSE(result.plan.has_value());
  }
}

TEST_CASE("Parser limits a plan to eight actions") {
  const dvo::CommandParser parser;
  const auto eight = parser.parse(
      "播放音乐暂停音乐播放音乐暂停音乐播放音乐暂停音乐播放音乐暂停音乐", context("eight"));
  REQUIRE(eight.ok());
  CHECK(eight.plan->actions.size() == 8);

  const auto nine = parser.parse(
      "播放音乐暂停音乐播放音乐暂停音乐播放音乐暂停音乐播放音乐暂停音乐播放音乐",
      context("nine"));
  CHECK_FALSE(nine.ok());
  REQUIRE(nine.error.has_value());
  CHECK(nine.error->code == "too_many_actions");
}

TEST_CASE("Plan IDs are deterministic per utterance and non-live sources are dry-run") {
  const dvo::CommandParser parser;
  auto replay_context = context("stable-id");
  replay_context.source = dvo::ExecutionSource::replay;
  replay_context.execution_mode = dvo::ExecutionMode::live;

  const auto first = parser.parse("播放音乐", replay_context);
  const auto second = parser.parse("播放音乐。", replay_context);
  REQUIRE(first.ok());
  REQUIRE(second.ok());
  CHECK(first.plan->command_id == second.plan->command_id);
  CHECK(first.plan->actions[0].action_id == second.plan->actions[0].action_id);
  CHECK(first.plan->execution_mode == dvo::ExecutionMode::dry_run);
  CHECK(first.plan->config_revision == replay_context.config_revision);
  CHECK(first.plan->recognition_generation == replay_context.recognition_generation);
  CHECK(first.plan->final_revision == replay_context.final_revision);
  CHECK(first.plan->schema_version == 2);
  CHECK(first.plan->origin == dvo::UtteranceOrigin::followup);
  CHECK(first.plan->activation_id == "activation-1");
  CHECK(first.plan->turn_index == 2);
  CHECK(first.plan->trigger_sample == 15000);
}

TEST_CASE("Replay parse debugging strips the wake word and preserves action order") {
  const dvo::CommandParser parser;
  auto replay_context = context("replay-parse-debug");
  replay_context.origin = dvo::UtteranceOrigin::keyword;
  replay_context.wake_word = "小助手";
  replay_context.source = dvo::ExecutionSource::replay;
  replay_context.execution_mode = dvo::ExecutionMode::dry_run;

  const auto parsed = parser.parse(
      "小助手，暂停音乐，然后增加音量百分之十，再播放音乐",
      replay_context);

  REQUIRE(parsed.ok());
  REQUIRE(parsed.plan->actions.size() == 3);
  CHECK(parsed.plan->normalized_text ==
        "暂停音乐;增加音量10%;播放音乐");
  CHECK(parsed.plan->actions[0].type == dvo::ActionType::media_pause);
  CHECK(parsed.plan->actions[1].type ==
        dvo::ActionType::master_volume_adjust);
  CHECK(parsed.plan->actions[1].volume_delta_percent == 10);
  CHECK(parsed.plan->actions[2].type == dvo::ActionType::media_play);
  CHECK(parsed.plan->execution_mode == dvo::ExecutionMode::dry_run);
}

TEST_CASE("Plan idempotency keys include runtime generation and final revision boundaries") {
  const dvo::CommandParser parser;
  auto first_context = context("same-utterance");
  const auto first = parser.parse("播放音乐", first_context);
  const auto duplicate = parser.parse("播放音乐", first_context);
  REQUIRE(first.ok());
  REQUIRE(duplicate.ok());
  CHECK(first.plan->command_id == duplicate.plan->command_id);
  CHECK(first.plan->actions[0].action_id == duplicate.plan->actions[0].action_id);

  auto different_generation = first_context;
  ++different_generation.recognition_generation;
  const auto generation_plan = parser.parse("播放音乐", different_generation);
  REQUIRE(generation_plan.ok());
  CHECK(generation_plan.plan->command_id != first.plan->command_id);

  auto different_final = first_context;
  ++different_final.final_revision;
  const auto final_plan = parser.parse("播放音乐", different_final);
  REQUIRE(final_plan.ok());
  CHECK(final_plan.plan->command_id != first.plan->command_id);

  auto different_runtime = first_context;
  different_runtime.runtime_session_id = "another-runtime";
  const auto runtime_plan = parser.parse("播放音乐", different_runtime);
  REQUIRE(runtime_plan.ok());
  CHECK(runtime_plan.plan->command_id != first.plan->command_id);
}

TEST_CASE("Configured command grammar and action limit replace the defaults") {
  dvo::CommandGrammar grammar;
  grammar.play_phrases = {"开始播放", "继续放歌"};
  grammar.pause_phrases = {"停止播放"};
  grammar.volume_up_phrases = {"音量加"};
  grammar.volume_down_phrases = {"音量减"};
  grammar.connectors = {"接下来"};
  grammar.max_actions_per_utterance = 2;
  const dvo::CommandParser parser(7, 20, std::move(grammar));

  const auto parsed = parser.parse("开始播放接下来音量加百分之二", context("custom"));
  REQUIRE(parsed.ok());
  REQUIRE(parsed.plan->actions.size() == 2);
  CHECK(parsed.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(parsed.plan->actions[1].volume_delta_percent == 2);
  CHECK_FALSE(parser.parse("播放音乐", context("old-default")).ok());
  CHECK_FALSE(parser.parse("开始播放停止播放音量减", context("too-many")).ok());
}

TEST_CASE("Configured grammar rejects phrase collisions") {
  dvo::CommandGrammar grammar;
  grammar.pause_phrases = {"播放音乐"};
  CHECK_THROWS_AS(dvo::CommandParser(5, 20, std::move(grammar)), std::invalid_argument);
}

TEST_CASE("Parser rejects invalid context empty input and invalid UTF-8") {
  const dvo::CommandParser parser;
  CHECK_FALSE(parser.parse("播放音乐", {}).ok());
  auto missing_runtime = context();
  missing_runtime.runtime_session_id.clear();
  CHECK_FALSE(parser.parse("播放音乐", missing_runtime).ok());
  CHECK_FALSE(parser.parse("，。！？", context()).ok());
  const std::string invalid{"\xF0\x28\x8C\x28", 4};
  const auto result = parser.parse(invalid, context());
  CHECK_FALSE(result.ok());
  REQUIRE(result.error.has_value());
  CHECK(result.error->code == "invalid_utf8");
}
