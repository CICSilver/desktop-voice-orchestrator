#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dvo/action_executor.h"
#include "dvo/announcement.h"
#include "dvo/command_parser.h"
#include "dvo/config.h"
#include "dvo/runtime.h"

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

  // ASR may drop the unstressed "之".
  const auto without_zhi = parser.parse("降低音量百分十", context("no-zhi"));
  REQUIRE(without_zhi.ok());
  CHECK(without_zhi.plan->actions[0].volume_delta_percent == -10);
  CHECK_FALSE(parser.parse("降低音量百分", context("no-number")).ok());
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

  // Natural-language listen requests ("再听音乐") must NEVER be rewritten to pause.
  const auto listen_again =
      parser.parse("再听音乐小助手", wake_context);
  CHECK_FALSE(listen_again.ok());

  CHECK_FALSE(parser.parse("再连音乐", wake_context).ok());
}

TEST_CASE("Wake-confirmed readings tolerate residue, fillers and a misheard wake word") {
  const dvo::CommandParser parser;
  auto wake_context = context("tolerant");
  wake_context.wake_word = "小助手";
  const auto actions_of = [&](const char* text) {
    const auto result = parser.parse(text, wake_context);
    return result.ok() ? result.plan->normalized_text : std::string("REJECTED");
  };

  // Short residue that is not itself a command, on the far side of the wake
  // word, is recognizer noise or background speech.
  CHECK(actions_of("再连音乐小助手两次") == "暂停音乐");
  CHECK(actions_of("播放音乐小助手动我心") == "播放音乐");
  CHECK(actions_of("回小助手暂停音乐") == "暂停音乐");
  // Hesitations and stray recognizer symbols.
  CHECK(actions_of("增加音量啊小助手") == "增加音量5%");
  CHECK(actions_of("<播放音乐小助手") == "播放音乐");
  // A wake word misheard by one character at either edge.
  CHECK(actions_of("小叔手，增加音量，再播放音乐") == "增加音量5%;播放音乐");
  CHECK(actions_of("暂停音乐，小助") == "暂停音乐");
  // A stuttered word.
  CHECK(actions_of("小助手暂停暂停音乐然后降低音量百分之十") == "暂停音乐;降低音量10%");

  // Longer residue, residue that is itself a command, unknown requests and a
  // near miss in the middle of the sentence all stay rejected or intact.
  CHECK(actions_of("再连音乐小助手两次暂停") == "REJECTED");
  CHECK(actions_of("暂停音小助手播放音乐") == "暂停音乐;播放音乐");
  CHECK(actions_of("小助手今天天气怎么样") == "REJECTED");
  CHECK(actions_of("今天天气怎么样小助手") == "REJECTED");
  CHECK(actions_of("小助手播放音乐然后打开窗口") == "REJECTED");
  CHECK(actions_of("播放小叔手音乐") == "REJECTED");
  CHECK(actions_of("暂停音乐小说") == "REJECTED");

  // Without a confirmed wake word only fillers are tolerated.
  const auto followup = context("tolerant-followup");
  CHECK(parser.parse("增加音量啊", followup).ok());
  CHECK_FALSE(parser.parse("播放音乐动我心", followup).ok());
}

TEST_CASE("A two-character wake word allows only a same-length near miss at an edge") {
  const dvo::CommandParser parser;
  auto wake_context = context("short-wake");
  wake_context.wake_word = "小克";
  const auto actions_of = [&](const char* text) {
    const auto result = parser.parse(text, wake_context);
    return result.ok() ? result.plan->normalized_text : std::string("REJECTED");
  };

  CHECK(actions_of("小克，暂停音乐") == "暂停音乐");
  CHECK(actions_of("增加音量小克") == "增加音量5%");
  // The recognizer writes a homophone once KWS has confirmed the wake word.
  CHECK(actions_of("小可，暂停音乐") == "暂停音乐");
  CHECK(actions_of("播放音乐，小科") == "播放音乐");
  // "克" dropped: the edge pair absorbs "暂", which the one-character
  // omission rule for wake-confirmed commands restores.
  CHECK(actions_of("小暂停音乐") == "暂停音乐");
  CHECK(actions_of("小可爱暂停音乐") == "REJECTED");
  CHECK(actions_of("播放小可音乐") == "REJECTED");

  // The ASR wake probe has no acoustic confirmation and needs the exact word.
  REQUIRE(dvo::find_wake_in_text("小克，暂停音乐", "小克"));
  CHECK(dvo::find_wake_in_text("小克，暂停音乐", "小克")->exact);
  CHECK_FALSE(dvo::find_wake_in_text("小可，暂停音乐", "小克"));
  CHECK_FALSE(dvo::find_wake_in_text("小可爱快来吃饭", "小克"));
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
  for (const auto* text : {"帮我未知命令", "播放音乐好吗", "播放音乐然后未知命令",
                           "播放音乐暂停音乐谢谢", "播放，音乐", "增加音量五",
                           "播放音乐并暂停音乐"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, context());
    CHECK_FALSE(result.ok());
    CHECK_FALSE(result.plan.has_value());
    REQUIRE(result.error.has_value());
  }
  // A sentence-final particle carries no meaning, like a hesitation.
  CHECK(parser.parse("播放音乐吧", context("particle")).ok());
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

TEST_CASE("The shipped grammar accepts everyday phrasings of each command") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();
  const dvo::CommandParser parser(config.commands.default_volume_step_percent,
                                  config.commands.max_spoken_volume_step_percent,
                                  dvo::command_grammar(config.commands));
  auto wake = context("everyday");
  wake.origin = dvo::UtteranceOrigin::keyword;
  wake.wake_word = "小克";
  const auto actions_of = [&](const char* text) {
    const auto result = parser.parse(text, wake);
    return result.ok() ? result.plan->normalized_text : std::string("REJECTED");
  };

  for (const char* text : {"小克，把音乐打开", "开一下音乐吧，小克", "小克继续播放", "小克继续播放音乐",
                           "小克放首歌", "来首歌小克", "小克，开始播放"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, wake);
    REQUIRE(result.ok());
    CHECK(result.plan->actions.front().type == dvo::ActionType::media_play);
  }
  for (const char* text : {"小克暂停", "小克，暂停一下", "停一下歌，小克", "小克先别放了", "小克别放了",
                           "不要放了小克", "小克把音乐关了", "小克停止播放"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, wake);
    REQUIRE(result.ok());
    CHECK(result.plan->actions.front().type == dvo::ActionType::media_pause);
  }
  CHECK(actions_of("小克大声点") == "大声点5%");
  CHECK(actions_of("小克声音小一点百分之十") == "声音小一点10%");
  CHECK(actions_of("小克暂停一下然后调大音量") == "暂停一下;调大音量5%");

  // Short phrases match exactly: a dropped character is not guessed back.
  CHECK(actions_of("小克别了") == "REJECTED");
  CHECK(actions_of("小克暂") == "REJECTED");
  // Four characters or more still tolerate one dropped character.
  const auto dropped = parser.parse("小克停播放", wake);
  REQUIRE(dropped.ok());
  CHECK(dropped.plan->actions.front().type == dvo::ActionType::media_pause);
  // Ordinary sentences that merely contain command words stay rejected.
  CHECK(actions_of("这首歌的音乐还挺好听的") == "REJECTED");
  CHECK(actions_of("暂停一下我去倒杯水") == "REJECTED");
}

TEST_CASE("The shipped grammar skips to the next song and likes the current one") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();
  const dvo::CommandParser parser(config.commands.default_volume_step_percent,
                                  config.commands.max_spoken_volume_step_percent,
                                  dvo::command_grammar(config.commands));
  auto wake = context("next-like");
  wake.wake_word = "小克";
  const auto actions_of = [&](const char* text) {
    const auto result = parser.parse(text, wake);
    return result.ok() ? result.plan->normalized_text : std::string("REJECTED");
  };

  for (const char* text : {"小克切歌", "换歌，小克", "小克，换一首", "小克换一首歌", "下一首小克",
                           "小克下一首歌"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, wake);
    REQUIRE(result.ok());
    REQUIRE(result.plan->actions.size() == 1);
    CHECK(result.plan->actions.front().type == dvo::ActionType::media_next);
  }
  for (const char* text : {"小克喜欢这首歌", "小克，我喜欢这首歌", "喜欢这个，小克", "小克我喜欢这个",
                           "收藏一下小克", "小克收藏一下吧"}) {
    CAPTURE(text);
    const auto result = parser.parse(text, wake);
    REQUIRE(result.ok());
    REQUIRE(result.plan->actions.size() == 1);
    CHECK(result.plan->actions.front().type == dvo::ActionType::media_like);
  }
  // Liking runs before skipping: the executor keeps the spoken order.
  const auto chained = parser.parse("小克收藏一下然后切歌", wake);
  REQUIRE(chained.ok());
  REQUIRE(chained.plan->actions.size() == 2);
  CHECK(chained.plan->actions[0].type == dvo::ActionType::media_like);
  CHECK(chained.plan->actions[1].type == dvo::ActionType::media_next);

  CHECK(actions_of("小克切") == "REJECTED");
  CHECK(actions_of("小克我喜欢这个歌手") == "REJECTED");
  CHECK(actions_of("下一首是什么歌") == "REJECTED");
  CHECK(actions_of("我喜欢这首歌的前奏") == "REJECTED");
}

TEST_CASE("The shipped grammar plays daily recommendations and sets play modes") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();
  const dvo::CommandParser parser(config.commands.default_volume_step_percent,
                                  config.commands.max_spoken_volume_step_percent,
                                  dvo::command_grammar(config.commands));
  auto wake = context("daily-mode");
  wake.wake_word = "小克";
  const auto type_of = [&](const char* text) {
    const auto result = parser.parse(text, wake);
    REQUIRE(result.ok());
    REQUIRE(result.plan->actions.size() == 1);
    return result.plan->actions.front().type;
  };

  for (const char* text : {"小克播放每日推荐", "放每日推荐，小克", "小克打开每日推荐", "小克每日推荐吧"}) {
    CAPTURE(text);
    CHECK(type_of(text) == dvo::ActionType::media_play_daily);
  }
  CHECK(type_of("小克顺序播放") == dvo::ActionType::media_mode_order);
  CHECK(type_of("列表循环小克") == dvo::ActionType::media_mode_list_loop);
  CHECK(type_of("小克，单曲循环") == dvo::ActionType::media_mode_single_loop);
  CHECK(type_of("小克随机播放") == dvo::ActionType::media_mode_shuffle);

  const auto chained = parser.parse("小克随机播放然后播放每日推荐", wake);
  REQUIRE(chained.ok());
  REQUIRE(chained.plan->actions.size() == 2);
  CHECK(chained.plan->actions[0].type == dvo::ActionType::media_mode_shuffle);
  CHECK(chained.plan->actions[1].type == dvo::ActionType::media_play_daily);

  CHECK_FALSE(parser.parse("每日推荐的歌挺好听", wake).ok());
  CHECK_FALSE(parser.parse("小克随机", wake).ok());
}

TEST_CASE("A trailing command prefix completes only after a connector and never from the wake word") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();
  const dvo::CommandParser parser(config.commands.default_volume_step_percent,
                                  config.commands.max_spoken_volume_step_percent,
                                  dvo::command_grammar(config.commands));
  auto wake = context("trailing-prefix");
  wake.wake_word = "小克";
  const auto actions_of = [&](const char* text, const dvo::CommandParseContext& parse_context) {
    const auto result = parser.parse(text, parse_context);
    return result.ok() ? result.plan->normalized_text : std::string("REJECTED");
  };

  // The "小" of a clipped suffix 小克 used to complete to "小声一点" and lower
  // the volume a second time. The residue is dropped, never executed.
  const auto clipped = parser.parse("降低音量百分之十小", wake);
  REQUIRE(clipped.ok());
  REQUIRE(clipped.plan->actions.size() == 1);
  CHECK(clipped.plan->actions[0].type == dvo::ActionType::master_volume_adjust);
  CHECK(clipped.plan->actions[0].volume_delta_percent == -10);
  // Only the residue goes, not the digit before it.
  CHECK(actions_of("降低音量百分之十二小", wake) == "降低音量12%");
  CHECK(actions_of("增加音量，小", wake) == "增加音量5%");
  // Without a wake word to explain it, the residue rejects the sentence.
  CHECK(actions_of("降低音量百分之十小", context("no-wake")) == "REJECTED");

  // Without a connector a trailing prefix is unknown text, not a command.
  CHECK(actions_of("播放音乐大", wake) == "REJECTED");
  CHECK(actions_of("播放音乐大", context("no-wake")) == "REJECTED");

  // After a connector a truncated last command still completes, unless the
  // remainder could be the start of the wake word.
  CHECK(actions_of("播放音乐然后大", wake) == "播放音乐;大声一点5%");
  CHECK(actions_of("播放音乐然后小", wake) == "REJECTED");
  const dvo::CommandParser default_parser;
  const auto truncated = default_parser.parse("播放音乐然后再暂", wake);
  REQUIRE(truncated.ok());
  REQUIRE(truncated.plan->actions.size() == 2);
  CHECK(truncated.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK(truncated.plan->actions[1].type == dvo::ActionType::media_pause);
  CHECK(truncated.plan->normalized_text == "播放音乐;暂停音乐");
}

TEST_CASE("Desktop lyrics commands stay off until phrases are configured") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  auto config = store.load();
  CHECK(config.commands.lyrics_show_phrases.empty());
  CHECK(config.commands.lyrics_hide_phrases.empty());
  auto wake = context("lyrics");
  wake.wake_word = "小克";
  const dvo::CommandParser shipped(5, 20, dvo::command_grammar(config.commands));
  CHECK_FALSE(shipped.parse("小克打开歌词", wake).ok());

  config.commands.lyrics_show_phrases = {"打开歌词"};
  config.commands.lyrics_hide_phrases = {"关闭歌词"};
  const dvo::CommandParser parser(5, 20, dvo::command_grammar(config.commands));
  const auto shown = parser.parse("小克打开歌词", wake);
  REQUIRE(shown.ok());
  CHECK(shown.plan->actions.front().type == dvo::ActionType::media_lyrics_show);
  const auto hidden = parser.parse("关闭歌词，小克", wake);
  REQUIRE(hidden.ok());
  CHECK(hidden.plan->actions.front().type == dvo::ActionType::media_lyrics_hide);
}

TEST_CASE("Empty next and like phrase lists disable those actions") {
  dvo::CommandGrammar grammar;
  CHECK(dvo::CommandParser(5, 20, grammar).parse("下一首", context()).ok());
  grammar.next_phrases.clear();
  grammar.like_phrases.clear();
  const dvo::CommandParser parser(5, 20, std::move(grammar));
  CHECK_FALSE(parser.parse("下一首", context()).ok());
  CHECK_FALSE(parser.parse("喜欢这首歌", context()).ok());
  CHECK(parser.parse("暂停音乐", context()).ok());
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

TEST_CASE("Natural-language listen intents are never hijacked as media.pause in rules mode") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();
  const dvo::CommandParser parser(
      static_cast<int>(config.commands.default_volume_step_percent),
      static_cast<int>(config.commands.max_spoken_volume_step_percent),
      dvo::command_grammar(config.commands));

  auto wake = context("listen-intent");
  wake.origin = dvo::UtteranceOrigin::keyword;
  wake.wake_word = "小克";

  for (const char* text : {"小克，我想听音乐", "小克，再听音乐", "小克，在听音乐",
                           "小克，听一下音乐", "我想听音乐，小克", "再听音乐，小克"}) {
    CAPTURE(text);
    const auto parsed = parser.parse(text, wake);
    CHECK_FALSE(parsed.ok());
    CHECK_FALSE(parsed.plan.has_value());
  }
}

namespace {

class MockCommandRouter final : public dvo::ILlmCommandRouter {
 public:
  using Handler = std::function<dvo::LlmRouteResult(const dvo::LlmRouteRequest&)>;

  explicit MockCommandRouter(Handler handler) : handler_(std::move(handler)) {}

  dvo::LlmRouteResult route(const dvo::LlmRouteRequest& request) const override {
    requests.push_back(request);
    return handler_(request);
  }

  mutable std::vector<dvo::LlmRouteRequest> requests;

 private:
  Handler handler_;
};

class RecordingExecutorBackend final : public dvo::IActionBackend {
 public:
  dvo::ActionResult execute(const dvo::PlannedAction& action,
                            std::stop_token) override {
    executed.push_back(action);
    dvo::ActionResult result;
    result.type = action.type;
    result.adapter = "mock-backend";
    result.status = dvo::ActionStatus::succeeded;
    return result;
  }

  std::vector<dvo::PlannedAction> executed;
};

}  // namespace

TEST_CASE("Hybrid parser keeps fast-path rules local, routes natural phrasings, and enforces strict validation") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml",
                         std::filesystem::temp_directory_path() / "dvo-no-overrides.toml", root);
  const auto config = store.load();

  dvo::LlmRouteResult next_response;
  auto router = std::make_shared<MockCommandRouter>(
      [&](const dvo::LlmRouteRequest&) { return next_response; });

  const dvo::CommandParser parser(
      static_cast<int>(config.commands.default_volume_step_percent),
      static_cast<int>(config.commands.max_spoken_volume_step_percent),
      dvo::command_grammar(config.commands),
      dvo::CommandParserMode::hybrid,
      router);

  auto wake = context("hybrid-case");
  wake.origin = dvo::UtteranceOrigin::keyword;
  wake.wake_word = "小克";

  // 1. Fast-path rule commands match locally without invoking the router.
  const auto fast_pause = parser.parse("小克，暂停音乐", wake);
  REQUIRE(fast_pause.ok());
  CHECK(fast_pause.plan->parser_version == dvo::kCommandParserVersion);
  REQUIRE(fast_pause.plan->actions.size() == 1);
  CHECK(fast_pause.plan->actions[0].type == dvo::ActionType::media_pause);
  CHECK(router->requests.empty());

  // Hard parameter range violations in rules must NOT fall through to the router.
  const auto excessive_vol = parser.parse("小克，增加音量50%", wake);
  CHECK_FALSE(excessive_vol.ok());
  REQUIRE(excessive_vol.error.has_value());
  CHECK(excessive_vol.error->code == "volume_amount_out_of_range");
  CHECK(router->requests.empty());

  // 2. Natural-language listen intent reaches the router with both raw_text and clean stripped_text.
  next_response = dvo::LlmRouteResult{
      .actions = {{.type = dvo::ActionType::media_play}},
      .error = std::nullopt,
      .router_name = "llm.mock",
  };
  const auto listen = parser.parse("小克，我想听音乐", wake);
  REQUIRE(listen.ok());
  REQUIRE(router->requests.size() == 1);
  CHECK(router->requests[0].raw_text == "小克，我想听音乐");
  CHECK(router->requests[0].stripped_text == "我想听音乐");
  CHECK(router->requests[0].wake_word == "小克");
  CHECK(router->requests[0].default_volume_delta_percent == 5);
  CHECK(router->requests[0].maximum_volume_delta_percent == 20);
  CHECK(listen.plan->parser_version == "llm.mock");
  REQUIRE(listen.plan->actions.size() == 1);
  CHECK(listen.plan->actions[0].type == dvo::ActionType::media_play);

  // 3. Compound action ("先把声音调小5%，再放首周杰伦的晴天") preserves order, deduplicates, and announces.
  next_response = dvo::LlmRouteResult{
      .actions =
          {
              {.type = dvo::ActionType::master_volume_adjust,
               .volume_delta_percent = -5},
              {.type = dvo::ActionType::media_search_play,
               .search_query = "晴天",
               .search_type = dvo::MediaSearchType::song,
               .search_artist = "周杰伦"},
          },
      .error = std::nullopt,
      .router_name = "llm.mock",
  };
  auto compound_ctx = wake;
  compound_ctx.utterance_id = "compound-jay-chou";
  compound_ctx.source = dvo::ExecutionSource::live;
  compound_ctx.execution_mode = dvo::ExecutionMode::live;
  const auto compound =
      parser.parse("小克，先把声音调小5%，再放首周杰伦的晴天", compound_ctx);
  REQUIRE(compound.ok());
  REQUIRE(router->requests.size() == 2);
  CHECK(router->requests[1].raw_text == "小克，先把声音调小5%，再放首周杰伦的晴天");
  CHECK(router->requests[1].stripped_text == "先把声音调小5%，再放首周杰伦的晴天");
  REQUIRE(compound.plan->actions.size() == 2);
  CHECK(compound.plan->actions[0].sequence == 1);
  CHECK(compound.plan->actions[0].type == dvo::ActionType::master_volume_adjust);
  CHECK(compound.plan->actions[0].volume_delta_percent == -5);
  CHECK(compound.plan->actions[1].sequence == 2);
  CHECK(compound.plan->actions[1].type == dvo::ActionType::media_search_play);
  CHECK(compound.plan->actions[1].search_query == "晴天");
  CHECK(compound.plan->actions[1].search_type == dvo::MediaSearchType::song);
  CHECK(compound.plan->actions[1].search_artist == "周杰伦");

  const auto announcement =
      dvo::make_plan_announcement(*compound.plan, "announce-compound");
  CHECK(announcement.text == "正在执行：降低音量 5%，然后播放 周杰伦 的 晴天");

  auto exec_backend = std::make_shared<RecordingExecutorBackend>();
  std::vector<dvo::ActionResult> exec_results;
  dvo::OrderedActionExecutor executor(
      exec_backend, {},
      [&](const dvo::ActionResult& r) { exec_results.push_back(r); });
  REQUIRE(executor.try_submit(*compound.plan).accepted());
  REQUIRE(executor.try_submit(*compound.plan).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  REQUIRE(exec_backend->executed.size() == 2);
  CHECK(exec_backend->executed[0].type == dvo::ActionType::master_volume_adjust);
  CHECK(exec_backend->executed[1].type == dvo::ActionType::media_search_play);
  CHECK(exec_backend->executed[1].search_query == "晴天");
  CHECK(exec_backend->executed[1].search_artist == "周杰伦");
  // First submission: 2*(started + succeeded) = 4; duplicate submission: 2*deduplicated = 2 -> total 6.
  REQUIRE(exec_results.size() == 6);
  CHECK(exec_results[4].status == dvo::ActionStatus::deduplicated);
  CHECK(exec_results[5].status == dvo::ActionStatus::deduplicated);
  executor.stop();

  // 4. Router validation strictly rejects chitchat empty arrays and malformed parameters.
  next_response = dvo::LlmRouteResult{.actions = {}, .router_name = "llm.mock"};
  const auto chitchat = parser.parse("小克，今天天气真不错", wake);
  CHECK_FALSE(chitchat.ok());
  REQUIRE(chitchat.error.has_value());
  CHECK(chitchat.error->code == "no_command_intent");

  next_response = dvo::LlmRouteResult{
      .actions = {{.type = dvo::ActionType::media_search_play,
                   .search_query = "   "}},
      .router_name = "llm.mock",
  };
  const auto empty_query = parser.parse("小克，随便搜首歌", wake);
  CHECK_FALSE(empty_query.ok());
  REQUIRE(empty_query.error.has_value());
  CHECK(empty_query.error->code == "invalid_search_query");

  next_response = dvo::LlmRouteResult{
      .actions = {{.type = dvo::ActionType::master_volume_adjust,
                   .volume_delta_percent = 50}},
      .router_name = "llm.mock",
  };
  const auto bad_router_volume = parser.parse("小克，声音拉满", wake);
  CHECK_FALSE(bad_router_volume.ok());
  REQUIRE(bad_router_volume.error.has_value());
  CHECK(bad_router_volume.error->code == "volume_amount_out_of_range");

  dvo::LlmRouteResult nine_actions;
  nine_actions.router_name = "llm.mock";
  nine_actions.actions.assign(9, dvo::ProposedAction{.type = dvo::ActionType::media_next});
  next_response = nine_actions;
  const auto too_many = parser.parse("小克，一直切歌", wake);
  CHECK_FALSE(too_many.ok());
  REQUIRE(too_many.error.has_value());
  CHECK(too_many.error->code == "too_many_actions");
}

TEST_CASE("Fast-path app.open rules match registered apps locally without shadowing media.play") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(
      root / "config/default.toml",
      root / "config/nonexistent-app-fastpath-local.toml",
      root);
  auto cfg = store.load();
  cfg.apps.scan_system_apps = false;
  cfg.apps.quickstart_dirs.clear();
  // Ensure even a hypothetical app alias named "音乐" cannot steal "打开音乐"!
  cfg.apps.entries.push_back(dvo::AppEntryConfig{
      .id = "fake_music_app",
      .display_name = "音乐",
      .aliases = {"音乐", "每日推荐"},
  });

  const auto catalog = std::make_shared<const dvo::AppCatalog>(
      dvo::AppCatalog::from_config(cfg.apps));

  std::size_t router_calls = 0;
  struct CountingRouter final : dvo::ILlmCommandRouter {
    std::size_t* calls{};
    dvo::LlmRouteResult route(const dvo::LlmRouteRequest&) const override {
      ++(*calls);
      return {.actions = {}, .router_name = "llm.counting"};
    }
  };
  auto counting_router = std::make_shared<CountingRouter>();
  counting_router->calls = &router_calls;

  const dvo::CommandParser parser(
      static_cast<int>(cfg.commands.default_volume_step_percent),
      static_cast<int>(cfg.commands.max_spoken_volume_step_percent),
      dvo::command_grammar(cfg.commands, catalog.get()),
      dvo::CommandParserMode::hybrid,
      counting_router);

  auto wake = context("app-fastpath-wake");
  wake.origin = dvo::UtteranceOrigin::keyword;
  wake.wake_word = "小克";

  const auto open_steam = parser.parse("小克，打开 Steam", wake);
  REQUIRE(open_steam.ok());
  REQUIRE(open_steam.plan->actions.size() == 1);
  CHECK(open_steam.plan->actions[0].type == dvo::ActionType::app_open);
  CHECK(open_steam.plan->actions[0].app_id == "steam");
  CHECK(open_steam.plan->actions[0].app_display_name == "Steam");
  CHECK(router_calls == 0);

  const auto open_ps = parser.parse("打开 PowerShell", context("app-fastpath-ps"));
  REQUIRE(open_ps.ok());
  REQUIRE(open_ps.plan->actions.size() == 1);
  CHECK(open_ps.plan->actions[0].type == dvo::ActionType::app_open);
  CHECK(open_ps.plan->actions[0].app_id == "powershell");
  CHECK(router_calls == 0);

  const auto open_homophone = parser.parse("小克，启动死盯然后切换到迪斯科", wake);
  REQUIRE(open_homophone.ok());
  REQUIRE(open_homophone.plan->actions.size() == 2);
  CHECK(open_homophone.plan->actions[0].type == dvo::ActionType::app_open);
  CHECK(open_homophone.plan->actions[0].app_id == "steam");
  CHECK(open_homophone.plan->actions[1].type == dvo::ActionType::app_open);
  CHECK(open_homophone.plan->actions[1].app_id == "discord");
  CHECK(router_calls == 0);

  // Critical regression guard: "打开音乐" and "打开每日推荐" MUST keep hitting media rules!
  const auto open_music = parser.parse("小克，打开音乐", wake);
  REQUIRE(open_music.ok());
  REQUIRE(open_music.plan->actions.size() == 1);
  CHECK(open_music.plan->actions[0].type == dvo::ActionType::media_play);
  CHECK_FALSE(open_music.plan->actions[0].app_id.has_value());
  CHECK(router_calls == 0);

  const auto open_daily = parser.parse("小克，打开每日推荐", wake);
  REQUIRE(open_daily.ok());
  REQUIRE(open_daily.plan->actions.size() == 1);
  CHECK(open_daily.plan->actions[0].type == dvo::ActionType::media_play_daily);
  CHECK(router_calls == 0);
}
