#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "dvo/config.h"

namespace {
std::filesystem::path temporary_directory() {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / ("dvo-config-test-" + std::to_string(suffix));
  std::filesystem::create_directories(path / "config");
  return path;
}
}  // namespace

TEST_CASE("checked-in default configuration is valid") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  dvo::ConfigStore store(root / "config/default.toml", root / "config/nonexistent-test-local.toml", root);
  const auto config = store.load();
  REQUIRE(config.audio.target_sample_rate == 16000);
  REQUIRE(config.ring.duration_ms == 20000);
  REQUIRE(config.commands.play_phrases ==
          std::vector<std::string>{"播放音乐", "打开音乐"});
  REQUIRE(config.commands.action_timeout_ms == 2000);
  REQUIRE(config.activation.enabled);
  REQUIRE(config.activation.idle_timeout_ms == 6000);
  REQUIRE(config.activation.hard_limit_ms == 20000);
  REQUIRE(config.announcements.backend == "log");
  REQUIRE_FALSE(config.announcements.barge_in);
  REQUIRE(config.commands.connectors ==
          std::vector<std::string>{"然后再", "然后", "再", "后", "接着", "并且",
                                   "以及", "和", "还有"});
  REQUIRE(config.aec.microphone_channel_index == 0);
  REQUIRE(config.aec.auto_delay_enabled);
  REQUIRE(config.aec.auto_delay_min_ms == 0);
  REQUIRE(config.aec.auto_delay_max_ms == 250);
  REQUIRE(config.aec.auto_delay_window_ms == 1000);
  REQUIRE(config.aec.auto_delay_update_ms == 500);
  REQUIRE(config.aec.auto_delay_min_correlation == Catch::Approx(0.35F));
  REQUIRE(store.validate(config).ok());

  auto unsafe = config;
  unsafe.asr.exact_final_redecode = false;
  const auto validation = store.validate(unsafe);
  REQUIRE_FALSE(validation.ok());

  unsafe = config;
  unsafe.web.bind = "localhost";
  REQUIRE_FALSE(store.validate(unsafe).ok());
}

TEST_CASE("AEC calibration patch is validated exposed and persisted") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  const auto temp = temporary_directory();
  const auto local = temp / "config/local.toml";
  dvo::ConfigStore store(root / "config/default.toml", local, root);
  auto config = store.load();

  auto result = store.apply_patch(
      config,
      {{"aec",
        {{"microphone_channel_index", -1},
         {"delay_offset_ms", 13},
         {"auto_delay_enabled", true},
         {"auto_delay_min_ms", 30},
         {"auto_delay_max_ms", 180},
         {"auto_delay_window_ms", 1200},
         {"auto_delay_update_ms", 300},
         {"auto_delay_min_correlation", 0.52}}}});
  REQUIRE(result.ok());
  const auto public_config = store.to_public_json(config);
  CHECK(public_config["aec"]["microphone_channel_index"] == -1);
  CHECK(public_config["aec"]["auto_delay_min_correlation"] ==
        Catch::Approx(0.52));
  const auto manifest = store.to_manifest_json(config);
  CHECK(manifest["aec"]["auto_delay_max_ms"] == 180);

  store.save_overrides(config);
  const auto reloaded = store.load();
  CHECK(reloaded.aec.microphone_channel_index == -1);
  CHECK(reloaded.aec.delay_offset_ms == 13);
  CHECK(reloaded.aec.auto_delay_min_ms == 30);
  CHECK(reloaded.aec.auto_delay_window_ms == 1200);

  const auto before = reloaded;
  auto invalid = reloaded;
  result = store.apply_patch(invalid,
                             {{"aec", {{"auto_delay_min_ms", 200},
                                        {"auto_delay_max_ms", 100}}}});
  REQUIRE_FALSE(result.ok());
  CHECK(invalid.aec.auto_delay_min_ms == before.aec.auto_delay_min_ms);
  std::filesystem::remove_all(temp);
}

TEST_CASE("command grammar patch uses nested phrases and survives overlay reload") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  const auto temp = temporary_directory();
  const auto local = temp / "config/local.toml";
  dvo::ConfigStore store(root / "config/default.toml", local, root);
  auto config = store.load();
  const auto original = config;

  auto result = store.apply_patch(
      config,
      {{"commands",
        {{"default_volume_step_percent", 7},
         {"max_actions_per_utterance", 2},
         {"action_timeout_ms", 1350},
         {"queue_capacity", 17},
         {"connectors", {"接下来"}},
         {"phrases",
          {{"play", {"开始播放"}},
           {"pause", {"停止播放"}},
           {"volume_up", {"音量加"}},
           {"volume_down", {"音量减"}}}}}}});
  REQUIRE(result.ok());
  CHECK(config.commands.default_volume_step_percent == 7);
  CHECK(config.commands.max_actions_per_utterance == 2);
  CHECK(config.commands.play_phrases == std::vector<std::string>{"开始播放"});
  const auto public_config = store.to_public_json(config);
  CHECK(public_config["commands"]["phrases"]["play"] ==
        nlohmann::json::array({"开始播放"}));

  store.save_overrides(config);
  const auto reloaded = store.load();
  CHECK(reloaded.commands.connectors == std::vector<std::string>{"接下来"});
  CHECK(reloaded.commands.volume_down_phrases == std::vector<std::string>{"音量减"});
  CHECK(reloaded.commands.action_timeout_ms == 1350);
  CHECK(reloaded.commands.queue_capacity == 17);

  auto invalid = original;
  result = store.apply_patch(
      invalid,
      {{"commands", {{"phrases", {{"play", {"同一个"}},
                                    {"pause", {"同一个"}}}}}}});
  REQUIRE_FALSE(result.ok());
  CHECK(invalid.commands.play_phrases == original.commands.play_phrases);
  std::filesystem::remove_all(temp);
}

TEST_CASE("hot config patch validates and saves an overlay") {
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  const auto temp = temporary_directory();
  const auto local = temp / "config/local.toml";
  dvo::ConfigStore store(root / "config/default.toml", local, root);
  auto config = store.load();
  const auto initial_revision = config.revision;
  auto result = store.apply_patch(config, {{"audio", {{"microphone_device", "test-endpoint"}}},
                                            {"kws", {{"threshold", 0.42}}},
                                            {"vad", {{"threshold", 0.33}}},
                                            {"activation", {{"idle_timeout_ms", 5000},
                                                            {"hard_limit_ms", 18000}}},
                                            {"announcements", {{"enabled", false},
                                                               {"tail_guard_ms", 250}}}});
  REQUIRE(result.ok());
  REQUIRE(config.kws.threshold == Catch::Approx(0.42F));
  REQUIRE(config.revision == initial_revision + 1);
  REQUIRE(config.activation.idle_timeout_ms == 5000);
  REQUIRE_FALSE(config.announcements.enabled);
  store.save_overrides(config);
  REQUIRE(std::filesystem::exists(local));
  const auto reloaded = store.load();
  REQUIRE(reloaded.audio.microphone_device == "test-endpoint");
  REQUIRE(reloaded.kws.threshold == Catch::Approx(0.42F));
  REQUIRE(reloaded.activation.hard_limit_ms == 18000);
  REQUIRE(reloaded.announcements.tail_guard_ms == 250);

  const auto before = config;
  result = store.apply_patch(config, {{"kws", {{"threshold", 2.0}}}});
  REQUIRE_FALSE(result.ok());
  REQUIRE(config.kws.threshold == before.kws.threshold);

  result = store.apply_patch(config, {{"activation", {{"idle_timeout_ms", 11000}}}});
  REQUIRE_FALSE(result.ok());
  std::filesystem::remove_all(temp);
}
