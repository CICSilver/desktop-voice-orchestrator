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
  REQUIRE(store.validate(config).ok());
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
                                            {"vad", {{"threshold", 0.33}}}});
  REQUIRE(result.ok());
  REQUIRE(config.kws.threshold == Catch::Approx(0.42F));
  REQUIRE(config.revision == initial_revision + 1);
  store.save_overrides(config);
  REQUIRE(std::filesystem::exists(local));
  const auto reloaded = store.load();
  REQUIRE(reloaded.audio.microphone_device == "test-endpoint");
  REQUIRE(reloaded.kws.threshold == Catch::Approx(0.42F));

  const auto before = config;
  result = store.apply_patch(config, {{"kws", {{"threshold", 2.0}}}});
  REQUIRE_FALSE(result.ok());
  REQUIRE(config.kws.threshold == before.kws.threshold);
  std::filesystem::remove_all(temp);
}
