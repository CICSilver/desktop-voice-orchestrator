#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "dvo/recorder.h"
#include "dvo/replay.h"

TEST_CASE("recording session stores all three streams and timeline") {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() / ("dvo-record-test-" + std::to_string(suffix));
  dvo::RecordingConfig config;
  config.session_root = root;
  dvo::SessionRecorder recorder(128);
  const auto session = recorder.start(config, {{"test", true}});

  dvo::AudioPacket mic;
  mic.stream = dvo::AudioStreamKind::microphone;
  mic.format = {48000, 1};
  mic.samples.assign(480, 0.25F);
  mic.qpc_100ns = 1000;
  REQUIRE(recorder.try_enqueue(mic));

  auto loop = mic;
  loop.stream = dvo::AudioStreamKind::loopback;
  loop.format = {48000, 2};
  loop.samples.assign(960, -0.1F);
  loop.qpc_100ns = 1100;
  REQUIRE(recorder.try_enqueue(loop));

  dvo::NormalizedFrame frame;
  frame.first_sample = 0;
  frame.samples.fill(0.5F);
  REQUIRE(recorder.try_enqueue(frame));
  REQUIRE(recorder.try_enqueue(dvo::RecordEvent{{{"type", "test"}}}));
  recorder.stop();

  REQUIRE(std::filesystem::exists(session / "mic.wav"));
  REQUIRE(std::filesystem::exists(session / "loopback.wav"));
  REQUIRE(std::filesystem::exists(session / "processed.wav"));
  REQUIRE(std::filesystem::exists(session / "timeline.ndjson"));
  REQUIRE(std::filesystem::exists(session / "manifest.json"));
  {
    std::ifstream manifest_file(session / "manifest.json");
    const auto manifest = nlohmann::json::parse(manifest_file);
    REQUIRE(manifest["complete"].get<bool>());
    REQUIRE(manifest["streams"]["microphone"]["frames"].get<std::uint64_t>() == 480);
    REQUIRE(manifest["streams"]["loopback"]["frames"].get<std::uint64_t>() == 480);
    REQUIRE(manifest["streams"]["processed"]["frames"].get<std::uint64_t>() == 160);
  }

  {
    dvo::FloatWavReader reader(session / "processed.wav");
    REQUIRE(reader.format().sample_rate == 16000);
    REQUIRE(reader.frame_count() == 160);
    REQUIRE(reader.read_frames(0, 160).front() == Catch::Approx(0.5F));
  }

  std::mutex replay_mutex;
  std::condition_variable replay_cv;
  std::vector<dvo::AudioPacket> replayed;
  {
    dvo::ReplayController replay([&](dvo::AudioPacket packet) {
      std::scoped_lock lock(replay_mutex);
      replayed.push_back(std::move(packet));
      replay_cv.notify_all();
    });
    replay.open(session);
    replay.set_speed(0.0);
    replay.play();
    std::unique_lock lock(replay_mutex);
    REQUIRE(replay_cv.wait_for(lock, std::chrono::seconds(2), [&] { return replayed.size() == 2; }));
  }
  REQUIRE(replayed[0].stream == dvo::AudioStreamKind::microphone);
  REQUIRE(replayed[1].stream == dvo::AudioStreamKind::loopback);
  REQUIRE(replayed[0].samples.front() == Catch::Approx(0.25F));
  REQUIRE(replayed[1].samples.front() == Catch::Approx(-0.1F));
  std::filesystem::remove_all(root);
}
