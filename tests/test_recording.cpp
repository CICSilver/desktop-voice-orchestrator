#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
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
  REQUIRE(recorder.try_enqueue(dvo::RecordEvent{{
      {"type", "capture_capability"},
      {"source", "live"},
      {"payload", {{"message", "microphone: raw stream option active"}}}}}));
  REQUIRE(recorder.try_enqueue(dvo::RecordEvent{{
      {"type", "action_succeeded"},
      {"source", "live"},
      {"payload", {{"action_id", "action-1"}, {"status", "succeeded"}}}}}));
  REQUIRE(recorder.try_enqueue(dvo::RecordEvent{{
      {"type", "action_started"},
      {"source", "live"},
      {"payload", {{"action_id", "action-1"}, {"status", "started"}}}}}));
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
    REQUIRE(manifest["capture_events"].size() == 1);
    REQUIRE(manifest["action_results"].size() == 1);
    REQUIRE(manifest["action_results"][0]["payload"]["action_id"] == "action-1");
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
  std::atomic<int> replay_state_updates{};
  {
    dvo::ReplayController replay([&](dvo::AudioPacket packet) {
      std::scoped_lock lock(replay_mutex);
      replayed.push_back(std::move(packet));
      replay_cv.notify_all();
    }, [&] {
      replay_state_updates.fetch_add(1, std::memory_order_relaxed);
    });
    replay.open(session);
    replay.set_speed(0.0);
    replay.play();
    std::unique_lock lock(replay_mutex);
    REQUIRE(replay_cv.wait_for(lock, std::chrono::seconds(2), [&] { return replayed.size() == 2; }));
    lock.unlock();
    REQUIRE(replay.wait_until_finished(std::chrono::seconds(2)));
    CHECK_FALSE(replay.state().value("dispatching", true));
    CHECK(replay.state()["error"].is_null());

    replay.play();
    lock.lock();
    REQUIRE(replay_cv.wait_for(lock, std::chrono::seconds(2), [&] { return replayed.size() == 4; }));
    lock.unlock();
    REQUIRE(replay.wait_until_finished(std::chrono::seconds(2)));
    CHECK(replay.state()["error"].is_null());
    CHECK(replay_state_updates.load(std::memory_order_relaxed) >= 2);
  }
  REQUIRE(replayed[0].stream == dvo::AudioStreamKind::microphone);
  REQUIRE(replayed[1].stream == dvo::AudioStreamKind::loopback);
  REQUIRE(replayed[2].stream == dvo::AudioStreamKind::microphone);
  REQUIRE(replayed[3].stream == dvo::AudioStreamKind::loopback);
  REQUIRE(replayed[0].samples.front() == Catch::Approx(0.25F));
  REQUIRE(replayed[1].samples.front() == Catch::Approx(-0.1F));
  REQUIRE(replayed[2].samples.front() == Catch::Approx(0.25F));
  REQUIRE(replayed[3].samples.front() == Catch::Approx(-0.1F));

  const auto no_mic_session = root / "no-mic";
  std::filesystem::create_directories(no_mic_session);
  {
    dvo::FloatWavWriter writer;
    writer.open(no_mic_session / "loopback.wav", {48000, 1});
    std::vector<float> samples(480, 0.1F);
    writer.write(samples);
  }
  {
    std::ofstream timeline(no_mic_session / "timeline.ndjson", std::ios::binary);
    timeline << nlohmann::json{{"kind", "loopback"}, {"offset_frames", 0},
                               {"frame_count", 480}, {"qpc_100ns", 1000}}.dump()
             << '\n';
  }
  {
    dvo::ReplayController replay([](dvo::AudioPacket) {});
    bool rejected{};
    try {
      replay.open(no_mic_session);
    } catch (const std::runtime_error& error) {
      rejected = true;
      CHECK(std::string(error.what()).find("no microphone") != std::string::npos);
    }
    CHECK(rejected);
  }

  std::mutex blocked_mutex;
  std::condition_variable blocked_cv;
  bool callback_entered{};
  bool release_callback{};
  {
    dvo::ReplayController replay([&](dvo::AudioPacket) {
      std::unique_lock lock(blocked_mutex);
      callback_entered = true;
      blocked_cv.notify_all();
      blocked_cv.wait(lock, [&] { return release_callback; });
    });
    replay.open(session);
    replay.set_speed(0.0);
    replay.play();
    {
      std::unique_lock lock(blocked_mutex);
      REQUIRE(blocked_cv.wait_for(lock, std::chrono::seconds(2),
                                  [&] { return callback_entered; }));
    }
    CHECK_FALSE(replay.wait_until_finished(std::chrono::milliseconds(10)));
    {
      std::lock_guard lock(blocked_mutex);
      release_callback = true;
      blocked_cv.notify_all();
    }
    REQUIRE(replay.wait_until_finished(std::chrono::seconds(2)));
  }
  std::filesystem::remove_all(root);
}

TEST_CASE("recording manifest is incomplete after a timestamp discontinuity") {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("dvo-record-gap-test-" + std::to_string(suffix));
  dvo::RecordingConfig config;
  config.session_root = root;
  dvo::SessionRecorder recorder(16);
  const auto session = recorder.start(config, nlohmann::json::object());

  dvo::AudioPacket packet;
  packet.stream = dvo::AudioStreamKind::microphone;
  packet.format = {48000, 1};
  packet.samples.assign(480, 0.0F);
  packet.qpc_100ns = 10'000;
  packet.discontinuity = true;
  REQUIRE(recorder.try_enqueue(std::move(packet)));
  recorder.stop();

  {
    std::ifstream manifest_file(session / "manifest.json");
    const auto manifest = nlohmann::json::parse(manifest_file);
    CHECK_FALSE(manifest["complete"].get<bool>());
    CHECK(manifest["streams"]["microphone"]["discontinuities"] == 1);
  }
  std::filesystem::remove_all(root);
}
