#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/audio_types.h"

namespace dvo {

class ReplayController {
 public:
  using PacketCallback = std::function<void(AudioPacket)>;
  explicit ReplayController(PacketCallback callback);
  ~ReplayController();

  void open(const std::filesystem::path& session);
  void play();
  void pause();
  void seek_seconds(double seconds);
  void set_speed(double speed);
  void stop();
  [[nodiscard]] nlohmann::json state() const;

 private:
  struct TimelineEntry {
    AudioStreamKind kind{};
    std::uint64_t offset_frames{};
    std::uint64_t frame_count{};
    std::uint64_t qpc_100ns{};
    std::uint64_t device_position{};
    bool silent{};
    bool discontinuity{};
    bool synthetic{};
  };

  void run(std::stop_token stop);
  void load_timeline();

  PacketCallback callback_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::jthread thread_;
  std::filesystem::path session_;
  std::vector<TimelineEntry> entries_;
  std::size_t cursor_{};
  bool playing_{};
  double speed_{1.0};
  std::optional<std::size_t> seek_target_;
  bool resume_after_seek_{};
};

}  // namespace dvo
