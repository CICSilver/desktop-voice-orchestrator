#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/audio_types.h"

namespace dvo {

class ReplayController {
 public:
  using PacketCallback = std::function<void(AudioPacket)>;
  using StateCallback = std::function<void(nlohmann::json)>;
  explicit ReplayController(PacketCallback callback,
                            StateCallback state_callback = {});
  ~ReplayController();

  void open(const std::filesystem::path& session);
  void play();
  void pause();
  void seek_seconds(double seconds);
  void seek_seconds(double seconds, bool resume_after_seek);
  void set_speed(double speed);
  [[nodiscard]] bool wait_until_quiescent(std::chrono::milliseconds timeout);
  [[nodiscard]] bool wait_until_finished(std::chrono::milliseconds timeout);
  void stop();
  [[nodiscard]] nlohmann::json state() const;

 private:
  struct TimelineEntry {
    AudioStreamKind kind{};
    std::uint64_t offset_frames{};
    std::uint64_t frame_count{};
    std::uint64_t qpc_100ns{};
    std::uint64_t arrival_qpc_100ns{};
    std::uint64_t device_position{};
    std::uint64_t stream_epoch{};
    std::uint64_t sequence{};
    bool silent{};
    bool discontinuity{};
    bool timestamp_error{};
    bool synthetic{};
  };

  void run(std::stop_token stop);
  void load_timeline();
  [[nodiscard]] nlohmann::json state_locked() const;
  void publish_state(nlohmann::json state) const;

  PacketCallback callback_;
  StateCallback state_callback_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::jthread thread_;
  std::filesystem::path session_;
  std::vector<TimelineEntry> entries_;
  std::size_t cursor_{};
  bool playing_{};
  bool dispatching_{};
  double speed_{1.0};
  std::string last_error_;
  std::uint64_t next_progress_qpc_100ns_{};
  std::optional<std::size_t> seek_target_;
  bool resume_after_seek_{};
};

}  // namespace dvo
