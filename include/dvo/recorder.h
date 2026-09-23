#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>

#include <nlohmann/json.hpp>

#include "dvo/audio_types.h"
#include "dvo/config.h"
#include "dvo/spsc_queue.h"
#include "dvo/wav_file.h"

namespace dvo {

struct RecordEvent { nlohmann::json value; };
struct RecordCandidate { std::shared_ptr<const UtteranceCandidate> value; };
using RecordItem = std::variant<AudioPacket, NormalizedFrame, RecordEvent, RecordCandidate>;

class SessionRecorder {
 public:
  explicit SessionRecorder(std::size_t queue_capacity = 4096);
  ~SessionRecorder();

  std::filesystem::path start(const RecordingConfig& config, const nlohmann::json& manifest);
  void stop(const nlohmann::json& final_metrics = nlohmann::json::object());
  bool try_enqueue(RecordItem item);
  [[nodiscard]] bool active() const { return active_.load(std::memory_order_acquire); }
  [[nodiscard]] bool incomplete() const { return incomplete_.load(std::memory_order_acquire); }
  [[nodiscard]] std::filesystem::path session_path() const;
  [[nodiscard]] std::int64_t started_at_unix_ms() const;

 private:
  struct StreamStats {
    AudioFormat format{};
    std::uint64_t frames{};
    std::uint64_t packets{};
    std::uint64_t first_qpc_100ns{};
    std::uint64_t last_qpc_100ns{};
    std::uint64_t synthetic_packets{};
    std::uint64_t timestamp_errors{};
    std::uint64_t discontinuities{};
    std::uint64_t last_stream_epoch{};
  };

  void run();
  void write_item(const AudioPacket& packet);
  void write_item(const NormalizedFrame& frame);
  void write_item(const RecordEvent& event);
  void write_item(const RecordCandidate& candidate);
  void close_files();

  SpscQueue<RecordItem> queue_;
  std::jthread thread_;
  std::atomic<bool> active_{};
  std::atomic<bool> incomplete_{};
  std::atomic<bool> closing_{};
  std::atomic<std::uint32_t> inflight_{};
  std::atomic<std::uint64_t> dropped_items_{};
  std::mutex enqueue_mutex_;
  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  std::filesystem::path session_path_;
  std::int64_t started_at_unix_ms_{};
  nlohmann::json manifest_;
  nlohmann::json action_results_{nlohmann::json::array()};
  nlohmann::json capture_events_{nlohmann::json::array()};
  std::unique_ptr<FloatWavWriter> mic_;
  std::unique_ptr<FloatWavWriter> loopback_;
  std::unique_ptr<FloatWavWriter> processed_;
  StreamStats mic_stats_;
  StreamStats loopback_stats_;
  StreamStats processed_stats_;
  std::ofstream timeline_;
  std::ofstream events_;
};

}  // namespace dvo
