#pragma once

#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "dvo/config.h"
#include "dvo/debug_server.h"
#include "dvo/inference.h"
#include "dvo/preprocessor.h"
#include "dvo/recorder.h"
#include "dvo/replay.h"
#include "dvo/ring_buffer.h"
#include "dvo/segmenter.h"
#include "dvo/spsc_queue.h"
#include "dvo/wasapi_capture.h"

namespace dvo {

class VoiceFrontendRuntime {
 public:
  VoiceFrontendRuntime(AppConfig config, ConfigStore store);
  ~VoiceFrontendRuntime();

  void start_live(bool enable_web = true);
  void start_replay(const std::filesystem::path& session, bool enable_web = true,
                    double speed = 1.0);
  [[nodiscard]] nlohmann::json run_benchmark(const std::filesystem::path& session);
  void stop();
  [[nodiscard]] bool running() const { return running_.load(std::memory_order_acquire); }
  [[nodiscard]] std::string debug_url() const { return debug_.url(); }

 private:
  struct SignalSummary { float min{}, max{}, rms{}; };
  void initialize_pipeline();
  void reset_pipeline(bool discontinuity);
  void start_processing();
  void start_captures();
  void stop_captures();
  void enqueue_packet(AudioPacket packet);
  void processing_loop(std::stop_token stop);
  void process_microphone(AudioPacket packet);
  void process_frame(const NormalizedFrame& frame, const SignalSummary& microphone,
                     const SignalSummary& loopback);
  void emit_event(std::string type, nlohmann::json payload,
                  std::uint64_t timestamp_sample = 0, std::string source = "live");
  void queue_capture_event(std::string type, std::string message);
  void drain_capture_events();
  [[nodiscard]] nlohmann::json handle_command(const nlohmann::json& command);
  [[nodiscard]] nlohmann::json session_list() const;
  [[nodiscard]] nlohmann::json metrics_json() const;
  [[nodiscard]] static SignalSummary summarize(const std::vector<float>& values);
  [[nodiscard]] static SignalSummary summarize(const NormalizedFrame& frame);

  mutable std::mutex config_mutex_;
  AppConfig config_;
  ConfigStore config_store_;
  std::unique_ptr<TimedRingBuffer> ring_;
  std::unique_ptr<IAudioPreprocessor> preprocessor_;
  std::unique_ptr<IVadDetector> vad_;
  std::unique_ptr<IKeywordSpotter> kws_;
  std::unique_ptr<UtteranceSegmenter> segmenter_;
  std::mutex pipeline_mutex_;

  std::unique_ptr<SpscQueue<AudioPacket>> microphone_queue_;
  std::unique_ptr<SpscQueue<AudioPacket>> loopback_queue_;
  std::optional<AudioPacket> latest_loopback_;
  SignalSummary latest_loopback_summary_{};
  std::atomic<bool> microphone_overflow_{};
  std::atomic<bool> loopback_overflow_{};

  WasapiCapture microphone_capture_;
  WasapiCapture loopback_capture_;
  std::unique_ptr<ReplayController> replay_;
  SessionRecorder recorder_;
  DebugServer debug_;
  std::jthread processing_thread_;
  std::atomic<bool> running_{};
  std::atomic<bool> replay_mode_{};

  mutable std::mutex capture_events_mutex_;
  std::deque<std::pair<std::string, std::string>> capture_events_;

  std::atomic<std::uint64_t> audio_packets_{};
  std::atomic<std::uint64_t> processed_frames_{};
  std::atomic<std::uint64_t> kws_hits_{};
  std::atomic<std::uint64_t> candidates_{};
  std::atomic<std::uint64_t> rejections_{};
  std::atomic<std::uint64_t> audio_queue_drops_{};
  std::atomic<std::uint64_t> discontinuities_{};
  std::uint64_t next_telemetry_sample_{};
};

}  // namespace dvo
