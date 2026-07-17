#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/action_executor.h"
#include "dvo/announcement.h"
#include "dvo/command_parser.h"
#include "dvo/config.h"
#include "dvo/debug_server.h"
#include "dvo/inference.h"
#include "dvo/preprocessor.h"
#include "dvo/recorder.h"
#include "dvo/replay.h"
#include "dvo/ring_buffer.h"
#include "dvo/segmenter.h"
#include "dvo/spsc_queue.h"
#include "dvo/streaming_recognizer.h"
#include "dvo/utterance_lifecycle.h"
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
  void reset_pipeline(bool discontinuity, bool reset_preprocessor = true);
  void start_processing();
  void stop_processing();
  void start_captures();
  void stop_captures();
  void reset_audio_queues();
  void enqueue_replay_packet(AudioPacket packet);
  void enqueue_packet(AudioPacket packet);
  [[nodiscard]] bool enqueue_capture_packet(RawCapturedPacket packet);
  void processing_loop(std::stop_token stop);
  void process_microphone(AudioPacket packet);
  void drain_preprocessor_frames(bool pipeline_reset_for_packet);
  void report_audio_queue_boundary(const char* trigger_stream,
                                   PreprocessResetReason reason);
  void process_frame(const NormalizedFrame& frame, const SignalSummary& microphone,
                     const SignalSummary& loopback);
  [[nodiscard]] bool drain_audio_assemblies();
  void complete_backfill(BackfillAssemblyResult result);
  void complete_candidate(CandidateAssemblyResult result);
  void begin_recognition(const UtteranceStart& start);
  void feed_recognizers(const NormalizedFrame& frame);
  void finalize_recognition(std::shared_ptr<const UtteranceCandidate> candidate);
  void cancel_recognition(const std::string& utterance_id, std::string reason);
  void cancel_recognition_generation(std::string reason);
  void erase_command_snapshot(const std::string& utterance_id,
                              std::uint64_t generation);
  void emit_utterance_lifecycle(UtteranceLifecycleEvent event);
  void handle_recognition_result(RecognitionResult result);
  void handle_action_result(const ActionResult& result);
  void handle_plan_started(const CommandPlan& plan);
  void handle_announcement_result(const AnnouncementResult& result);
  void emit_activation_events(std::vector<ActivationTransition> events,
                              std::string source = {});
  void emit_activation_state(std::string source = {});
  void emit_activation_state_if_changed(std::uint64_t timestamp_sample,
                                        const std::string& source);
  void abandon_followup(UtteranceOrigin origin,
                        const std::string& activation_id,
                        std::uint32_t turn_index,
                        std::uint64_t at_sample,
                        std::string reason);
  [[nodiscard]] bool announcement_capture_blocked(
      std::uint64_t at_sample) const;
  void stop_async_services();
  void emit_event(std::string type, nlohmann::json payload,
                  std::uint64_t timestamp_sample = 0, std::string source = "live");
  void queue_capture_event(std::string type, std::string message);
  void drain_capture_events();
  [[nodiscard]] nlohmann::json handle_command(const nlohmann::json& command);
  [[nodiscard]] nlohmann::json session_list() const;
  [[nodiscard]] nlohmann::json metrics_json() const;
  [[nodiscard]] PreprocessDiagnostics preprocess_diagnostics_snapshot() const;
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
  std::unique_ptr<CandidateAssembler> candidate_assembler_;
  std::unique_ptr<IStreamingRecognizer> asr_;
  // Parser instances are immutable. In-flight utterances retain shared snapshots
  // across command-grammar hot updates.
  std::shared_ptr<const CommandParser> command_parser_;
  std::unique_ptr<AnnouncementDispatcher> announcement_dispatcher_;
  std::unique_ptr<OrderedActionExecutor> action_executor_;
  mutable std::mutex announcement_state_mutex_;
  nlohmann::json latest_live_announcement_{nullptr};
  nlohmann::json latest_replay_announcement_{nullptr};
  mutable std::mutex pipeline_mutex_;
  // HTTP commands mutate replay, recording, and configuration state. They are
  // intentionally serialized independently from recognition snapshots.
  std::mutex control_mutex_;
  std::mutex command_mutex_;

  struct UtteranceCommandSnapshot {
    std::shared_ptr<const CommandParser> parser;
    std::uint64_t config_revision{};
    bool commands_enabled{};
    UtteranceOrigin origin{UtteranceOrigin::keyword};
    std::string activation_id;
    std::uint32_t turn_index{};
    std::uint64_t trigger_sample{};
  };
  std::uint64_t command_config_revision_{};
  bool commands_enabled_{};
  std::unordered_map<std::string, UtteranceCommandSnapshot> command_snapshots_;
  std::unordered_map<std::string, UtteranceCommandSnapshot> completed_command_snapshots_;
  std::deque<std::string> completed_command_snapshot_order_;

  struct ActiveRecognitionFeed {
    std::uint64_t generation{};
    std::uint64_t stream_start_sample{};
    UtteranceOrigin origin{UtteranceOrigin::keyword};
    std::string activation_id;
    std::uint32_t turn_index{};
    std::uint64_t trigger_sample{};
    std::uint64_t pending_first_sample{};
    std::size_t chunk_samples{kFrameSamples};
    std::vector<float> pending;
  };
  struct PendingRecognitionStart {
    std::uint64_t generation{};
    std::uint64_t stream_start_sample{};
    UtteranceOrigin origin{UtteranceOrigin::keyword};
    std::string activation_id;
    std::uint32_t turn_index{};
    std::uint64_t trigger_sample{};
    std::size_t chunk_samples{kFrameSamples};
    std::uint64_t pending_first_sample{};
    std::vector<float> pending;
    std::uint64_t command_config_revision{};
    std::size_t backfill_span_count{};
  };
  std::unordered_map<std::string, ActiveRecognitionFeed> active_recognizers_;
  std::unordered_map<std::string, PendingRecognitionStart> pending_recognition_starts_;
  std::atomic<std::uint64_t> recognition_generation_{1};

  using AudioIngressPacket = std::variant<AudioPacket, RawCapturedPacket>;
  std::unique_ptr<SpscQueue<AudioIngressPacket>> microphone_queue_;
  std::unique_ptr<SpscQueue<AudioIngressPacket>> loopback_queue_;
  SignalSummary latest_microphone_summary_{};
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
  std::atomic<bool> announcement_playback_active_{};
  std::atomic<std::uint64_t> announcement_guard_until_sample_{};
  std::atomic<bool> announcement_kws_reset_pending_{};
  // Protected by pipeline_mutex_. Backend wall time is diagnostic-only; the
  // activation extension is derived from this 16 kHz timeline position.
  std::uint64_t announcement_playback_start_sample_{};
  bool vad_speech_active_{};
  bool suppress_current_vad_interval_{};
  ActivationSnapshot last_published_activation_state_{};
  std::string runtime_session_id_;
  std::mutex replay_callback_mutex_;

  mutable std::mutex capture_events_mutex_;
  std::deque<std::pair<std::string, std::string>> capture_events_;
  nlohmann::json capture_capabilities_{nlohmann::json::object()};

  std::atomic<std::uint64_t> audio_packets_{};
  std::atomic<std::uint64_t> processed_frames_{};
  std::atomic<std::uint64_t> kws_hits_{};
  std::atomic<std::uint64_t> candidates_{};
  std::atomic<std::uint64_t> rejections_{};
  std::atomic<std::uint64_t> audio_queue_drops_{};
  std::atomic<std::uint64_t> audio_reset_backlog_drops_{};
  std::atomic<std::uint64_t> discontinuities_{};
  std::atomic<std::uint64_t> asr_partials_{};
  std::atomic<std::uint64_t> asr_finals_{};
  std::atomic<std::uint64_t> asr_dropped_{};
  std::atomic<std::uint64_t> assembly_dropped_{};
  std::atomic<std::uint64_t> command_plans_{};
  std::atomic<std::uint64_t> command_rejections_{};
  std::atomic<std::uint64_t> activations_started_{};
  std::atomic<std::uint64_t> activations_refreshed_{};
  std::atomic<std::uint64_t> activations_expired_{};
  std::atomic<std::uint64_t> activations_cancelled_{};
  std::atomic<std::uint64_t> followup_turns_{};
  std::atomic<std::uint64_t> announcements_requested_{};
  std::atomic<std::uint64_t> announcements_dropped_{};
  std::atomic<std::uint64_t> announcements_failed_{};
  std::atomic<std::uint64_t> actions_succeeded_{};
  std::atomic<std::uint64_t> actions_failed_{};
  StreamingRecognizerState last_published_asr_state_{
      StreamingRecognizerState::unavailable};
  std::uint64_t next_telemetry_sample_{};
};

}  // namespace dvo
