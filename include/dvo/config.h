#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dvo {

struct AudioConfig {
  std::string microphone_device{"default"};
  std::string loopback_device{"default"};
  bool follow_default_device{true};
  std::uint32_t target_sample_rate{16000};
  std::uint32_t frame_ms{10};
  std::uint32_t queue_capacity_ms{2000};
};

struct PreprocessConfig {
  std::string implementation{"webrtc_aec3"};
  bool aec_enabled{true};
};

struct AecConfig {
  bool enabled{true};
  std::uint32_t processing_rate_hz{16000};
  bool high_pass_filter{true};
  bool noise_suppression{false};
  bool gain_control{false};
  bool request_raw_capture{true};
  bool request_post_volume_loopback{true};
  std::uint32_t alignment_wait_ms{30};
  std::uint32_t target_render_buffer_ms{60};
  std::uint32_t max_render_buffer_ms{500};
  std::uint32_t drift_window_ms{5000};
  std::int32_t max_drift_ppm{1000};
  std::uint32_t hard_resync_error_ms{80};
  std::int32_t delay_offset_ms{0};
  // -1 averages all microphone channels; otherwise selects a zero-based
  // channel before resampling/AEC. Channel 0 is the safe default for arrays.
  std::int32_t microphone_channel_index{0};
  bool auto_delay_enabled{true};
  std::uint32_t auto_delay_min_ms{0};
  std::uint32_t auto_delay_max_ms{250};
  std::uint32_t auto_delay_window_ms{1000};
  std::uint32_t auto_delay_update_ms{500};
  float auto_delay_min_correlation{0.35F};
  std::string missing_render_policy{"bypass_reset"};
  std::uint32_t stats_hz{1};
};

struct RingConfig { std::uint32_t duration_ms{20000}; };

struct KwsConfig {
  bool enabled{true};
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  std::filesystem::path joiner;
  std::filesystem::path tokens;
  std::filesystem::path lexicon;
  std::filesystem::path keywords;
  std::string provider{"cpu"};
  std::int32_t num_threads{1};
  std::int32_t max_active_paths{4};
  std::int32_t num_trailing_blanks{1};
  float boosting_score{1.0F};
  float threshold{0.25F};
};

struct VadConfig {
  bool enabled{true};
  std::filesystem::path model;
  std::string provider{"cpu"};
  std::int32_t num_threads{1};
  float threshold{0.25F};
  std::int32_t window_size{512};
  std::uint32_t min_speech_ms{150};
  std::uint32_t min_silence_ms{600};
  std::uint32_t max_speech_ms{18000};
};

struct SegmentationConfig {
  std::uint32_t wake_guard_ms{100};
  std::uint32_t pre_roll_ms{150};
  std::uint32_t post_roll_ms{900};
  std::uint32_t endpoint_silence_ms{900};
  std::uint32_t max_candidate_ms{18000};
  std::uint32_t min_command_speech_ms{250};
  std::uint32_t embedded_join_silence_ms{150};
  std::uint32_t assembly_queue_capacity{4};
};

struct ActivationConfig {
  bool enabled{true};
  std::uint32_t idle_timeout_ms{6000};
  std::uint32_t hard_limit_ms{20000};
};

struct AsrConfig {
  bool enabled{true};
  std::string implementation{"sherpa_online_paraformer"};
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  std::filesystem::path tokens;
  std::string provider{"cpu"};
  std::int32_t num_threads{1};
  std::uint32_t feed_chunk_ms{100};
  std::uint32_t max_active_streams{2};
  std::uint32_t queue_capacity{512};
  std::uint32_t max_pending_audio_ms{60000};
  bool emit_partials{true};
  bool exact_final_redecode{true};
};

struct CommandsConfig {
  bool enabled{true};
  std::uint32_t default_volume_step_percent{5};
  std::uint32_t max_spoken_volume_step_percent{20};
  std::uint32_t max_actions_per_utterance{8};
  std::uint32_t action_timeout_ms{2000};
  std::uint32_t queue_capacity{32};
  std::vector<std::string> play_phrases{"播放音乐", "打开音乐"};
  std::vector<std::string> pause_phrases{"暂停音乐"};
  std::vector<std::string> volume_up_phrases{"增加音量"};
  std::vector<std::string> volume_down_phrases{"降低音量"};
  std::vector<std::string> connectors{"然后再", "然后", "再", "后", "接着", "并且", "以及", "和", "还有"};
};

struct AnnouncementsConfig {
  bool enabled{true};
  std::string backend{"log"};
  std::uint32_t queue_capacity{32};
  bool barge_in{false};
  std::uint32_t tail_guard_ms{200};
};

struct WebConfig {
  bool enabled{true};
  std::string bind{"127.0.0.1"};
  std::uint16_t port{8765};
  std::uint32_t telemetry_hz{20};
  std::filesystem::path static_root{"web"};
};

struct RecordingConfig {
  std::filesystem::path session_root{"data/sessions"};
  std::uint32_t queue_capacity_ms{5000};
};

struct AppConfig {
  AudioConfig audio;
  PreprocessConfig preprocess;
  AecConfig aec;
  RingConfig ring;
  KwsConfig kws;
  VadConfig vad;
  SegmentationConfig segmentation;
  ActivationConfig activation;
  AsrConfig asr;
  CommandsConfig commands;
  AnnouncementsConfig announcements;
  WebConfig web;
  RecordingConfig recording;
  std::filesystem::path project_root;
  std::uint64_t revision{1};
};

struct ConfigValidation {
  std::vector<std::string> errors;
  [[nodiscard]] bool ok() const { return errors.empty(); }
};

class ConfigStore {
 public:
  ConfigStore(std::filesystem::path defaults, std::filesystem::path overrides,
              std::filesystem::path project_root);

  [[nodiscard]] AppConfig load() const;
  [[nodiscard]] ConfigValidation validate(const AppConfig& config) const;
  [[nodiscard]] nlohmann::json to_public_json(const AppConfig& config) const;
  [[nodiscard]] nlohmann::json to_manifest_json(const AppConfig& config) const;
  ConfigValidation apply_patch(AppConfig& config, const nlohmann::json& patch) const;
  void save_overrides(const AppConfig& config) const;

 private:
  std::filesystem::path defaults_;
  std::filesystem::path overrides_;
  std::filesystem::path project_root_;
};

}  // namespace dvo
