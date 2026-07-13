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
  std::string implementation{"bypass"};
  bool aec_enabled{false};
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
  std::uint32_t post_roll_ms{150};
  std::uint32_t endpoint_silence_ms{900};
  std::uint32_t max_candidate_ms{18000};
  std::uint32_t min_command_speech_ms{250};
  std::uint32_t embedded_join_silence_ms{150};
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
  RingConfig ring;
  KwsConfig kws;
  VadConfig vad;
  SegmentationConfig segmentation;
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
