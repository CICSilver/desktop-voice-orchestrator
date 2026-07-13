#include "dvo/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <toml++/toml.hpp>

namespace dvo {
namespace {

template <typename T>
T read(const toml::table& table, std::string_view path, T fallback) {
  if (const auto value = table.at_path(path).template value<T>()) return *value;
  return fallback;
}

std::filesystem::path read_path(const toml::table& table, std::string_view path,
  const std::filesystem::path& fallback) {
  return std::filesystem::u8path(read<std::string>(table, path, fallback.string()));
}

void merge_table(toml::table& target, const toml::table& source) {
  for (const auto& [key, node] : source) {
    if (node.is_table()) {
      auto* destination = target.get_as<toml::table>(key);
      if (!destination) {
        target.insert_or_assign(key, *node.as_table());
      } else {
        merge_table(*destination, *node.as_table());
      }
    } else {
      target.insert_or_assign(key, node);
    }
  }
}

std::filesystem::path resolve(const std::filesystem::path& root,
                              const std::filesystem::path& value) {
  return value.is_absolute() ? value : root / value;
}

template <typename T>
void assign_if(const nlohmann::json& object, const char* key, T& value) {
  if (const auto it = object.find(key); it != object.end()) value = it->get<T>();
}

std::string toml_string(std::string_view value) {
  std::string result{"\""};
  for (const auto ch : value) {
    if (ch == '\\' || ch == '\"') result.push_back('\\');
    result.push_back(ch);
  }
  result.push_back('\"');
  return result;
}

}  // namespace

ConfigStore::ConfigStore(std::filesystem::path defaults, std::filesystem::path overrides,
                         std::filesystem::path project_root)
    : defaults_(std::move(defaults)), overrides_(std::move(overrides)),
      project_root_(std::move(project_root)) {}

AppConfig ConfigStore::load() const {
  toml::table table = toml::parse_file(defaults_.string());
  if (std::filesystem::exists(overrides_)) {
    auto overlay = toml::parse_file(overrides_.string());
    merge_table(table, overlay);
  }

  AppConfig c;
  c.project_root = project_root_;
  c.audio.microphone_device = read<std::string>(table, "audio.microphone_device", "default");
  c.audio.loopback_device = read<std::string>(table, "audio.loopback_device", "default");
  c.audio.follow_default_device = read<bool>(table, "audio.follow_default_device", true);
  c.audio.target_sample_rate = read<std::int64_t>(table, "audio.target_sample_rate", 16000);
  c.audio.frame_ms = read<std::int64_t>(table, "audio.frame_ms", 10);
  c.audio.queue_capacity_ms = read<std::int64_t>(table, "audio.queue_capacity_ms", 2000);
  c.preprocess.implementation = read<std::string>(table, "preprocess.implementation", "bypass");
  c.preprocess.aec_enabled = read<bool>(table, "preprocess.aec_enabled", false);
  c.ring.duration_ms = read<std::int64_t>(table, "ring.duration_ms", 20000);

  c.kws.enabled = read<bool>(table, "kws.enabled", true);
  c.kws.encoder = resolve(project_root_, read_path(table, "kws.encoder", {}));
  c.kws.decoder = resolve(project_root_, read_path(table, "kws.decoder", {}));
  c.kws.joiner = resolve(project_root_, read_path(table, "kws.joiner", {}));
  c.kws.tokens = resolve(project_root_, read_path(table, "kws.tokens", {}));
  c.kws.lexicon = resolve(project_root_, read_path(table, "kws.lexicon", {}));
  c.kws.keywords = resolve(project_root_, read_path(table, "kws.keywords", {}));
  c.kws.provider = read<std::string>(table, "kws.provider", "cpu");
  c.kws.num_threads = read<std::int64_t>(table, "kws.num_threads", 1);
  c.kws.max_active_paths = read<std::int64_t>(table, "kws.max_active_paths", 4);
  c.kws.num_trailing_blanks = read<std::int64_t>(table, "kws.num_trailing_blanks", 1);
  c.kws.boosting_score = read<double>(table, "kws.boosting_score", 1.0);
  c.kws.threshold = read<double>(table, "kws.threshold", 0.25);

  c.vad.enabled = read<bool>(table, "vad.enabled", true);
  c.vad.model = resolve(project_root_, read_path(table, "vad.model", {}));
  c.vad.provider = read<std::string>(table, "vad.provider", "cpu");
  c.vad.num_threads = read<std::int64_t>(table, "vad.num_threads", 1);
  c.vad.threshold = read<double>(table, "vad.threshold", 0.25);
  c.vad.window_size = read<std::int64_t>(table, "vad.window_size", 512);
  c.vad.min_speech_ms = read<std::int64_t>(table, "vad.min_speech_ms", 150);
  c.vad.min_silence_ms = read<std::int64_t>(table, "vad.min_silence_ms", 600);
  c.vad.max_speech_ms = read<std::int64_t>(table, "vad.max_speech_ms", 18000);

  c.segmentation.wake_guard_ms = read<std::int64_t>(table, "segmentation.wake_guard_ms", 100);
  c.segmentation.pre_roll_ms = read<std::int64_t>(table, "segmentation.pre_roll_ms", 150);
  c.segmentation.post_roll_ms = read<std::int64_t>(table, "segmentation.post_roll_ms", 150);
  c.segmentation.endpoint_silence_ms = read<std::int64_t>(table, "segmentation.endpoint_silence_ms", 900);
  c.segmentation.max_candidate_ms = read<std::int64_t>(table, "segmentation.max_candidate_ms", 18000);
  c.segmentation.min_command_speech_ms = read<std::int64_t>(table, "segmentation.min_command_speech_ms", 250);
  c.segmentation.embedded_join_silence_ms = read<std::int64_t>(table, "segmentation.embedded_join_silence_ms", 150);

  c.web.enabled = read<bool>(table, "web.enabled", true);
  c.web.bind = read<std::string>(table, "web.bind", "127.0.0.1");
  c.web.port = read<std::int64_t>(table, "web.port", 8765);
  c.web.telemetry_hz = read<std::int64_t>(table, "web.telemetry_hz", 20);
  c.web.static_root = resolve(project_root_, read_path(table, "web.static_root", "web"));
  c.recording.session_root = resolve(project_root_, read_path(table, "recording.session_root", "data/sessions"));
  c.recording.queue_capacity_ms = read<std::int64_t>(table, "recording.queue_capacity_ms", 5000);

  const auto result = validate(c);
  if (!result.ok()) {
    std::ostringstream message;
    message << "invalid configuration:";
    for (const auto& error : result.errors) message << "\n - " << error;
    throw std::runtime_error(message.str());
  }
  return c;
}

ConfigValidation ConfigStore::validate(const AppConfig& c) const {
  ConfigValidation v;
  const auto range = [&v](bool valid, std::string message) {
    if (!valid) v.errors.push_back(std::move(message));
  };
  range(c.audio.target_sample_rate == 16000, "audio.target_sample_rate must be 16000 in phase 1");
  range(c.audio.frame_ms == 10, "audio.frame_ms must be 10 in phase 1");
  range(c.audio.queue_capacity_ms >= 500 && c.audio.queue_capacity_ms <= 10000,
        "audio.queue_capacity_ms must be in [500, 10000]");
  range(c.preprocess.implementation == "bypass", "preprocess.implementation must be 'bypass'");
  range(!c.preprocess.aec_enabled, "preprocess.aec_enabled must remain false in phase 1");
  range(c.ring.duration_ms >= 1000 && c.ring.duration_ms <= 120000,
        "ring.duration_ms must be in [1000, 120000]");
  range(c.kws.threshold >= 0.0F && c.kws.threshold <= 1.0F, "kws.threshold must be in [0, 1]");
  range(c.kws.boosting_score >= 0.0F && c.kws.boosting_score <= 20.0F,
        "kws.boosting_score must be in [0, 20]");
  range(c.kws.num_threads >= 1 && c.kws.num_threads <= 16, "kws.num_threads must be in [1, 16]");
  range(c.vad.threshold >= 0.0F && c.vad.threshold <= 1.0F, "vad.threshold must be in [0, 1]");
  range(c.vad.window_size == 512, "vad.window_size must be 512 for the pinned 16 kHz model");
  range(c.segmentation.max_candidate_ms <= c.ring.duration_ms,
        "segmentation.max_candidate_ms must not exceed ring.duration_ms");
  range(c.segmentation.endpoint_silence_ms >= c.vad.min_silence_ms,
        "segmentation.endpoint_silence_ms must be >= vad.min_silence_ms");
  range(c.web.bind == "127.0.0.1" || c.web.bind == "localhost",
        "web.bind must be loopback-only");
  range(c.web.port > 0, "web.port must be non-zero");
  range(c.web.telemetry_hz >= 1 && c.web.telemetry_hz <= 60, "web.telemetry_hz must be in [1, 60]");
  return v;
}

nlohmann::json ConfigStore::to_public_json(const AppConfig& c) const {
  return {
      {"revision", c.revision},
      {"audio", {{"microphone_device", c.audio.microphone_device},
                 {"loopback_device", c.audio.loopback_device}}},
      {"kws", {{"threshold", c.kws.threshold}, {"boosting_score", c.kws.boosting_score},
               {"max_active_paths", c.kws.max_active_paths}, {"num_threads", c.kws.num_threads},
               {"provider", c.kws.provider}, {"encoder", c.kws.encoder.string()},
               {"decoder", c.kws.decoder.string()}, {"joiner", c.kws.joiner.string()},
               {"tokens", c.kws.tokens.string()}, {"keywords", c.kws.keywords.string()}}},
      {"vad", {{"threshold", c.vad.threshold}, {"min_speech_ms", c.vad.min_speech_ms},
               {"min_silence_ms", c.vad.min_silence_ms}, {"max_speech_ms", c.vad.max_speech_ms},
               {"provider", c.vad.provider}, {"num_threads", c.vad.num_threads},
               {"model", c.vad.model.string()}}},
      {"segmentation", {{"wake_guard_ms", c.segmentation.wake_guard_ms},
                         {"pre_roll_ms", c.segmentation.pre_roll_ms},
                         {"post_roll_ms", c.segmentation.post_roll_ms},
                         {"endpoint_silence_ms", c.segmentation.endpoint_silence_ms},
                         {"max_candidate_ms", c.segmentation.max_candidate_ms},
                         {"min_command_speech_ms", c.segmentation.min_command_speech_ms},
                         {"embedded_join_silence_ms", c.segmentation.embedded_join_silence_ms}}},
      {"web", {{"telemetry_hz", c.web.telemetry_hz}}},
      {"cold", {{"microphone_device", c.audio.microphone_device},
                 {"loopback_device", c.audio.loopback_device},
                 {"kws_provider", c.kws.provider}, {"kws_num_threads", c.kws.num_threads},
                 {"kws_encoder", c.kws.encoder.string()}, {"kws_decoder", c.kws.decoder.string()},
                 {"kws_joiner", c.kws.joiner.string()}, {"kws_tokens", c.kws.tokens.string()},
                 {"kws_keywords", c.kws.keywords.string()}, {"vad_provider", c.vad.provider},
                 {"vad_num_threads", c.vad.num_threads}, {"vad_model", c.vad.model.string()}}}
  };
}

nlohmann::json ConfigStore::to_manifest_json(const AppConfig& c) const {
  return {
      {"revision", c.revision},
      {"audio", {{"microphone_device", c.audio.microphone_device},
                 {"loopback_device", c.audio.loopback_device},
                 {"follow_default_device", c.audio.follow_default_device},
                 {"target_sample_rate", c.audio.target_sample_rate}, {"frame_ms", c.audio.frame_ms},
                 {"queue_capacity_ms", c.audio.queue_capacity_ms}}},
      {"preprocess", {{"implementation", c.preprocess.implementation},
                      {"aec_enabled", c.preprocess.aec_enabled}}},
      {"ring", {{"duration_ms", c.ring.duration_ms}}},
      {"kws", {{"enabled", c.kws.enabled}, {"encoder", c.kws.encoder.string()},
               {"decoder", c.kws.decoder.string()}, {"joiner", c.kws.joiner.string()},
               {"tokens", c.kws.tokens.string()}, {"lexicon", c.kws.lexicon.string()},
               {"keywords", c.kws.keywords.string()}, {"provider", c.kws.provider},
               {"num_threads", c.kws.num_threads}, {"max_active_paths", c.kws.max_active_paths},
               {"num_trailing_blanks", c.kws.num_trailing_blanks},
               {"boosting_score", c.kws.boosting_score}, {"threshold", c.kws.threshold}}},
      {"vad", {{"enabled", c.vad.enabled}, {"model", c.vad.model.string()},
               {"provider", c.vad.provider}, {"num_threads", c.vad.num_threads},
               {"threshold", c.vad.threshold}, {"window_size", c.vad.window_size},
               {"min_speech_ms", c.vad.min_speech_ms}, {"min_silence_ms", c.vad.min_silence_ms},
               {"max_speech_ms", c.vad.max_speech_ms}}},
      {"segmentation", {{"wake_guard_ms", c.segmentation.wake_guard_ms},
                         {"pre_roll_ms", c.segmentation.pre_roll_ms},
                         {"post_roll_ms", c.segmentation.post_roll_ms},
                         {"endpoint_silence_ms", c.segmentation.endpoint_silence_ms},
                         {"max_candidate_ms", c.segmentation.max_candidate_ms},
                         {"min_command_speech_ms", c.segmentation.min_command_speech_ms},
                         {"embedded_join_silence_ms", c.segmentation.embedded_join_silence_ms}}},
      {"web", {{"enabled", c.web.enabled}, {"bind", c.web.bind}, {"port", c.web.port},
               {"telemetry_hz", c.web.telemetry_hz}, {"static_root", c.web.static_root.string()}}},
      {"recording", {{"session_root", c.recording.session_root.string()},
                     {"queue_capacity_ms", c.recording.queue_capacity_ms}}}
  };
}

ConfigValidation ConfigStore::apply_patch(AppConfig& config, const nlohmann::json& patch) const {
  auto candidate = config;
  try {
    if (const auto it = patch.find("audio"); it != patch.end()) {
      assign_if(*it, "microphone_device", candidate.audio.microphone_device);
      assign_if(*it, "loopback_device", candidate.audio.loopback_device);
    }
    if (const auto it = patch.find("kws"); it != patch.end()) {
      assign_if(*it, "threshold", candidate.kws.threshold);
      assign_if(*it, "boosting_score", candidate.kws.boosting_score);
      assign_if(*it, "provider", candidate.kws.provider);
      assign_if(*it, "num_threads", candidate.kws.num_threads);
      const auto path = [&](const char* key, std::filesystem::path& output) {
        if (const auto value = it->find(key); value != it->end()) {
          output = resolve(project_root_, std::filesystem::u8path(value->get<std::string>()));
        }
      };
      path("encoder", candidate.kws.encoder);
      path("decoder", candidate.kws.decoder);
      path("joiner", candidate.kws.joiner);
      path("tokens", candidate.kws.tokens);
      path("keywords", candidate.kws.keywords);
    }
    if (const auto it = patch.find("vad"); it != patch.end()) {
      assign_if(*it, "threshold", candidate.vad.threshold);
      assign_if(*it, "min_speech_ms", candidate.vad.min_speech_ms);
      assign_if(*it, "min_silence_ms", candidate.vad.min_silence_ms);
      assign_if(*it, "max_speech_ms", candidate.vad.max_speech_ms);
      assign_if(*it, "provider", candidate.vad.provider);
      assign_if(*it, "num_threads", candidate.vad.num_threads);
      if (const auto value = it->find("model"); value != it->end()) {
        candidate.vad.model = resolve(project_root_, std::filesystem::u8path(value->get<std::string>()));
      }
    }
    if (const auto it = patch.find("segmentation"); it != patch.end()) {
      assign_if(*it, "wake_guard_ms", candidate.segmentation.wake_guard_ms);
      assign_if(*it, "pre_roll_ms", candidate.segmentation.pre_roll_ms);
      assign_if(*it, "post_roll_ms", candidate.segmentation.post_roll_ms);
      assign_if(*it, "endpoint_silence_ms", candidate.segmentation.endpoint_silence_ms);
      assign_if(*it, "max_candidate_ms", candidate.segmentation.max_candidate_ms);
      assign_if(*it, "min_command_speech_ms", candidate.segmentation.min_command_speech_ms);
      assign_if(*it, "embedded_join_silence_ms", candidate.segmentation.embedded_join_silence_ms);
    }
    if (const auto it = patch.find("web"); it != patch.end()) {
      assign_if(*it, "telemetry_hz", candidate.web.telemetry_hz);
    }
  } catch (const std::exception& e) {
    return ConfigValidation{{std::string("invalid patch value: ") + e.what()}};
  }
  auto validation = validate(candidate);
  if (validation.ok()) {
    candidate.revision = config.revision + 1;
    config = std::move(candidate);
  }
  return validation;
}

void ConfigStore::save_overrides(const AppConfig& c) const {
  std::filesystem::create_directories(overrides_.parent_path());
  const auto temporary = overrides_.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create config override file");
  out << "# 由调试界面生成。这里只保存可调参数；完整说明和默认值请查看 default.toml。\n\n";
  out << "[audio]\nmicrophone_device = " << toml_string(c.audio.microphone_device)
      << "\nloopback_device = " << toml_string(c.audio.loopback_device) << "\n\n";
  out << "[kws]\nencoder = " << toml_string(c.kws.encoder.generic_string())
      << "\ndecoder = " << toml_string(c.kws.decoder.generic_string())
      << "\njoiner = " << toml_string(c.kws.joiner.generic_string())
      << "\ntokens = " << toml_string(c.kws.tokens.generic_string())
      << "\nkeywords = " << toml_string(c.kws.keywords.generic_string())
      << "\nprovider = " << toml_string(c.kws.provider) << "\nnum_threads = " << c.kws.num_threads
      << "\nthreshold = " << c.kws.threshold << "\nboosting_score = " << c.kws.boosting_score << "\n\n";
  out << "[vad]\nmodel = " << toml_string(c.vad.model.generic_string())
      << "\nprovider = " << toml_string(c.vad.provider) << "\nnum_threads = " << c.vad.num_threads
      << "\nthreshold = " << c.vad.threshold << "\nmin_speech_ms = " << c.vad.min_speech_ms
      << "\nmin_silence_ms = " << c.vad.min_silence_ms << "\nmax_speech_ms = " << c.vad.max_speech_ms << "\n\n";
  out << "[segmentation]\nwake_guard_ms = " << c.segmentation.wake_guard_ms
      << "\npre_roll_ms = " << c.segmentation.pre_roll_ms << "\npost_roll_ms = " << c.segmentation.post_roll_ms
      << "\nendpoint_silence_ms = " << c.segmentation.endpoint_silence_ms
      << "\nmax_candidate_ms = " << c.segmentation.max_candidate_ms
      << "\nmin_command_speech_ms = " << c.segmentation.min_command_speech_ms
      << "\nembedded_join_silence_ms = " << c.segmentation.embedded_join_silence_ms << "\n\n";
  out << "[web]\ntelemetry_hz = " << c.web.telemetry_hz << "\n";
  out.close();
  if (!out) throw std::runtime_error("failed while writing config override file");
  std::error_code error;
  std::filesystem::remove(overrides_, error);
  std::filesystem::rename(temporary, overrides_);
}

}  // namespace dvo
