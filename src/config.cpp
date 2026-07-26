#include "dvo/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <toml++/toml.hpp>

#include "dvo/command_parser.h"

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

std::vector<std::string> read_strings(const toml::table& table, std::string_view path,
                                      std::vector<std::string> fallback) {
  const auto* values = table.at_path(path).as_array();
  if (!values) return fallback;
  std::vector<std::string> result;
  result.reserve(values->size());
  for (const auto& value : *values) {
    if (const auto text = value.value<std::string>()) result.push_back(*text);
  }
  return result;
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

std::string toml_array(const std::vector<std::string>& values) {
  std::ostringstream result;
  result << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) result << ", ";
    result << toml_string(values[i]);
  }
  result << ']';
  return result.str();
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
  c.preprocess.implementation = read<std::string>(table, "preprocess.implementation", "webrtc_aec3");
  c.preprocess.aec_enabled = read<bool>(table, "preprocess.aec_enabled", true);
  c.aec.enabled = read<bool>(table, "aec.enabled", true);
  c.aec.processing_rate_hz = read<std::int64_t>(table, "aec.processing_rate_hz", 16000);
  c.aec.high_pass_filter = read<bool>(table, "aec.high_pass_filter", true);
  c.aec.noise_suppression = read<bool>(table, "aec.noise_suppression", false);
  c.aec.gain_control = read<bool>(table, "aec.gain_control", false);
  c.aec.request_raw_capture = read<bool>(table, "aec.request_raw_capture", true);
  c.aec.request_post_volume_loopback = read<bool>(table, "aec.request_post_volume_loopback", true);
  c.aec.alignment_wait_ms = read<std::int64_t>(table, "aec.alignment_wait_ms", 30);
  c.aec.target_render_buffer_ms = read<std::int64_t>(table, "aec.target_render_buffer_ms", 60);
  c.aec.max_render_buffer_ms = read<std::int64_t>(table, "aec.max_render_buffer_ms", 500);
  c.aec.drift_window_ms = read<std::int64_t>(table, "aec.drift_window_ms", 5000);
  c.aec.max_drift_ppm = read<std::int64_t>(table, "aec.max_drift_ppm", 1000);
  c.aec.hard_resync_error_ms = read<std::int64_t>(table, "aec.hard_resync_error_ms", 80);
  c.aec.delay_offset_ms = read<std::int64_t>(table, "aec.delay_offset_ms", 0);
  c.aec.microphone_channel_index =
      read<std::int64_t>(table, "aec.microphone_channel_index", 0);
  c.aec.auto_delay_enabled = read<bool>(table, "aec.auto_delay_enabled", true);
  c.aec.auto_delay_min_ms = read<std::int64_t>(table, "aec.auto_delay_min_ms", 0);
  c.aec.auto_delay_max_ms = read<std::int64_t>(table, "aec.auto_delay_max_ms", 250);
  c.aec.auto_delay_window_ms =
      read<std::int64_t>(table, "aec.auto_delay_window_ms", 1000);
  c.aec.auto_delay_update_ms =
      read<std::int64_t>(table, "aec.auto_delay_update_ms", 500);
  c.aec.auto_delay_min_correlation =
      read<double>(table, "aec.auto_delay_min_correlation", 0.35);
  c.aec.missing_render_policy = read<std::string>(table, "aec.missing_render_policy", "bypass_reset");
  c.aec.stats_hz = read<std::int64_t>(table, "aec.stats_hz", 1);
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
  c.segmentation.post_roll_ms = read<std::int64_t>(table, "segmentation.post_roll_ms", 900);
  c.segmentation.endpoint_silence_ms = read<std::int64_t>(table, "segmentation.endpoint_silence_ms", 900);
  c.segmentation.max_candidate_ms = read<std::int64_t>(table, "segmentation.max_candidate_ms", 18000);
  c.segmentation.min_command_speech_ms = read<std::int64_t>(table, "segmentation.min_command_speech_ms", 250);
  c.segmentation.embedded_join_silence_ms = read<std::int64_t>(table, "segmentation.embedded_join_silence_ms", 150);
  c.segmentation.assembly_queue_capacity = read<std::int64_t>(table, "segmentation.assembly_queue_capacity", 4);

  c.activation.enabled = read<bool>(table, "activation.enabled", true);
  c.activation.idle_timeout_ms =
      read<std::int64_t>(table, "activation.idle_timeout_ms", 6000);
  c.activation.hard_limit_ms =
      read<std::int64_t>(table, "activation.hard_limit_ms", 20000);

  c.asr.enabled = read<bool>(table, "asr.enabled", true);
  c.asr.implementation = read<std::string>(table, "asr.implementation", "sherpa_online_paraformer");
  c.asr.encoder = resolve(project_root_, read_path(table, "asr.encoder", {}));
  c.asr.decoder = resolve(project_root_, read_path(table, "asr.decoder", {}));
  c.asr.tokens = resolve(project_root_, read_path(table, "asr.tokens", {}));
  c.asr.provider = read<std::string>(table, "asr.provider", "cpu");
  c.asr.num_threads = read<std::int64_t>(table, "asr.num_threads", 1);
  c.asr.feed_chunk_ms = read<std::int64_t>(table, "asr.feed_chunk_ms", 100);
  c.asr.max_active_streams = read<std::int64_t>(table, "asr.max_active_streams", 2);
  c.asr.queue_capacity = read<std::int64_t>(table, "asr.queue_capacity", 512);
  c.asr.max_pending_audio_ms = read<std::int64_t>(table, "asr.max_pending_audio_ms", 60000);
  c.asr.emit_partials = read<bool>(table, "asr.emit_partials", true);
  c.asr.exact_final_redecode = read<bool>(table, "asr.exact_final_redecode", true);

  c.commands.enabled = read<bool>(table, "commands.enabled", true);
  c.commands.default_volume_step_percent =
      read<std::int64_t>(table, "commands.default_volume_step_percent", 5);
  c.commands.max_spoken_volume_step_percent =
      read<std::int64_t>(table, "commands.max_spoken_volume_step_percent", 20);
  c.commands.max_actions_per_utterance =
      read<std::int64_t>(table, "commands.max_actions_per_utterance", 8);
  c.commands.action_timeout_ms = read<std::int64_t>(table, "commands.action_timeout_ms", 2000);
  c.commands.queue_capacity = read<std::int64_t>(table, "commands.queue_capacity", 32);
  c.commands.play_phrases = read_strings(table, "commands.phrases.play", {"播放音乐"});
  c.commands.pause_phrases = read_strings(table, "commands.phrases.pause", {"暂停音乐"});
  c.commands.volume_up_phrases = read_strings(table, "commands.phrases.volume_up", {"增加音量"});
  c.commands.volume_down_phrases = read_strings(table, "commands.phrases.volume_down", {"降低音量"});
  c.commands.connectors = read_strings(
      table, "commands.connectors",
      {"然后再", "然后", "再", "后", "接着", "并且", "以及", "和", "还有"});

  c.announcements.enabled = read<bool>(table, "announcements.enabled", true);
  c.announcements.backend = read<std::string>(table, "announcements.backend", "log");
  c.announcements.queue_capacity =
      read<std::int64_t>(table, "announcements.queue_capacity", 32);
  c.announcements.barge_in = read<bool>(table, "announcements.barge_in", false);
  c.announcements.tail_guard_ms =
      read<std::int64_t>(table, "announcements.tail_guard_ms", 200);

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
  range(c.audio.target_sample_rate == 16000, "audio.target_sample_rate must be 16000");
  range(c.audio.frame_ms == 10, "audio.frame_ms must be 10");
  range(c.audio.queue_capacity_ms >= 500 && c.audio.queue_capacity_ms <= 10000,
        "audio.queue_capacity_ms must be in [500, 10000]");
  range(c.preprocess.implementation == "bypass" || c.preprocess.implementation == "webrtc_aec3",
        "preprocess.implementation must be 'bypass' or 'webrtc_aec3'");
  range(c.aec.processing_rate_hz == 16000, "aec.processing_rate_hz must be 16000 in phase 2");
  range(c.aec.alignment_wait_ms <= 100, "aec.alignment_wait_ms must be in [0, 100]");
  range(c.aec.target_render_buffer_ms >= 10 &&
            c.aec.target_render_buffer_ms <= c.aec.max_render_buffer_ms,
        "aec.target_render_buffer_ms must be in [10, max_render_buffer_ms]");
  range(c.aec.max_render_buffer_ms >= 100 && c.aec.max_render_buffer_ms <= 2000,
        "aec.max_render_buffer_ms must be in [100, 2000]");
  range(c.aec.drift_window_ms >= 1000 && c.aec.drift_window_ms <= 60000,
        "aec.drift_window_ms must be in [1000, 60000]");
  range(c.aec.max_drift_ppm >= 0 && c.aec.max_drift_ppm <= 5000,
        "aec.max_drift_ppm must be in [0, 5000]");
  range(c.aec.hard_resync_error_ms >= 10 && c.aec.hard_resync_error_ms <= 1000,
        "aec.hard_resync_error_ms must be in [10, 1000]");
  range(c.aec.delay_offset_ms >= -500 && c.aec.delay_offset_ms <= 500,
        "aec.delay_offset_ms must be in [-500, 500]");
  range(c.aec.microphone_channel_index >= -1 &&
            c.aec.microphone_channel_index <= 31,
        "aec.microphone_channel_index must be in [-1, 31]");
  range(c.aec.auto_delay_min_ms <= c.aec.auto_delay_max_ms &&
            c.aec.auto_delay_max_ms <= 500,
        "aec auto delay range must satisfy 0 <= min <= max <= 500 ms");
  range(c.aec.auto_delay_window_ms >= 500 &&
            c.aec.auto_delay_window_ms <= 5000,
        "aec.auto_delay_window_ms must be in [500, 5000]");
  range(c.aec.auto_delay_update_ms >= 100 &&
            c.aec.auto_delay_update_ms <= c.aec.auto_delay_window_ms,
        "aec.auto_delay_update_ms must be in [100, auto_delay_window_ms]");
  range(c.aec.auto_delay_min_correlation >= 0.10F &&
            c.aec.auto_delay_min_correlation <= 0.95F,
        "aec.auto_delay_min_correlation must be in [0.10, 0.95]");
  range(c.aec.missing_render_policy == "bypass_reset",
        "aec.missing_render_policy must be 'bypass_reset'");
  range(c.aec.stats_hz >= 1 && c.aec.stats_hz <= 20, "aec.stats_hz must be in [1, 20]");
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
  range(c.segmentation.assembly_queue_capacity >= 1 &&
            c.segmentation.assembly_queue_capacity <= 64,
        "segmentation.assembly_queue_capacity must be in [1, 64]");
  range(c.activation.idle_timeout_ms >= 1000 &&
            c.activation.idle_timeout_ms <= 10000,
        "activation.idle_timeout_ms must be in [1000, 10000]");
  range(c.activation.hard_limit_ms >= 10000 &&
            c.activation.hard_limit_ms <= 60000,
        "activation.hard_limit_ms must be in [10000, 60000]");
  range(c.activation.hard_limit_ms >= c.activation.idle_timeout_ms,
        "activation.hard_limit_ms must be >= activation.idle_timeout_ms");
  range(c.asr.implementation == "sherpa_online_paraformer",
        "asr.implementation must be 'sherpa_online_paraformer'");
  range(c.asr.num_threads >= 1 && c.asr.num_threads <= 16, "asr.num_threads must be in [1, 16]");
  range(c.asr.feed_chunk_ms >= 10 && c.asr.feed_chunk_ms <= 1000 && c.asr.feed_chunk_ms % 10 == 0,
        "asr.feed_chunk_ms must be a multiple of 10 in [10, 1000]");
  range(c.asr.max_active_streams >= 1 && c.asr.max_active_streams <= 8,
        "asr.max_active_streams must be in [1, 8]");
  range(c.asr.queue_capacity >= 8 && c.asr.queue_capacity <= 4096,
        "asr.queue_capacity must be in [8, 4096]");
  range(c.asr.max_pending_audio_ms >= 1000 && c.asr.max_pending_audio_ms <= 300000,
        "asr.max_pending_audio_ms must be in [1000, 300000]");
  range(c.asr.exact_final_redecode,
        "asr.exact_final_redecode must remain true: only exact candidate PCM may execute commands");
  range(c.commands.default_volume_step_percent >= 1 &&
            c.commands.default_volume_step_percent <= c.commands.max_spoken_volume_step_percent,
        "commands.default_volume_step_percent must be within the spoken volume range");
  range(c.commands.max_spoken_volume_step_percent >= 1 &&
            c.commands.max_spoken_volume_step_percent <= 20,
        "commands.max_spoken_volume_step_percent must be in [1, 20]");
  range(c.commands.max_actions_per_utterance >= 1 && c.commands.max_actions_per_utterance <= 8,
        "commands.max_actions_per_utterance must be in [1, 8]");
  range(c.commands.action_timeout_ms >= 100 && c.commands.action_timeout_ms <= 10000,
        "commands.action_timeout_ms must be in [100, 10000]");
  range(c.commands.queue_capacity >= 1 && c.commands.queue_capacity <= 256,
        "commands.queue_capacity must be in [1, 256]");
  range(c.announcements.backend == "log",
        "announcements.backend must be 'log'");
  range(c.announcements.queue_capacity >= 1 &&
            c.announcements.queue_capacity <= 256,
        "announcements.queue_capacity must be in [1, 256]");
  range(c.announcements.tail_guard_ms <= 1000,
        "announcements.tail_guard_ms must be in [0, 1000]");
  range(!c.commands.play_phrases.empty() && !c.commands.pause_phrases.empty() &&
            !c.commands.volume_up_phrases.empty() && !c.commands.volume_down_phrases.empty(),
        "commands phrase lists must not be empty");
  try {
    CommandGrammar grammar;
    grammar.play_phrases = c.commands.play_phrases;
    grammar.pause_phrases = c.commands.pause_phrases;
    grammar.volume_up_phrases = c.commands.volume_up_phrases;
    grammar.volume_down_phrases = c.commands.volume_down_phrases;
    grammar.connectors = c.commands.connectors;
    grammar.max_actions_per_utterance = c.commands.max_actions_per_utterance;
    const CommandParser parser(static_cast<int>(c.commands.default_volume_step_percent),
                               static_cast<int>(c.commands.max_spoken_volume_step_percent),
                               std::move(grammar));
    (void)parser;
  } catch (const std::exception& error) {
    range(false, std::string("commands grammar is invalid: ") + error.what());
  }
  range(c.web.bind == "127.0.0.1",
        "web.bind must be exactly '127.0.0.1'");
  range(c.web.port > 0, "web.port must be non-zero");
  range(c.web.telemetry_hz >= 1 && c.web.telemetry_hz <= 60, "web.telemetry_hz must be in [1, 60]");
  return v;
}

nlohmann::json ConfigStore::to_public_json(const AppConfig& c) const {
  return {
      {"revision", c.revision},
      {"audio", {{"microphone_device", c.audio.microphone_device},
                 {"loopback_device", c.audio.loopback_device}}},
      {"preprocess", {{"implementation", c.preprocess.implementation},
                      {"aec_enabled", c.preprocess.aec_enabled}}},
      {"aec", {{"enabled", c.aec.enabled},
               {"processing_rate_hz", c.aec.processing_rate_hz},
               {"high_pass_filter", c.aec.high_pass_filter},
               {"noise_suppression", c.aec.noise_suppression},
               {"gain_control", c.aec.gain_control},
               {"request_raw_capture", c.aec.request_raw_capture},
               {"request_post_volume_loopback", c.aec.request_post_volume_loopback},
               {"alignment_wait_ms", c.aec.alignment_wait_ms},
               {"target_render_buffer_ms", c.aec.target_render_buffer_ms},
               {"max_render_buffer_ms", c.aec.max_render_buffer_ms},
               {"drift_window_ms", c.aec.drift_window_ms},
               {"max_drift_ppm", c.aec.max_drift_ppm},
               {"hard_resync_error_ms", c.aec.hard_resync_error_ms},
               {"delay_offset_ms", c.aec.delay_offset_ms},
               {"microphone_channel_index", c.aec.microphone_channel_index},
               {"auto_delay_enabled", c.aec.auto_delay_enabled},
               {"auto_delay_min_ms", c.aec.auto_delay_min_ms},
               {"auto_delay_max_ms", c.aec.auto_delay_max_ms},
               {"auto_delay_window_ms", c.aec.auto_delay_window_ms},
               {"auto_delay_update_ms", c.aec.auto_delay_update_ms},
               {"auto_delay_min_correlation", c.aec.auto_delay_min_correlation},
               {"missing_render_policy", c.aec.missing_render_policy},
               {"stats_hz", c.aec.stats_hz}}},
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
                         {"embedded_join_silence_ms", c.segmentation.embedded_join_silence_ms},
                         {"assembly_queue_capacity", c.segmentation.assembly_queue_capacity}}},
      {"activation", {{"enabled", c.activation.enabled},
                       {"idle_timeout_ms", c.activation.idle_timeout_ms},
                       {"hard_limit_ms", c.activation.hard_limit_ms}}},
      {"asr", {{"enabled", c.asr.enabled}, {"emit_partials", c.asr.emit_partials},
               {"provider", c.asr.provider}, {"num_threads", c.asr.num_threads},
               {"feed_chunk_ms", c.asr.feed_chunk_ms},
               {"encoder", c.asr.encoder.string()}, {"decoder", c.asr.decoder.string()},
               {"tokens", c.asr.tokens.string()}}},
      {"commands", {{"enabled", c.commands.enabled},
                    {"default_volume_step_percent", c.commands.default_volume_step_percent},
                    {"max_spoken_volume_step_percent", c.commands.max_spoken_volume_step_percent},
                    {"max_actions_per_utterance", c.commands.max_actions_per_utterance},
                     {"play_phrases", c.commands.play_phrases},
                     {"pause_phrases", c.commands.pause_phrases},
                     {"volume_up_phrases", c.commands.volume_up_phrases},
                     {"volume_down_phrases", c.commands.volume_down_phrases},
                     {"phrases", {{"play", c.commands.play_phrases},
                                   {"pause", c.commands.pause_phrases},
                                   {"volume_up", c.commands.volume_up_phrases},
                                   {"volume_down", c.commands.volume_down_phrases}}},
                     {"connectors", c.commands.connectors}}},
      {"announcements", {{"enabled", c.announcements.enabled},
                          {"backend", c.announcements.backend},
                          {"queue_capacity", c.announcements.queue_capacity},
                          {"barge_in", c.announcements.barge_in},
                          {"tail_guard_ms", c.announcements.tail_guard_ms}}},
      {"web", {{"telemetry_hz", c.web.telemetry_hz}}},
      {"cold", {{"microphone_device", c.audio.microphone_device},
                 {"loopback_device", c.audio.loopback_device},
                 {"preprocess_implementation", c.preprocess.implementation},
                 {"aec_processing_rate_hz", c.aec.processing_rate_hz},
                 {"aec_raw_capture", c.aec.request_raw_capture},
                 {"aec_post_volume_loopback", c.aec.request_post_volume_loopback},
                 {"kws_provider", c.kws.provider}, {"kws_num_threads", c.kws.num_threads},
                 {"kws_encoder", c.kws.encoder.string()}, {"kws_decoder", c.kws.decoder.string()},
                 {"kws_joiner", c.kws.joiner.string()}, {"kws_tokens", c.kws.tokens.string()},
                 {"kws_keywords", c.kws.keywords.string()}, {"vad_provider", c.vad.provider},
                 {"vad_num_threads", c.vad.num_threads}, {"vad_model", c.vad.model.string()},
                 {"asr_provider", c.asr.provider}, {"asr_num_threads", c.asr.num_threads},
                 {"asr_encoder", c.asr.encoder.string()}, {"asr_decoder", c.asr.decoder.string()},
                 {"asr_tokens", c.asr.tokens.string()},
                 {"announcements_backend", c.announcements.backend},
                 {"announcements_queue_capacity", c.announcements.queue_capacity}}}
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
      {"aec", {{"enabled", c.aec.enabled},
               {"processing_rate_hz", c.aec.processing_rate_hz},
               {"high_pass_filter", c.aec.high_pass_filter},
               {"noise_suppression", c.aec.noise_suppression},
               {"gain_control", c.aec.gain_control},
               {"request_raw_capture", c.aec.request_raw_capture},
               {"request_post_volume_loopback", c.aec.request_post_volume_loopback},
               {"alignment_wait_ms", c.aec.alignment_wait_ms},
               {"target_render_buffer_ms", c.aec.target_render_buffer_ms},
               {"max_render_buffer_ms", c.aec.max_render_buffer_ms},
               {"drift_window_ms", c.aec.drift_window_ms},
               {"max_drift_ppm", c.aec.max_drift_ppm},
               {"hard_resync_error_ms", c.aec.hard_resync_error_ms},
               {"delay_offset_ms", c.aec.delay_offset_ms},
               {"microphone_channel_index", c.aec.microphone_channel_index},
               {"auto_delay_enabled", c.aec.auto_delay_enabled},
               {"auto_delay_min_ms", c.aec.auto_delay_min_ms},
               {"auto_delay_max_ms", c.aec.auto_delay_max_ms},
               {"auto_delay_window_ms", c.aec.auto_delay_window_ms},
               {"auto_delay_update_ms", c.aec.auto_delay_update_ms},
               {"auto_delay_min_correlation", c.aec.auto_delay_min_correlation},
               {"missing_render_policy", c.aec.missing_render_policy},
               {"stats_hz", c.aec.stats_hz}}},
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
                         {"embedded_join_silence_ms", c.segmentation.embedded_join_silence_ms},
                         {"assembly_queue_capacity", c.segmentation.assembly_queue_capacity}}},
      {"activation", {{"enabled", c.activation.enabled},
                       {"idle_timeout_ms", c.activation.idle_timeout_ms},
                       {"hard_limit_ms", c.activation.hard_limit_ms}}},
      {"asr", {{"enabled", c.asr.enabled}, {"implementation", c.asr.implementation},
               {"encoder", c.asr.encoder.string()}, {"decoder", c.asr.decoder.string()},
               {"tokens", c.asr.tokens.string()}, {"provider", c.asr.provider},
               {"num_threads", c.asr.num_threads}, {"feed_chunk_ms", c.asr.feed_chunk_ms},
               {"max_active_streams", c.asr.max_active_streams},
               {"queue_capacity", c.asr.queue_capacity},
               {"max_pending_audio_ms", c.asr.max_pending_audio_ms},
               {"emit_partials", c.asr.emit_partials},
               {"exact_final_redecode", c.asr.exact_final_redecode}}},
      {"commands", {{"enabled", c.commands.enabled},
                    {"default_volume_step_percent", c.commands.default_volume_step_percent},
                    {"max_spoken_volume_step_percent", c.commands.max_spoken_volume_step_percent},
                    {"max_actions_per_utterance", c.commands.max_actions_per_utterance},
                    {"action_timeout_ms", c.commands.action_timeout_ms},
                    {"queue_capacity", c.commands.queue_capacity},
                    {"phrases", {{"play", c.commands.play_phrases},
                                  {"pause", c.commands.pause_phrases},
                                  {"volume_up", c.commands.volume_up_phrases},
                                  {"volume_down", c.commands.volume_down_phrases}}},
                    {"connectors", c.commands.connectors}}},
      {"announcements", {{"enabled", c.announcements.enabled},
                          {"backend", c.announcements.backend},
                          {"queue_capacity", c.announcements.queue_capacity},
                          {"barge_in", c.announcements.barge_in},
                          {"tail_guard_ms", c.announcements.tail_guard_ms}}},
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
    if (const auto it = patch.find("preprocess"); it != patch.end()) {
      assign_if(*it, "implementation", candidate.preprocess.implementation);
      assign_if(*it, "aec_enabled", candidate.preprocess.aec_enabled);
    }
    if (const auto it = patch.find("aec"); it != patch.end()) {
      assign_if(*it, "enabled", candidate.aec.enabled);
      assign_if(*it, "alignment_wait_ms", candidate.aec.alignment_wait_ms);
      assign_if(*it, "target_render_buffer_ms", candidate.aec.target_render_buffer_ms);
      assign_if(*it, "max_render_buffer_ms", candidate.aec.max_render_buffer_ms);
      assign_if(*it, "drift_window_ms", candidate.aec.drift_window_ms);
      assign_if(*it, "max_drift_ppm", candidate.aec.max_drift_ppm);
      assign_if(*it, "hard_resync_error_ms", candidate.aec.hard_resync_error_ms);
      assign_if(*it, "delay_offset_ms", candidate.aec.delay_offset_ms);
      assign_if(*it, "microphone_channel_index", candidate.aec.microphone_channel_index);
      assign_if(*it, "auto_delay_enabled", candidate.aec.auto_delay_enabled);
      assign_if(*it, "auto_delay_min_ms", candidate.aec.auto_delay_min_ms);
      assign_if(*it, "auto_delay_max_ms", candidate.aec.auto_delay_max_ms);
      assign_if(*it, "auto_delay_window_ms", candidate.aec.auto_delay_window_ms);
      assign_if(*it, "auto_delay_update_ms", candidate.aec.auto_delay_update_ms);
      assign_if(*it, "auto_delay_min_correlation",
                candidate.aec.auto_delay_min_correlation);
      assign_if(*it, "stats_hz", candidate.aec.stats_hz);
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
      assign_if(*it, "assembly_queue_capacity", candidate.segmentation.assembly_queue_capacity);
    }
    if (const auto it = patch.find("activation"); it != patch.end()) {
      assign_if(*it, "enabled", candidate.activation.enabled);
      assign_if(*it, "idle_timeout_ms", candidate.activation.idle_timeout_ms);
      assign_if(*it, "hard_limit_ms", candidate.activation.hard_limit_ms);
    }
    if (const auto it = patch.find("asr"); it != patch.end()) {
      assign_if(*it, "enabled", candidate.asr.enabled);
      assign_if(*it, "emit_partials", candidate.asr.emit_partials);
      assign_if(*it, "provider", candidate.asr.provider);
      assign_if(*it, "num_threads", candidate.asr.num_threads);
      assign_if(*it, "feed_chunk_ms", candidate.asr.feed_chunk_ms);
      const auto path = [&](const char* key, std::filesystem::path& output) {
        if (const auto value = it->find(key); value != it->end()) {
          output = resolve(project_root_, std::filesystem::u8path(value->get<std::string>()));
        }
      };
      path("encoder", candidate.asr.encoder);
      path("decoder", candidate.asr.decoder);
      path("tokens", candidate.asr.tokens);
    }
    if (const auto it = patch.find("commands"); it != patch.end()) {
      assign_if(*it, "enabled", candidate.commands.enabled);
      assign_if(*it, "default_volume_step_percent", candidate.commands.default_volume_step_percent);
      assign_if(*it, "max_spoken_volume_step_percent", candidate.commands.max_spoken_volume_step_percent);
      assign_if(*it, "max_actions_per_utterance", candidate.commands.max_actions_per_utterance);
      assign_if(*it, "action_timeout_ms", candidate.commands.action_timeout_ms);
      assign_if(*it, "queue_capacity", candidate.commands.queue_capacity);
      assign_if(*it, "play_phrases", candidate.commands.play_phrases);
      assign_if(*it, "pause_phrases", candidate.commands.pause_phrases);
      assign_if(*it, "volume_up_phrases", candidate.commands.volume_up_phrases);
      assign_if(*it, "volume_down_phrases", candidate.commands.volume_down_phrases);
      assign_if(*it, "connectors", candidate.commands.connectors);
      if (const auto phrases = it->find("phrases"); phrases != it->end()) {
        assign_if(*phrases, "play", candidate.commands.play_phrases);
        assign_if(*phrases, "pause", candidate.commands.pause_phrases);
        assign_if(*phrases, "volume_up", candidate.commands.volume_up_phrases);
        assign_if(*phrases, "volume_down", candidate.commands.volume_down_phrases);
      }
    }
    if (const auto it = patch.find("announcements"); it != patch.end()) {
      assign_if(*it, "enabled", candidate.announcements.enabled);
      assign_if(*it, "backend", candidate.announcements.backend);
      assign_if(*it, "queue_capacity", candidate.announcements.queue_capacity);
      assign_if(*it, "barge_in", candidate.announcements.barge_in);
      assign_if(*it, "tail_guard_ms", candidate.announcements.tail_guard_ms);
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
  out << "[preprocess]\nimplementation = " << toml_string(c.preprocess.implementation)
      << "\naec_enabled = " << (c.preprocess.aec_enabled ? "true" : "false") << "\n\n";
  out << "[aec]\nenabled = " << (c.aec.enabled ? "true" : "false")
      << "\nalignment_wait_ms = " << c.aec.alignment_wait_ms
      << "\ntarget_render_buffer_ms = " << c.aec.target_render_buffer_ms
      << "\nmax_render_buffer_ms = " << c.aec.max_render_buffer_ms
      << "\ndrift_window_ms = " << c.aec.drift_window_ms
      << "\nmax_drift_ppm = " << c.aec.max_drift_ppm
      << "\nhard_resync_error_ms = " << c.aec.hard_resync_error_ms
      << "\ndelay_offset_ms = " << c.aec.delay_offset_ms
      << "\nmicrophone_channel_index = " << c.aec.microphone_channel_index
      << "\nauto_delay_enabled = " << (c.aec.auto_delay_enabled ? "true" : "false")
      << "\nauto_delay_min_ms = " << c.aec.auto_delay_min_ms
      << "\nauto_delay_max_ms = " << c.aec.auto_delay_max_ms
      << "\nauto_delay_window_ms = " << c.aec.auto_delay_window_ms
      << "\nauto_delay_update_ms = " << c.aec.auto_delay_update_ms
      << "\nauto_delay_min_correlation = " << c.aec.auto_delay_min_correlation
      << "\nstats_hz = " << c.aec.stats_hz << "\n\n";
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
      << "\nembedded_join_silence_ms = " << c.segmentation.embedded_join_silence_ms
      << "\nassembly_queue_capacity = " << c.segmentation.assembly_queue_capacity << "\n\n";
  out << "[activation]\nenabled = " << (c.activation.enabled ? "true" : "false")
      << "\nidle_timeout_ms = " << c.activation.idle_timeout_ms
      << "\nhard_limit_ms = " << c.activation.hard_limit_ms << "\n\n";
  out << "[asr]\nenabled = " << (c.asr.enabled ? "true" : "false")
      << "\nencoder = " << toml_string(c.asr.encoder.generic_string())
      << "\ndecoder = " << toml_string(c.asr.decoder.generic_string())
      << "\ntokens = " << toml_string(c.asr.tokens.generic_string())
      << "\nprovider = " << toml_string(c.asr.provider)
      << "\nnum_threads = " << c.asr.num_threads
      << "\nfeed_chunk_ms = " << c.asr.feed_chunk_ms
      << "\nemit_partials = " << (c.asr.emit_partials ? "true" : "false") << "\n\n";
  out << "[commands]\nenabled = " << (c.commands.enabled ? "true" : "false")
       << "\ndefault_volume_step_percent = " << c.commands.default_volume_step_percent
       << "\nmax_spoken_volume_step_percent = " << c.commands.max_spoken_volume_step_percent
       << "\nmax_actions_per_utterance = " << c.commands.max_actions_per_utterance
       << "\naction_timeout_ms = " << c.commands.action_timeout_ms
       << "\nqueue_capacity = " << c.commands.queue_capacity
       << "\nconnectors = " << toml_array(c.commands.connectors) << "\n\n"
      << "[commands.phrases]\nplay = " << toml_array(c.commands.play_phrases)
      << "\npause = " << toml_array(c.commands.pause_phrases)
      << "\nvolume_up = " << toml_array(c.commands.volume_up_phrases)
      << "\nvolume_down = " << toml_array(c.commands.volume_down_phrases) << "\n\n";
  out << "[announcements]\nenabled = "
      << (c.announcements.enabled ? "true" : "false")
      << "\nbackend = " << toml_string(c.announcements.backend)
      << "\nqueue_capacity = " << c.announcements.queue_capacity
      << "\nbarge_in = " << (c.announcements.barge_in ? "true" : "false")
      << "\ntail_guard_ms = " << c.announcements.tail_guard_ms << "\n\n";
  out << "[web]\ntelemetry_hz = " << c.web.telemetry_hz << "\n";
  out.close();
  if (!out) throw std::runtime_error("failed while writing config override file");
  std::error_code error;
  std::filesystem::remove(overrides_, error);
  std::filesystem::rename(temporary, overrides_);
}

}  // namespace dvo
