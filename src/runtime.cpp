#include "dvo/runtime.h"

#include "dvo/benchmark_report.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "dvo/json_utils.h"
#include "dvo/hash.h"
#include "dvo/wav_file.h"
#include "dvo/windows_action_backend.h"

namespace dvo {
namespace {

// The executor retains 4096 terminal action IDs. Keeping at most 512 plans
// guarantees that all eight actions of every replayable snapshot still fit in
// the deduplication ledger.
constexpr std::size_t kCompletedCommandSnapshotCapacity = 4096 / kMaxCommandActions;

std::uint64_t candidate_end_sample(const UtteranceCandidate& candidate) {
  if (!candidate.source_spans.empty()) return candidate.source_spans.back().end;
  if (candidate.wake_span) return candidate.wake_span->end;
  return candidate.trigger_sample;
}

nlohmann::json activation_json(const ActivationSnapshot& activation) {
  return {{"active", activation.active()},
          {"state", to_string(activation.state)},
          {"activation_id", activation.activation_id},
          {"turn_index", activation.turn_index},
          {"idle_deadline_sample", activation.idle_deadline_sample},
          {"hard_deadline_sample", activation.hard_deadline_sample},
          {"playback_hold", activation.playback_hold}};
}

nlohmann::json announcement_request_json(const AnnouncementRequest& request,
                                         std::string status) {
  return {{"schema_version", request.schema_version},
          {"announcement_id", request.announcement_id},
          {"kind", to_string(request.kind)},
          {"status", std::move(status)},
          {"text", request.text},
          {"source", to_string(request.source)},
          {"origin", to_string(request.origin)},
          {"activation_id", request.activation_id},
          {"turn_index", request.turn_index},
          {"trigger_sample", request.trigger_sample},
          {"utterance_id", request.utterance_id},
          {"command_id", request.command_id},
          {"action_ids", request.action_ids},
          {"timestamp_sample", request.timestamp_sample}};
}

nlohmann::json announcement_result_json(const AnnouncementResult& result) {
  return {{"schema_version", result.schema_version},
          {"announcement_id", result.announcement_id},
          {"kind", to_string(result.kind)},
          {"status", to_string(result.status)},
          {"text", result.text},
          {"source", to_string(result.source)},
          {"origin", to_string(result.origin)},
          {"activation_id", result.activation_id},
          {"turn_index", result.turn_index},
          {"trigger_sample", result.trigger_sample},
          {"utterance_id", result.utterance_id},
          {"command_id", result.command_id},
          {"action_ids", result.action_ids},
          {"timestamp_sample", result.timestamp_sample},
          {"backend", result.backend},
          {"render_endpoint", result.render_endpoint},
          {"error_code", result.error_code},
          {"message", result.message},
          {"duration_ms", result.duration.count()}};
}

std::string command_snapshot_key(std::string_view utterance_id,
                                 std::uint64_t generation) {
  return std::to_string(utterance_id.size()) + ":" + std::string(utterance_id) + ":g" +
         std::to_string(generation);
}

std::string make_runtime_session_id() {
  static std::atomic<std::uint64_t> sequence{};
  const auto now = static_cast<std::uint64_t>(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::random_device random;
  const auto salt = (static_cast<std::uint64_t>(random()) << 32U) |
                    static_cast<std::uint64_t>(random());
  return std::to_string(now) + "-" + std::to_string(salt) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

nlohmann::json command_plan_json(const CommandPlan& plan) {
  auto actions = nlohmann::json::array();
  for (const auto& action : plan.actions) {
    auto action_value = nlohmann::json{{"action_id", action.action_id},
                                       {"sequence", action.sequence},
                                       {"type", to_string(action.type)}};
    action_value["volume_delta_percent"] =
        action.volume_delta_percent
            ? nlohmann::json(*action.volume_delta_percent)
            : nlohmann::json(nullptr);
    actions.push_back(std::move(action_value));
  }
  auto value = nlohmann::json{{"schema_version", plan.schema_version},
          {"command_id", plan.command_id}, {"runtime_session_id", plan.runtime_session_id},
          {"utterance_id", plan.utterance_id},
          {"origin", to_string(plan.origin)}, {"activation_id", plan.activation_id},
          {"turn_index", plan.turn_index}, {"trigger_sample", plan.trigger_sample},
          {"raw_text", plan.raw_text}, {"normalized_text", plan.normalized_text},
          {"parser_version", plan.parser_version}, {"config_revision", plan.config_revision},
           {"recognition_generation", plan.recognition_generation},
           {"final_revision", plan.final_revision},
           {"timestamp_sample", plan.timestamp_sample},
           {"source", to_string(plan.source)},
          {"execution_mode", to_string(plan.execution_mode)}, {"actions", std::move(actions)}};
  value["wake_position"] = plan.wake_position.empty()
                               ? nlohmann::json(nullptr)
                               : nlohmann::json(plan.wake_position);
  return value;
}

nlohmann::json parse_debug_base(std::string_view input_text,
                                const CommandParseContext& context,
                                std::string_view kind,
                                std::string_view stage) {
  return {{"schema_version", 1},
          {"kind", kind},
          {"stage", stage},
          {"matched", false},
          {"input_text", input_text},
          {"utterance_id", context.utterance_id},
          {"origin", to_string(context.origin)},
          {"wake_word", context.wake_word},
          {"config_revision", context.config_revision},
          {"recognition_generation", context.recognition_generation},
          {"final_revision", context.final_revision},
          {"timestamp_sample", context.timestamp_sample},
          {"source", to_string(context.source)},
          {"execution_mode", to_string(context.execution_mode)},
          {"parser_version", kCommandParserVersion}};
}

nlohmann::json parse_debug_success(std::string_view input_text,
                                   const CommandParseContext& context,
                                   const CommandPlan& plan,
                                   std::string_view kind) {
  auto value = parse_debug_base(input_text, context, kind, "parser");
  value["matched"] = true;
  value["normalized_text"] = plan.normalized_text;
  value["plan"] = command_plan_json(plan);
  value["error"] = nullptr;
  return value;
}

nlohmann::json parse_debug_failure(std::string_view input_text,
                                   const CommandParseContext& context,
                                   std::string_view kind,
                                   std::string_view stage,
                                   std::string code,
                                   std::string message,
                                   std::size_t byte_offset = 0) {
  auto value = parse_debug_base(input_text, context, kind, stage);
  value["normalized_text"] = "";
  value["plan"] = nullptr;
  value["error"] = {{"code", std::move(code)},
                    {"message", std::move(message)},
                    {"byte_offset", byte_offset}};
  return value;
}

nlohmann::json action_result_json(const ActionResult& result) {
  auto value = nlohmann::json{
      {"schema_version", result.schema_version},
      {"command_id", result.command_id}, {"utterance_id", result.utterance_id},
      {"origin", to_string(result.origin)}, {"activation_id", result.activation_id},
      {"turn_index", result.turn_index}, {"trigger_sample", result.trigger_sample},
      {"action_id", result.action_id}, {"sequence", result.sequence},
      {"type", to_string(result.type)}, {"status", to_string(result.status)},
      {"source", to_string(result.source)},
      {"timestamp_sample", result.timestamp_sample},
      {"adapter", result.adapter},
      {"target_id", result.target_id}, {"error_code", result.error_code},
      {"message", result.message}, {"clamped", result.clamped},
      {"verified", result.verified}, {"duration_ms", result.duration.count()}};
  const auto optional_json = [](const auto& item) {
    return item ? nlohmann::json(*item) : nlohmann::json(nullptr);
  };
  value["requested_volume_delta_percent"] =
      optional_json(result.requested_volume_delta_percent);
  value["volume_before"] = optional_json(result.volume_before);
  value["volume_requested"] = optional_json(result.volume_requested);
  value["volume_after"] = optional_json(result.volume_after);
  value["mute_before"] = optional_json(result.mute_before);
  value["mute_after"] = optional_json(result.mute_after);
  return value;
}

CommandGrammar command_grammar(const CommandsConfig& config) {
  CommandGrammar grammar;
  grammar.play_phrases = config.play_phrases;
  grammar.pause_phrases = config.pause_phrases;
  grammar.volume_up_phrases = config.volume_up_phrases;
  grammar.volume_down_phrases = config.volume_down_phrases;
  grammar.connectors = config.connectors;
  grammar.max_actions_per_utterance = config.max_actions_per_utterance;
  return grammar;
}

nlohmann::json preprocess_json(const PreprocessDiagnostics& diagnostics) {
  auto value = nlohmann::json{
          {"state", to_string(diagnostics.state)}, {"enabled", diagnostics.aec_requested},
          {"compiled", diagnostics.aec_compiled}, {"active", diagnostics.aec_active},
          {"degraded", diagnostics.degraded}, {"render_available", diagnostics.render_available},
          {"render_synthetic", diagnostics.render_synthetic},
          {"microphone_rate_hz", diagnostics.microphone_rate_hz},
          {"render_rate_hz", diagnostics.render_rate_hz},
          {"drift_ppm", diagnostics.relative_drift_ppm},
            {"drift_estimate_valid", diagnostics.drift_estimate_valid},
            {"drift_out_of_range", diagnostics.drift_out_of_range},
            {"external_delay_ms", diagnostics.stream_delay_ms},
            {"auto_delay_enabled", diagnostics.auto_delay_enabled},
            {"auto_delay_available", diagnostics.auto_delay_available},
            {"auto_delay_ms", diagnostics.auto_delay_available
                                ? nlohmann::json(diagnostics.auto_delay_ms)
                                : nlohmann::json(nullptr)},
          {"auto_delay_confidence", diagnostics.auto_delay_available
                                        ? nlohmann::json(diagnostics.auto_delay_confidence)
                                          : nlohmann::json(nullptr)},
            {"auto_delay_updates", diagnostics.auto_delay_updates},
            {"auto_delay_rejections", diagnostics.auto_delay_rejections},
            {"render_fifo_ms", diagnostics.render_buffered_samples * 1000.0 /
                                  static_cast<double>(kProcessingSampleRate)},
          {"render_target_ms", diagnostics.target_render_buffer_samples * 1000.0 /
                                   static_cast<double>(kProcessingSampleRate)},
          {"processing_average_ms", diagnostics.average_processing_time_us / 1000.0},
          {"processing_max_ms", diagnostics.max_processing_time_us / 1000.0},
          {"reset_count", diagnostics.resets}, {"fallback_frames", diagnostics.fallback_frames},
          {"last_reset_reason", to_string(diagnostics.last_reset_reason)}};
  const auto optional_json = [](const auto& item) {
    return item ? nlohmann::json(*item) : nlohmann::json(nullptr);
  };
  value["erle_db"] = optional_json(diagnostics.echo_return_loss_enhancement_db);
  value["erl_db"] = optional_json(diagnostics.echo_return_loss_db);
  value["residual_echo_likelihood"] =
      optional_json(diagnostics.residual_echo_likelihood);
  value["divergent_filter_fraction"] =
      optional_json(diagnostics.divergent_filter_fraction);
  value["estimated_delay_ms"] = optional_json(diagnostics.estimated_delay_ms);
  value["resampler"] = nlohmann::json{
      {"speex_compiled", diagnostics.speexdsp_compiled},
      {"microphone_speex", diagnostics.microphone_resampler_speex},
      {"render_speex", diagnostics.render_resampler_speex},
      {"render_rate_updates", diagnostics.render_resampler_rate_updates},
      {"synthetic_samples_replaced", diagnostics.render_synthetic_samples_replaced},
      {"failures", diagnostics.resampler_failures},
      {"last_error_code", diagnostics.last_resampler_error_code},
      {"last_input_expected", diagnostics.last_resampler_input_expected},
      {"last_input_consumed", diagnostics.last_resampler_input_consumed}};
  return value;
}

bool valid_session_name(std::string_view name) {
  return !name.empty() && name.size() <= 128 &&
         std::all_of(name.begin(), name.end(), [](unsigned char character) {
           return std::isalnum(character) != 0 || character == '-' ||
                  character == '_';
         });
}

std::optional<nlohmann::json> inspect_session_directory(
    const std::filesystem::path& session) {
  std::error_code error;
  const auto manifest_path = session / "manifest.json";
  const auto timeline_path = session / "timeline.ndjson";
  if (!std::filesystem::is_regular_file(manifest_path, error) ||
      !std::filesystem::is_regular_file(timeline_path, error) ||
      std::filesystem::file_size(timeline_path, error) == 0 || error) {
    return std::nullopt;
  }

  nlohmann::json manifest;
  try {
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input) return std::nullopt;
    manifest = nlohmann::json::parse(input);
  } catch (...) {
    return std::nullopt;
  }

  auto streams = nlohmann::json::object();
  double preferred_duration{};
  std::string default_stream;
  const auto add_stream = [&](std::string_view name,
                              std::string_view filename) {
    try {
      FloatWavReader reader(session / filename);
      const auto format = reader.format();
      const auto frames = reader.frame_count();
      if (format.sample_rate == 0 || frames == 0) return;
      const auto duration =
          static_cast<double>(frames) / static_cast<double>(format.sample_rate);
      streams[std::string(name)] = {
          {"present", true},       {"frames", frames},
          {"sample_rate", format.sample_rate},
          {"channels", format.channels},
          {"duration_seconds", duration}};
      if (default_stream.empty() || name == "processed") {
        default_stream = std::string(name);
        preferred_duration = duration;
      }
    } catch (...) {
      // A malformed or unsupported stream is excluded without hiding other
      // usable tracks in the same session.
    }
  };
  add_stream("microphone", "mic.wav");
  add_stream("loopback", "loopback.wav");
  add_stream("processed", "processed.wav");

  // Analysis replay requires raw microphone packets. Audible playback can use
  // any stream, but sessions without microphone audio are not valid captures.
  if (!streams.contains("microphone")) return std::nullopt;
  return nlohmann::json{
      {"name", session.filename().string()},
      {"path", session.string()},
      {"complete", manifest.value("complete", false)},
      {"default_stream", default_stream},
      {"duration_seconds", preferred_duration},
      {"streams", std::move(streams)}};
}

}  // namespace

VoiceFrontendRuntime::VoiceFrontendRuntime(AppConfig config, ConfigStore store)
    : config_(std::move(config)), config_store_(std::move(store)),
      recorder_(std::max<std::size_t>(256,
          ((config_.recording.queue_capacity_ms + config_.audio.frame_ms - 1) /
           config_.audio.frame_ms) * 3 + 256)),
      debug_(8192) {}

VoiceFrontendRuntime::~VoiceFrontendRuntime() { stop(); }

void VoiceFrontendRuntime::initialize_pipeline() {
  std::scoped_lock lock(pipeline_mutex_);
  runtime_session_id_ = make_runtime_session_id();
  const auto capacity = static_cast<std::size_t>(config_.ring.duration_ms) * kProcessingSampleRate / 1000;
  ring_ = std::make_unique<TimedRingBuffer>(capacity);
  microphone_ring_ = std::make_unique<TimedRingBuffer>(capacity);
  const auto timeline = make_preprocessor_timeline_config(config_.aec);
  if (config_.preprocess.aec_enabled && config_.aec.enabled &&
      config_.preprocess.implementation == "webrtc_aec3") {
    preprocessor_ = std::make_unique<WebRtcAec3Preprocessor>(config_.audio, timeline);
  } else {
    preprocessor_ = std::make_unique<BypassPreprocessor>(config_.audio, timeline);
  }
  // Preserve a microphone-only KWS path beside the AEC path. Fallback hits
  // are later gated by processed VAD speech so render-only audio cannot
  // self-wake the assistant.
  keyword_preprocessor_ =
      std::make_unique<BypassPreprocessor>(config_.audio, timeline);
  vad_ = create_vad(config_.vad);
  kws_ = create_keyword_spotter(config_.kws);
  microphone_kws_ = create_keyword_spotter(config_.kws);
  segmenter_ = std::make_unique<UtteranceSegmenter>(
      config_.segmentation, config_.activation, *ring_);
  candidate_assembler_ = std::make_unique<CandidateAssembler>(
      *ring_, config_.segmentation.assembly_queue_capacity);
  microphone_candidate_assembler_ = std::make_unique<CandidateAssembler>(
      *microphone_ring_, config_.segmentation.assembly_queue_capacity);
  StreamingRecognizerConfig asr_config;
  asr_config.enabled = config_.asr.enabled;
  asr_config.encoder = config_.asr.encoder;
  asr_config.decoder = config_.asr.decoder;
  asr_config.tokens = config_.asr.tokens;
  asr_config.provider = config_.asr.provider;
  asr_config.num_threads = config_.asr.num_threads;
  asr_config.sample_rate = kProcessingSampleRate;
  asr_config.queue_capacity = config_.asr.queue_capacity;
  asr_config.max_active_streams = config_.asr.max_active_streams;
  asr_config.max_pending_audio_ms = config_.asr.max_pending_audio_ms;
  asr_ = create_streaming_recognizer(
      std::move(asr_config), [this](RecognitionResult result) {
        handle_recognition_result(std::move(result));
      });
  last_published_asr_state_ = asr_->state();
  {
    std::scoped_lock command_lock(command_mutex_);
    command_parser_ = std::make_shared<const CommandParser>(
        static_cast<int>(config_.commands.default_volume_step_percent),
        static_cast<int>(config_.commands.max_spoken_volume_step_percent),
        command_grammar(config_.commands));
    command_config_revision_ = config_.revision;
    commands_enabled_ = config_.commands.enabled;
    command_snapshots_.clear();
    completed_command_snapshots_.clear();
    completed_command_snapshot_order_.clear();
  }
  announcement_dispatcher_ = std::make_unique<AnnouncementDispatcher>(
      std::make_shared<LogAnnouncementBackend>(),
      AnnouncementDispatcherConfig{config_.announcements.queue_capacity, false},
      [this](const AnnouncementResult& result) {
        handle_announcement_result(result);
      });
  action_executor_ = std::make_unique<OrderedActionExecutor>(
      create_windows_action_backend(
          {config_.audio.loopback_device, config_.commands.action_timeout_ms}),
      ActionExecutorConfig{config_.commands.queue_capacity, 4096, ExecutionMode::live},
      [this](const ActionResult& result) { handle_action_result(result); },
      [this](const CommandPlan& plan) { handle_plan_started(plan); });
  {
    std::scoped_lock announcement_lock(announcement_state_mutex_);
    latest_live_announcement_ = nullptr;
    latest_replay_announcement_ = nullptr;
  }
  active_recognizers_.clear();
  pending_recognition_starts_.clear();
  recognition_generation_.fetch_add(1, std::memory_order_acq_rel);
  announcement_playback_active_.store(false, std::memory_order_release);
  announcement_guard_until_sample_.store(0, std::memory_order_release);
  announcement_kws_reset_pending_.store(false, std::memory_order_release);
  announcement_playback_start_sample_ = 0;
  vad_speech_active_ = false;
  suppress_current_vad_interval_ = false;
  pending_microphone_keywords_.clear();
  last_runtime_keyword_.clear();
  last_runtime_keyword_sample_ = 0;
  microphone_candidate_fallback_after_sample_ = 0;
  microphone_audio_activations_.clear();
  last_published_activation_state_ = {};
  reset_audio_queues();
  telemetry_queue_ = std::make_unique<SpscQueue<TelemetrySample>>(128);
  telemetry_queue_drops_.store(0, std::memory_order_release);
  telemetry_enabled_.store(false, std::memory_order_release);
  next_telemetry_sample_ = 0;
  view_sample_origin_.reset();
}

void VoiceFrontendRuntime::reset_audio_queues() {
  const auto queue_capacity = std::max<std::size_t>(
      1, (config_.audio.queue_capacity_ms + config_.audio.frame_ms - 1) / config_.audio.frame_ms);
  microphone_queue_ = std::make_unique<SpscQueue<AudioIngressPacket>>(queue_capacity);
  loopback_queue_ = std::make_unique<SpscQueue<AudioIngressPacket>>(queue_capacity);
  microphone_overflow_.store(false, std::memory_order_release);
  loopback_overflow_.store(false, std::memory_order_release);
  latest_microphone_summary_ = {};
  latest_loopback_summary_ = {};
}

void VoiceFrontendRuntime::start_live(bool enable_web) {
  stop();
  replay_mode_.store(false, std::memory_order_release);
  initialize_pipeline();
  running_.store(true, std::memory_order_release);
  if (enable_web && config_.web.enabled) {
    debug_.start(
        config_.web,
        [this](const auto& command) { return handle_command(command); },
        [this](const auto& request) {
          return session_audio_resource(request);
        });
    telemetry_enabled_.store(true, std::memory_order_release);
  }
  start_processing();
  start_captures();
  emit_event("runtime_started", {{"mode", "live"}, {"vad", vad_->status()}, {"kws", kws_->status()}});
  emit_event("aec_status", preprocess_json(preprocess_diagnostics_snapshot()));
  emit_event("asr_status", {{"state", to_string(asr_->state())},
                            {"status", asr_->status()}});
  emit_event("runtime_mode", {{"mode", "live"}});
  emit_activation_state("live");
  emit_event("config_state", {{"config", config_store_.to_public_json(config_)}});
  emit_event("sessions", session_list());
}

void VoiceFrontendRuntime::start_replay(const std::filesystem::path& session, bool enable_web,
                                        double speed) {
  stop();
  replay_mode_.store(true, std::memory_order_release);
  initialize_pipeline();
  running_.store(true, std::memory_order_release);
  if (enable_web && config_.web.enabled) {
    debug_.start(
        config_.web,
        [this](const auto& command) { return handle_command(command); },
        [this](const auto& request) {
          return session_audio_resource(request);
        });
    telemetry_enabled_.store(true, std::memory_order_release);
  }
  start_processing();
  replay_ = std::make_unique<ReplayController>(
      [this](AudioPacket packet) { enqueue_replay_packet(std::move(packet)); },
      [this](nlohmann::json state) {
        emit_event("replay_state", std::move(state), 0, "replay");
      });
  replay_->open(session);
  replay_->set_speed(speed);
  replay_->play();
  emit_event("runtime_started", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
  emit_event("aec_status", preprocess_json(preprocess_diagnostics_snapshot()), 0, "replay");
  emit_event("asr_status", {{"state", to_string(asr_->state())},
                            {"status", asr_->status()}}, 0, "replay");
  emit_event("runtime_mode", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
  emit_activation_state("replay");
  emit_event("config_state", {{"config", config_store_.to_public_json(config_)}}, 0, "replay");
  emit_event("sessions", session_list(), 0, "replay");
}

nlohmann::json VoiceFrontendRuntime::run_benchmark(const std::filesystem::path& session) {
  {
    std::scoped_lock lock(benchmark_events_mutex_);
    benchmark_events_.clear();
  }
  benchmark_capture_enabled_.store(true, std::memory_order_release);
  struct CaptureGuard {
    std::atomic<bool>& enabled;
    ~CaptureGuard() { enabled.store(false, std::memory_order_release); }
  } capture_guard{benchmark_capture_enabled_};
  const auto started = std::chrono::steady_clock::now();
  const auto deadline = started + std::chrono::minutes(5);
  const auto remaining = [&] {
    const auto now = std::chrono::steady_clock::now();
    return now >= deadline
               ? std::chrono::milliseconds{0}
               : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
  };
  try {
    start_replay(session, false, 0.0);
  } catch (...) {
    stop();
    throw;
  }
  const bool replay_finished = replay_ && replay_->wait_until_finished(remaining());

  bool pipeline_idle{};
  while (replay_finished && running() && remaining().count() > 0) {
    if (microphone_queue_->size() != 0 || loopback_queue_->size() != 0 ||
        (candidate_assembler_ && candidate_assembler_->outstanding() != 0) ||
        (microphone_candidate_assembler_ &&
         microphone_candidate_assembler_->outstanding() != 0)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    // No producer remains after wait_until_finished(). Requesting stop after
    // both queues become empty and then joining is the actual processing
    // barrier: if processing_loop already popped the last packet, join waits
    // for that packet to finish. Merely taking pipeline_mutex_ here could win
    // the lock race ahead of the popped packet and report a false idle state.
    stop_processing();
    pipeline_idle = microphone_queue_->size() == 0 &&
                    loopback_queue_->size() == 0 &&
                    (!candidate_assembler_ ||
                     candidate_assembler_->outstanding() == 0) &&
                    (!microphone_candidate_assembler_ ||
                     microphone_candidate_assembler_->outstanding() == 0);
    break;
  }

  const bool asr_load_finished =
      asr_ ? asr_->wait_until_loaded(remaining()) : true;
  const bool asr_idle = asr_ ? asr_->wait_until_idle(remaining()) : true;
  const bool actions_idle =
      action_executor_ ? action_executor_->wait_until_idle(remaining()) : true;
  const bool announcements_idle = announcement_dispatcher_
                                      ? announcement_dispatcher_->wait_until_idle(remaining())
                                      : true;
  auto metrics = metrics_json();
  metrics["elapsed_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started).count();
  metrics["session"] = session.string();
  metrics["benchmark_wait"] = {{"replay_finished", replay_finished},
                               {"pipeline_idle", pipeline_idle},
                               {"asr_load_finished", asr_load_finished},
                                {"asr_idle", asr_idle},
                                {"actions_idle", actions_idle},
                                {"announcements_idle", announcements_idle},
                                {"timed_out", !(replay_finished && pipeline_idle &&
                                                 asr_load_finished && asr_idle &&
                                                 actions_idle && announcements_idle)},
                               {"active_provisional_streams",
                                active_recognizers_.size()}};
  benchmark_capture_enabled_.store(false, std::memory_order_release);
  std::vector<nlohmann::json> replayed_events;
  {
    std::scoped_lock lock(benchmark_events_mutex_);
    replayed_events = benchmark_events_;
  }
  std::vector<std::string> event_read_errors;
  const auto recorded_events =
      read_benchmark_events(session / "events.ndjson", &event_read_errors);
  metrics["comparison"] = build_benchmark_comparison(
      recorded_events, replayed_events, kFrameSamples);
  metrics["comparison"]["historical_events_available"] =
      std::filesystem::exists(session / "events.ndjson");
  metrics["comparison"]["event_read_errors"] = event_read_errors;
  stop();
  return metrics;
}

void VoiceFrontendRuntime::start_processing() {
  telemetry_thread_ =
      std::jthread([this](std::stop_token stop) { telemetry_loop(stop); });
  processing_thread_ = std::jthread([this](std::stop_token stop) { processing_loop(stop); });
}

void VoiceFrontendRuntime::stop_processing() {
  if (processing_thread_.joinable()) {
    processing_thread_.request_stop();
    processing_thread_.join();
  }
  if (telemetry_thread_.joinable()) {
    telemetry_thread_.request_stop();
    telemetry_thread_.join();
  }
}

void VoiceFrontendRuntime::telemetry_loop(std::stop_token stop) {
  while (!stop.stop_requested() ||
         (telemetry_queue_ && telemetry_queue_->size() != 0)) {
    TelemetrySample sample;
    if (!telemetry_queue_ || !telemetry_queue_->try_pop(sample)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    publish_telemetry(std::move(sample));
  }
}

void VoiceFrontendRuntime::publish_telemetry(TelemetrySample sample) {
  const auto summary_json = [](const SignalSummary& summary) {
    return nlohmann::json{{"min", summary.min}, {"max", summary.max},
                          {"rms", summary.rms}};
  };
  emit_event(
      "telemetry",
      {{"sample", sample.sample},
       {"view_sample", sample.view_sample},
       {"microphone", summary_json(sample.microphone)},
       {"loopback", summary_json(sample.loopback)},
       {"processed", summary_json(sample.processed)},
       {"vad", sample.vad},
       {"queue_depth", sample.audio_queue_depth},
       {"asr_queue_depth", sample.asr_queue_depth},
       {"asr_pending_requests", sample.asr_pending_requests},
       {"asr_pending_audio_samples", sample.asr_pending_audio_samples},
       {"asr_state", to_string(sample.asr_state)},
       {"assembly_outstanding", sample.assembly_outstanding},
       {"action_queue_depth", sample.action_queue_depth},
       {"announcement_queue_depth", sample.announcement_queue_depth},
       {"activation", activation_json(sample.activation)},
       {"aec", preprocess_json(sample.aec)},
       {"telemetry_dropped",
        sample.debug_dropped +
            telemetry_queue_drops_.load(std::memory_order_relaxed)},
       {"audio_queue_drops", sample.audio_queue_drops},
       {"audio_reset_backlog_drops", sample.audio_reset_backlog_drops}},
      sample.sample, sample.replay ? "replay" : "live");
}

void VoiceFrontendRuntime::start_captures() {
  const auto event = [this](std::string type, std::string message) {
    queue_capture_event(std::move(type), std::move(message));
  };
  const auto microphone_pool_slots = microphone_queue_->capacity() + 2;
  const auto loopback_pool_slots = loopback_queue_->capacity() + 2;
  microphone_capture_.start(
      {AudioStreamKind::microphone, config_.audio.microphone_device,
       config_.audio.follow_default_device, config_.aec.request_raw_capture, false,
       microphone_pool_slots},
      [this](RawCapturedPacket packet) {
        return enqueue_capture_packet(std::move(packet));
      }, event);
  loopback_capture_.start(
      {AudioStreamKind::loopback, config_.audio.loopback_device,
       config_.audio.follow_default_device, false,
       config_.aec.request_post_volume_loopback, loopback_pool_slots},
      [this](RawCapturedPacket packet) {
        return enqueue_capture_packet(std::move(packet));
      }, event);
}

void VoiceFrontendRuntime::stop_captures() {
  microphone_capture_.stop();
  loopback_capture_.stop();
}

void VoiceFrontendRuntime::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel) && !processing_thread_.joinable()) return;
  telemetry_enabled_.store(false, std::memory_order_release);
  // Stop accepting control requests and wait for any admitted HTTP command
  // before replay/config/audio objects are destroyed.
  debug_.stop();
  stop_captures();
  if (replay_) { replay_->stop(); replay_.reset(); }
  stop_processing();
  {
    std::scoped_lock lock(pipeline_mutex_);
    if (segmenter_) {
      segmenter_->cancel_activation(ring_ ? ring_->tail() : 0,
                                    "runtime_stopped");
      emit_activation_events(segmenter_->take_activation_events());
      emit_activation_state_if_changed(ring_ ? ring_->tail() : 0,
                                       replay_mode_.load() ? "replay" : "live");
    }
  }
  stop_async_services();
  recorder_.stop(metrics_json());
}

void VoiceFrontendRuntime::stop_async_services() {
  if (candidate_assembler_) candidate_assembler_->stop();
  candidate_assembler_.reset();
  if (microphone_candidate_assembler_) {
    microphone_candidate_assembler_->stop();
  }
  microphone_candidate_assembler_.reset();
  const auto generation = recognition_generation_.load(std::memory_order_acquire);
  if (asr_) asr_->cancel_generation(generation);
  active_recognizers_.clear();
  pending_recognition_starts_.clear();
  asr_.reset();
  if (action_executor_) action_executor_->stop(false);
  action_executor_.reset();
  if (announcement_dispatcher_) announcement_dispatcher_->stop(false);
  announcement_dispatcher_.reset();
  announcement_playback_active_.store(false, std::memory_order_release);
  announcement_guard_until_sample_.store(0, std::memory_order_release);
  announcement_kws_reset_pending_.store(false, std::memory_order_release);
  announcement_playback_start_sample_ = 0;
  {
    std::scoped_lock command_lock(command_mutex_);
    command_snapshots_.clear();
    completed_command_snapshots_.clear();
    completed_command_snapshot_order_.clear();
    command_parser_.reset();
    command_config_revision_ = 0;
    commands_enabled_ = false;
  }
}

void VoiceFrontendRuntime::enqueue_replay_packet(AudioPacket packet) {
  std::scoped_lock lock(replay_callback_mutex_);
  if (!replay_mode_.load(std::memory_order_acquire)) return;
  enqueue_packet(std::move(packet));
}

void VoiceFrontendRuntime::enqueue_packet(AudioPacket packet) {
  audio_packets_.fetch_add(1, std::memory_order_relaxed);
  const auto stream = packet.stream;
  auto& queue = stream == AudioStreamKind::microphone ? *microphone_queue_ : *loopback_queue_;
  auto& overflow = stream == AudioStreamKind::microphone ? microphone_overflow_ : loopback_overflow_;
  AudioIngressPacket ingress(std::in_place_type<AudioPacket>, std::move(packet));
  if (replay_mode_.load(std::memory_order_acquire)) {
    while (running_.load(std::memory_order_acquire) &&
           !queue.try_push(std::move(ingress))) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return;
  }
  if (!queue.try_push(std::move(ingress))) {
    overflow.store(true, std::memory_order_release);
    audio_queue_drops_.fetch_add(1, std::memory_order_relaxed);
  }
}

bool VoiceFrontendRuntime::enqueue_capture_packet(RawCapturedPacket packet) {
  audio_packets_.fetch_add(1, std::memory_order_relaxed);
  const auto stream = packet.stream;
  const auto pool_drops = packet.pool_drops_before;
  auto& queue = stream == AudioStreamKind::microphone ? *microphone_queue_ :
                                                        *loopback_queue_;
  auto& overflow = stream == AudioStreamKind::microphone ? microphone_overflow_ :
                                                           loopback_overflow_;
  if (pool_drops != 0) {
    overflow.store(true, std::memory_order_release);
    audio_queue_drops_.fetch_add(pool_drops, std::memory_order_relaxed);
  }
  AudioIngressPacket ingress(std::in_place_type<RawCapturedPacket>,
                             std::move(packet));
  if (!running_.load(std::memory_order_acquire) ||
      !queue.try_push(std::move(ingress))) {
    overflow.store(true, std::memory_order_release);
    audio_queue_drops_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

void VoiceFrontendRuntime::processing_loop(std::stop_token stop) {
  const auto materialize = [](AudioIngressPacket packet) {
    if (auto* replay = std::get_if<AudioPacket>(&packet)) {
      return std::move(*replay);
    }
    return materialize_audio_packet(
        std::move(std::get<RawCapturedPacket>(packet)));
  };
  while (!stop.stop_requested()) {
    drain_capture_events();
    bool worked = false;
    {
      std::scoped_lock lock(pipeline_mutex_);
      worked = drain_audio_assemblies();
    }
    AudioIngressPacket loopback_ingress;
    if (loopback_queue_->try_pop(loopback_ingress)) {
      worked = true;
      AudioPacket loopback;
      try {
        loopback = materialize(std::move(loopback_ingress));
      } catch (const std::exception& error) {
        loopback_overflow_.store(true, std::memory_order_release);
        audio_queue_drops_.fetch_add(1, std::memory_order_relaxed);
        queue_capture_event("capture_conversion_error",
                            std::string("loopback: ") + error.what());
        continue;
      }
      if (loopback_overflow_.exchange(false)) loopback.discontinuity = true;
      latest_loopback_summary_ = summarize(loopback.samples);
      if (recorder_.active()) recorder_.try_enqueue(loopback);
      std::scoped_lock lock(pipeline_mutex_);
      const auto pushed = preprocessor_->PushPacket(loopback);
      bool pipeline_reset_for_packet{};
      if (pushed.reset_required || loopback.discontinuity || loopback.timestamp_error) {
        emit_event("audio_timeline_reset",
                   {{"stream", "loopback"}, {"reason", to_string(pushed.reset_reason)}});
        report_audio_queue_boundary("loopback", pushed.reset_reason);
        // A render-only gap invalidates AEC/ring candidate continuity, but the
        // microphone sample clock is still trustworthy. Keeping the keyword
        // decoder warm avoids starting a fresh KWS stream on the render/AEC
        // transient produced when media playback changes state.
        reset_pipeline(true, false, false);
        pipeline_reset_for_packet = true;
      }
      drain_preprocessor_frames(pipeline_reset_for_packet);
    }
    AudioIngressPacket microphone_ingress;
    if (microphone_queue_->try_pop(microphone_ingress)) {
      worked = true;
      AudioPacket microphone;
      try {
        microphone = materialize(std::move(microphone_ingress));
      } catch (const std::exception& error) {
        microphone_overflow_.store(true, std::memory_order_release);
        audio_queue_drops_.fetch_add(1, std::memory_order_relaxed);
        queue_capture_event("capture_conversion_error",
                            std::string("microphone: ") + error.what());
        continue;
      }
      if (microphone_overflow_.exchange(false)) microphone.discontinuity = true;
      process_microphone(std::move(microphone));
    }
    if (!worked) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void VoiceFrontendRuntime::process_microphone(AudioPacket packet) {
  latest_microphone_summary_ = summarize(packet.samples);
  if (recorder_.active()) recorder_.try_enqueue(packet);
  std::scoped_lock lock(pipeline_mutex_);
  const auto keyword_pushed = keyword_preprocessor_->PushPacket(packet);
  if (keyword_pushed.reset_required || packet.discontinuity ||
      packet.timestamp_error) {
    microphone_kws_->reset(ring_ ? ring_->tail() : 0);
    pending_microphone_keywords_.clear();
  }
  drain_keyword_preprocessor_frames();

  const auto pushed = preprocessor_->PushPacket(packet);
  bool pipeline_reset_for_packet{};
  if (pushed.reset_required || packet.discontinuity || packet.timestamp_error) {
    emit_event("audio_timeline_reset",
               {{"stream", "microphone"}, {"reason", to_string(pushed.reset_reason)}});
    report_audio_queue_boundary("microphone", pushed.reset_reason);
    reset_pipeline(true, false);
    pipeline_reset_for_packet = true;
  }
  drain_preprocessor_frames(pipeline_reset_for_packet);
}

void VoiceFrontendRuntime::drain_keyword_preprocessor_frames() {
  NormalizedFrame frame;
  while (keyword_preprocessor_->TryPopFrame(frame)) {
    if (frame.discontinuity) continue;
    if (microphone_ring_) {
      microphone_ring_->push(frame.first_sample, frame.samples);
    }
    if (auto hit = microphone_kws_->accept(frame)) {
      pending_microphone_keywords_.push_back(std::move(*hit));
      constexpr std::size_t maximum_pending_hits = 8;
      if (pending_microphone_keywords_.size() > maximum_pending_hits) {
        pending_microphone_keywords_.pop_front();
      }
    }
  }
}

void VoiceFrontendRuntime::drain_preprocessor_frames(
    bool pipeline_reset_for_packet) {
  NormalizedFrame frame;
  while (preprocessor_->TryPopFrame(frame)) {
    if (frame.discontinuity) {
      // Preserve the diagnostic stream, but never feed an uncertain 10 ms
      // interval to KWS/VAD/segmentation. PushPacket may already have reset the
      // generation for this packet, so do not advance it a second time.
      if (recorder_.active()) recorder_.try_enqueue(frame);
      if (!pipeline_reset_for_packet) {
        // A processed-only gap is the signature of the AEC failure observed
        // when media render changes state. Preserve the independent microphone
        // ring so a later microphone-only wake can use uncorrupted candidate
        // PCM for its authoritative exact-final decode.
        microphone_candidate_fallback_after_sample_ = frame.first_sample;
        emit_event("audio_timeline_reset",
                   {{"stream", "processed"},
                    {"reason", "normalized_frame_discontinuity"}},
                   frame.first_sample,
                   replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
        report_audio_queue_boundary("processed",
                                    PreprocessResetReason::discontinuity);
        // A processed-only boundary can be emitted after a render reset. The
        // microphone path already performs a full keyword reset when its own
        // clock is discontinuous, so do not clear KWS a second time here.
        reset_pipeline(true, false, false);
        pipeline_reset_for_packet = true;
      }
      continue;
    }
    process_frame(frame, latest_microphone_summary_, latest_loopback_summary_);
  }
}

void VoiceFrontendRuntime::report_audio_queue_boundary(
    const char* trigger_stream, PreprocessResetReason reason) {
  // Packets after an explicit discontinuity remain ordered future audio. If
  // they are drained here, their producer-side sequence numbers are skipped;
  // the first surviving packet then creates another sequence-gap reset, which
  // can turn a single device glitch into a self-sustaining reset loop. Keep the
  // bounded queues intact and let the state reset/generation bump establish
  // the trust boundary. A genuinely full queue is already reported by the
  // producer and its first subsequently accepted packet carries discontinuity.
  // Keeping queued packets also makes unlimited-speed replay independent of
  // worker scheduling. The compatibility metric therefore stays at zero.
  emit_event("audio_queue_reset_boundary",
             {{"trigger_stream", trigger_stream},
              {"reason", to_string(reason)},
              {"microphone_discarded", 0},
              {"loopback_discarded", 0},
              {"microphone_remaining",
               microphone_queue_ ? microphone_queue_->size() : 0},
              {"loopback_remaining",
               loopback_queue_ ? loopback_queue_->size() : 0},
              {"preserved_after_boundary", true}});
}

void VoiceFrontendRuntime::process_frame(const NormalizedFrame& frame,
                                         const SignalSummary& microphone,
                                         const SignalSummary& loopback) {
  if (!view_sample_origin_) view_sample_origin_ = frame.first_sample;
  ring_->push(frame.first_sample, frame.samples);
  processed_frames_.fetch_add(1, std::memory_order_relaxed);
  if (recorder_.active()) recorder_.try_enqueue(frame);

  if (asr_) {
    const auto asr_state = asr_->state();
    if (asr_state != last_published_asr_state_) {
      last_published_asr_state_ = asr_state;
      emit_event("asr_status",
                 {{"state", to_string(asr_state)}, {"status", asr_->status()}},
                 frame.first_sample,
                 replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    }
  }

  const auto vad_update = vad_->accept(frame);
  const bool vad_started = vad_update.speech && !vad_speech_active_;
  vad_speech_active_ = vad_update.speech;
  const auto frame_end = frame.first_sample + frame.samples.size();
  const auto vad_transition_sample = vad_update.speech
                                         ? frame.first_sample
                                         : (!vad_update.completed.empty()
                                                ? vad_update.completed.back().span.end
                                                : frame_end);
  const bool capture_blocked =
      announcement_capture_blocked(vad_transition_sample);
  if (!capture_blocked &&
      announcement_kws_reset_pending_.exchange(false,
                                                std::memory_order_acq_rel)) {
    kws_->reset(frame_end);
    microphone_kws_->reset(frame_end);
    pending_microphone_keywords_.clear();
    last_runtime_keyword_.clear();
    last_runtime_keyword_sample_ = 0;
  }
  if (capture_blocked && vad_started) {
    suppress_current_vad_interval_ = true;
  }
  if (auto start = segmenter_->set_vad_state(
          vad_update.speech, vad_transition_sample,
          !capture_blocked && !suppress_current_vad_interval_)) {
    followup_turns_.fetch_add(1, std::memory_order_relaxed);
    begin_recognition(*start);
  }
  for (const auto& interval : vad_update.completed) {
    if (!suppress_current_vad_interval_) segmenter_->add_vad_interval(interval);
  }
  if (!vad_update.speech && !vad_update.completed.empty()) {
    suppress_current_vad_interval_ = false;
  }

  const bool processed_speech_present =
      vad_update.speech || !vad_update.completed.empty();
  if (auto hit = kws_->accept(frame)) {
    process_keyword_hit(std::move(*hit), capture_blocked,
                        processed_speech_present, "aec");
  }
  while (!pending_microphone_keywords_.empty() &&
         pending_microphone_keywords_.front().detected_at_sample <= frame_end) {
    auto hit = std::move(pending_microphone_keywords_.front());
    pending_microphone_keywords_.pop_front();
    process_keyword_hit(std::move(hit), capture_blocked,
                        processed_speech_present, "microphone_fallback");
  }

  feed_recognizers(frame);

  auto segmented = segmenter_->advance_for_assembly(
      frame.first_sample + frame.samples.size());
  for (auto& request : segmented.assemblies) {
    const auto utterance_id = request.candidate.utterance_id;
    const auto origin = request.candidate.origin;
    const auto activation_id = request.candidate.activation_id;
    const auto turn_index = request.candidate.turn_index;
    const auto trigger_sample = request.candidate.trigger_sample;
    const bool use_microphone_audio =
        microphone_audio_activations_.contains(activation_id);
    auto* assembler = use_microphone_audio
                          ? microphone_candidate_assembler_.get()
                          : candidate_assembler_.get();
    if (assembler && assembler->try_submit(std::move(request))) {
      emit_event("candidate_assembly_queued",
                 {{"utterance_id", utterance_id},
                  {"origin", to_string(origin)},
                   {"activation_id", activation_id},
                   {"turn_index", turn_index},
                   {"trigger_sample", trigger_sample},
                  {"recognition_audio_source",
                   use_microphone_audio ? "microphone" : "aec"},
                  {"outstanding", assembler->outstanding()}},
                 frame.first_sample, replay_mode_ ? "replay" : "live");
    } else {
      constexpr auto reason = "candidate assembly queue is full";
      assembly_dropped_.fetch_add(1, std::memory_order_relaxed);
      rejections_.fetch_add(1, std::memory_order_relaxed);
      emit_event("candidate_rejected",
                 {{"utterance_id", utterance_id},
                  {"origin", to_string(origin)},
                   {"activation_id", activation_id},
                   {"turn_index", turn_index},
                   {"trigger_sample", trigger_sample},
                  {"reason", reason}},
                 frame.first_sample, replay_mode_ ? "replay" : "live");
      abandon_followup(origin, activation_id, turn_index, frame_end, reason);
      cancel_recognition(utterance_id, reason);
    }
  }
  for (const auto& rejection : segmented.rejections) {
    rejections_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate_rejected", {{"utterance_id", rejection.utterance_id},
                                       {"origin", to_string(rejection.origin)},
                                       {"activation_id", rejection.activation_id},
                                       {"turn_index", rejection.turn_index},
                                       {"trigger_sample", rejection.trigger_sample},
                                       {"reason", rejection.reason}},
               rejection.end_sample,
               replay_mode_ ? "replay" : "live");
    abandon_followup(rejection.origin, rejection.activation_id,
                     rejection.turn_index, rejection.end_sample,
                     rejection.reason);
    cancel_recognition(rejection.utterance_id, rejection.reason);
  }
  emit_activation_events(std::move(segmented.activation_events));
  emit_activation_state_if_changed(
      frame_end, replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");

  if (telemetry_enabled_.load(std::memory_order_relaxed) &&
      frame.first_sample >= next_telemetry_sample_) {
    TelemetrySample telemetry;
    telemetry.sample = frame.first_sample;
    telemetry.view_sample = frame.first_sample - *view_sample_origin_;
    telemetry.microphone = microphone;
    telemetry.loopback = loopback;
    telemetry.processed = summarize(frame);
    telemetry.vad = vad_update.speech;
    telemetry.replay = replay_mode_.load(std::memory_order_relaxed);
    telemetry.audio_queue_depth = microphone_queue_->size();
    if (asr_) {
      telemetry.asr_queue_depth = asr_->queue_depth();
      telemetry.asr_pending_requests = asr_->pending_requests();
      telemetry.asr_pending_audio_samples = asr_->pending_audio_samples();
      telemetry.asr_state = asr_->state();
    }
    telemetry.assembly_outstanding =
        candidate_assembler_ ? candidate_assembler_->outstanding() : 0;
    telemetry.action_queue_depth =
        action_executor_ ? action_executor_->queue_size() : 0;
    telemetry.announcement_queue_depth =
        announcement_dispatcher_ ? announcement_dispatcher_->queue_size() : 0;
    telemetry.activation = segmenter_->activation();
    telemetry.aec = preprocessor_->Diagnostics();
    telemetry.debug_dropped = debug_.dropped();
    telemetry.audio_queue_drops =
        audio_queue_drops_.load(std::memory_order_relaxed);
    telemetry.audio_reset_backlog_drops =
        audio_reset_backlog_drops_.load(std::memory_order_relaxed);
    if (!telemetry_queue_ || !telemetry_queue_->try_push(std::move(telemetry))) {
      telemetry_queue_drops_.fetch_add(1, std::memory_order_relaxed);
    }
    const auto interval = std::max<std::uint64_t>(kFrameSamples,
        kProcessingSampleRate / std::max<std::uint32_t>(1, config_.web.telemetry_hz));
    next_telemetry_sample_ = frame.first_sample + interval;
  }
}

void VoiceFrontendRuntime::process_keyword_hit(
    KwsHit hit, bool capture_blocked, bool processed_speech_present,
    std::string_view detector) {
  const auto debounce_samples =
      static_cast<std::uint64_t>(kProcessingSampleRate) / 2;
  if (!last_runtime_keyword_.empty() &&
      last_runtime_keyword_ == hit.keyword &&
      hit.detected_at_sample <
          last_runtime_keyword_sample_ + debounce_samples) {
    return;
  }
  if (detector == "microphone_fallback" && !processed_speech_present) {
    emit_event("kws_suppressed",
               {{"reason", "microphone_fallback_without_processed_speech"},
                {"keyword", hit.keyword},
                {"detector", detector}},
               hit.detected_at_sample,
               replay_mode_ ? "replay" : "live");
    return;
  }

  last_runtime_keyword_ = hit.keyword;
  last_runtime_keyword_sample_ = hit.detected_at_sample;
  const auto guard =
      static_cast<std::uint64_t>(config_.segmentation.wake_guard_ms) *
      kProcessingSampleRate / 1000;
  hit.wake_span.start =
      hit.wake_span.start > guard ? hit.wake_span.start - guard : 0;
  // Do not extend the wake interval into following speech. In a one-breath
  // "wake word + command" utterance, even a 100 ms tail guard can remove the
  // initial consonant of the command. Residual wake text is stripped by the
  // command parser instead of sacrificing command audio.
  auto payload = kws_json(hit);
  payload["detector"] = detector;
  kws_hits_.fetch_add(1, std::memory_order_relaxed);
  emit_event("kws_hit", std::move(payload), hit.detected_at_sample,
             replay_mode_ ? "replay" : "live");
  if (capture_blocked || suppress_current_vad_interval_) {
    emit_event("kws_suppressed",
               {{"reason", "announcement_playback_guard"},
                {"keyword", hit.keyword},
                {"detector", detector}},
               hit.detected_at_sample,
               replay_mode_ ? "replay" : "live");
    return;
  }

  const bool rearming = segmenter_->activation().active();
  const bool use_microphone_audio =
      detector == "microphone_fallback" &&
      microphone_candidate_fallback_after_sample_ != 0 &&
      hit.detected_at_sample >= microphone_candidate_fallback_after_sample_;
  if (auto start = segmenter_->add_kws_hit(std::move(hit))) {
    if (use_microphone_audio) {
      microphone_audio_activations_.insert(start->activation_id);
      emit_event("recognition_audio_fallback",
                 {{"activation_id", start->activation_id},
                  {"reason", "aec_processed_stream_discontinuity"},
                  {"audio_source", "microphone"}},
                 start->trigger_sample,
                 replay_mode_ ? "replay" : "live");
    }
    emit_activation_events(segmenter_->take_activation_events());
    if (rearming) {
      if (candidate_assembler_) candidate_assembler_->cancel_pending();
      if (microphone_candidate_assembler_) {
        microphone_candidate_assembler_->cancel_pending();
      }
      cancel_recognition_generation("keyword activation rearmed");
    }
    begin_recognition(*start);
  }
}

bool VoiceFrontendRuntime::drain_audio_assemblies() {
  bool drained{};
  const auto drain = [this, &drained](CandidateAssembler* assembler) {
    if (!assembler) return;
    AudioAssemblyResult result;
    while (assembler->try_pop(result)) {
      drained = true;
      std::visit(
          [this](auto value) {
            using Result = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Result, BackfillAssemblyResult>) {
              complete_backfill(std::move(value));
            } else {
              complete_candidate(std::move(value));
            }
          },
          std::move(result));
    }
  };
  drain(candidate_assembler_.get());
  drain(microphone_candidate_assembler_.get());
  return drained;
}

void VoiceFrontendRuntime::complete_backfill(BackfillAssemblyResult result) {
  const auto pending_it = pending_recognition_starts_.find(result.utterance_id);
  if (pending_it == pending_recognition_starts_.end()) {
    erase_command_snapshot(result.utterance_id, result.recognition_generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                              result.utterance_id, result.recognition_generation,
                              result.stream_start_sample, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              result.rejection.empty()
                                  ? "recognition start no longer pending"
                                  : result.rejection,
                              result.origin, result.activation_id,
                              result.turn_index, result.trigger_sample});
    emit_event("asr_backfill_stale",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(result.origin)},
                {"activation_id", result.activation_id},
                {"turn_index", result.turn_index},
                {"trigger_sample", result.trigger_sample},
                {"reason", result.rejection.empty()
                               ? "recognition start no longer pending"
                               : result.rejection}},
               result.stream_start_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }
  auto pending = std::move(pending_it->second);
  pending_recognition_starts_.erase(pending_it);

  if (!result.rejection.empty() ||
      result.recognition_generation != pending.generation ||
      pending.generation != recognition_generation_.load(std::memory_order_acquire)) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    if (result.recognition_generation != pending.generation ||
        pending.generation !=
            recognition_generation_.load(std::memory_order_acquire)) {
      erase_command_snapshot(result.utterance_id, pending.generation);
    }
    if (result.recognition_generation != pending.generation ||
        pending.generation !=
            recognition_generation_.load(std::memory_order_acquire)) {
      emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                                result.utterance_id, pending.generation,
                                result.stream_start_sample, 0,
                                replay_mode_.load(std::memory_order_acquire)
                                     ? "replay"
                                     : "live",
                                "stale recognition generation", pending.origin,
                                pending.activation_id, pending.turn_index,
                                pending.trigger_sample});
    }
    emit_event("asr_backfill_rejected",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(pending.origin)},
                {"activation_id", pending.activation_id},
                {"turn_index", pending.turn_index},
                {"reason", result.rejection.empty() ? "stale recognition generation"
                                                     : result.rejection}},
               result.stream_start_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  const auto recognizer_state = asr_ ? asr_->state()
                                     : StreamingRecognizerState::unavailable;
  if (!asr_ || recognizer_state == StreamingRecognizerState::unavailable ||
      recognizer_state == StreamingRecognizerState::stopped) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_unavailable",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(pending.origin)},
                {"activation_id", pending.activation_id},
                {"turn_index", pending.turn_index},
                {"state", to_string(recognizer_state)},
                {"status", asr_ ? asr_->status() : "recognizer unavailable"}},
               result.stream_start_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  auto backfill = std::make_shared<std::vector<float>>(std::move(result.pcm));
  RecognitionBegin begin;
  begin.utterance_id = result.utterance_id;
  begin.generation = pending.generation;
  begin.source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  begin.backfill_first_sample = backfill->empty() ? pending.stream_start_sample
                                                  : result.first_sample;
  begin.backfill = std::move(backfill);
  const auto submitted = asr_->try_begin(std::move(begin));
  if (submitted != RecognitionSubmitStatus::accepted) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_overloaded",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(pending.origin)},
                {"activation_id", pending.activation_id},
                {"turn_index", pending.turn_index},
                {"status", to_string(submitted)}},
               result.stream_start_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  ActiveRecognitionFeed active;
  active.generation = pending.generation;
  active.stream_start_sample = pending.stream_start_sample;
  active.origin = pending.origin;
  active.activation_id = pending.activation_id;
  active.turn_index = pending.turn_index;
  active.trigger_sample = pending.trigger_sample;
  active.pending_first_sample = pending.pending_first_sample;
  active.chunk_samples = pending.chunk_samples;
  active.pending = std::move(pending.pending);
  active.pending.reserve(std::max(active.pending.capacity(), active.chunk_samples * 2));
  active_recognizers_.insert_or_assign(result.utterance_id, std::move(active));
  emit_event("asr_queued",
             {{"utterance_id", result.utterance_id},
              {"generation", pending.generation},
              {"origin", to_string(pending.origin)},
              {"activation_id", pending.activation_id},
              {"turn_index", pending.turn_index},
              {"trigger_sample", pending.trigger_sample},
              {"state", to_string(recognizer_state)},
              {"command_config_revision", pending.command_config_revision},
              {"backfill_spans", pending.backfill_span_count},
              {"backfill_truncated", result.truncated}},
             result.stream_start_sample,
             replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
}

void VoiceFrontendRuntime::complete_candidate(CandidateAssemblyResult result) {
  if (!result.candidate) {
    rejections_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate_rejected",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(result.origin)},
                {"activation_id", result.activation_id},
                {"turn_index", result.turn_index},
                {"trigger_sample", result.trigger_sample},
                {"reason", result.rejection}},
               result.trigger_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    abandon_followup(result.origin, result.activation_id, result.turn_index,
                     result.end_sample, result.rejection);
    cancel_recognition(result.utterance_id, result.rejection);
    return;
  }

  candidates_.fetch_add(1, std::memory_order_relaxed);
  auto shared = std::make_shared<const UtteranceCandidate>(
      std::move(*result.candidate));
  const auto timestamp = candidate_end_sample(*shared);
  emit_event("candidate", candidate_json(*shared), timestamp,
             replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
  if (recorder_.active()) recorder_.try_enqueue(RecordCandidate{shared});
  finalize_recognition(std::move(shared));
}

void VoiceFrontendRuntime::begin_recognition(const UtteranceStart& start) {
  const auto generation = recognition_generation_.load(std::memory_order_acquire);
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  emit_utterance_lifecycle({UtteranceLifecyclePhase::begin, start.utterance_id,
                            generation, start.stream_start_sample, 0, source,
                            start.origin == UtteranceOrigin::keyword
                                ? "keyword-associated utterance created"
                                : "activation follow-up utterance created",
                            start.origin, start.activation_id, start.turn_index,
                            start.trigger_sample});
  if (!asr_ || !config_.asr.enabled) {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, start.utterance_id,
                              generation, start.stream_start_sample, 0, source,
                              "streaming recognition is disabled", start.origin,
                              start.activation_id, start.turn_index,
                              start.trigger_sample});
    return;
  }
  UtteranceCommandSnapshot command_snapshot;
  {
    std::scoped_lock command_lock(command_mutex_);
    command_snapshot = {command_parser_, command_config_revision_, commands_enabled_,
                        start.origin, start.activation_id, start.turn_index,
                        start.trigger_sample};
    const auto key = command_snapshot_key(start.utterance_id, generation);
    completed_command_snapshots_.erase(key);
    std::erase(completed_command_snapshot_order_, key);
    command_snapshots_.insert_or_assign(key, command_snapshot);
  }
  const auto recognizer_state = asr_->state();
  if (recognizer_state == StreamingRecognizerState::unavailable ||
      recognizer_state == StreamingRecognizerState::stopped) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_unavailable", {{"utterance_id", start.utterance_id},
                                    {"origin", to_string(start.origin)},
                                    {"activation_id", start.activation_id},
                                    {"turn_index", start.turn_index},
                                    {"state", to_string(recognizer_state)},
                                    {"status", asr_->status()}},
               start.stream_start_sample,
               replay_mode_ ? "replay" : "live");
    return;
  }
  const auto starting_streams = active_recognizers_.size() +
                                pending_recognition_starts_.size();
  if (starting_streams >= config_.asr.max_active_streams) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_stream_rejected",
               {{"utterance_id", start.utterance_id},
                {"origin", to_string(start.origin)},
                {"activation_id", start.activation_id},
                {"turn_index", start.turn_index},
                {"reason", "active_stream_limit"},
                {"active_streams", starting_streams},
                {"max_active_streams", config_.asr.max_active_streams}},
               start.stream_start_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  BackfillAssemblyRequest request;
  request.utterance_id = start.utterance_id;
  request.origin = start.origin;
  request.activation_id = start.activation_id;
  request.turn_index = start.turn_index;
  request.trigger_sample = start.trigger_sample;
  request.source_spans = start.backfill_spans;
  request.trailing_silence_samples = 0;
  request.stream_start_sample = start.stream_start_sample;
  request.recognition_generation = generation;
  if (!candidate_assembler_ || !candidate_assembler_->try_submit(std::move(request))) {
    assembly_dropped_.fetch_add(1, std::memory_order_relaxed);
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_backfill_rejected",
               {{"utterance_id", start.utterance_id},
                {"origin", to_string(start.origin)},
                {"activation_id", start.activation_id},
                {"turn_index", start.turn_index},
                {"reason", "audio assembly queue is full"}},
               start.stream_start_sample,
               replay_mode_ ? "replay" : "live");
    return;
  }

  PendingRecognitionStart pending;
  pending.generation = generation;
  pending.stream_start_sample = start.stream_start_sample;
  pending.origin = start.origin;
  pending.activation_id = start.activation_id;
  pending.turn_index = start.turn_index;
  pending.trigger_sample = start.trigger_sample;
  pending.chunk_samples = std::max<std::size_t>(
      kFrameSamples, static_cast<std::size_t>(config_.asr.feed_chunk_ms) *
                         kProcessingSampleRate / 1000);
  const auto pending_limit =
      static_cast<std::size_t>(config_.asr.max_pending_audio_ms) *
      kProcessingSampleRate / 1000;
  pending.pending.reserve(std::min(pending_limit, pending.chunk_samples * 2));
  pending.command_config_revision = command_snapshot.config_revision;
  pending.backfill_span_count = start.backfill_spans.size();
  pending_recognition_starts_.insert_or_assign(start.utterance_id, pending);
  emit_event("asr_backfill_queued",
             {{"utterance_id", start.utterance_id},
              {"generation", generation},
              {"origin", to_string(start.origin)},
              {"activation_id", start.activation_id},
              {"turn_index", start.turn_index},
              {"trigger_sample", start.trigger_sample},
              {"command_config_revision", command_snapshot.config_revision},
              {"backfill_spans", start.backfill_spans.size()},
              {"assembly_outstanding", candidate_assembler_->outstanding()}},
             start.stream_start_sample, replay_mode_ ? "replay" : "live");
}

void VoiceFrontendRuntime::feed_recognizers(const NormalizedFrame& frame) {
  if (!asr_ || (active_recognizers_.empty() &&
                pending_recognition_starts_.empty())) return;
  const auto frame_end = frame.first_sample + frame.samples.size();

  const auto pending_limit = static_cast<std::size_t>(config_.asr.max_pending_audio_ms) *
                             kProcessingSampleRate / 1000;
  for (auto& [utterance_id, pending] : pending_recognition_starts_) {
    if (frame_end <= pending.stream_start_sample) continue;
    const auto start_sample = std::max(frame.first_sample, pending.stream_start_sample);
    const auto offset = static_cast<std::size_t>(start_sample - frame.first_sample);
    const auto incoming = frame.samples.size() - offset;
    if (pending.pending.size() + incoming > pending_limit) {
      asr_dropped_.fetch_add(1, std::memory_order_relaxed);
      emit_event("asr_overloaded",
                 {{"utterance_id", utterance_id},
                  {"origin", to_string(pending.origin)},
                  {"activation_id", pending.activation_id},
                  {"turn_index", pending.turn_index},
                  {"trigger_sample", pending.trigger_sample},
                  {"status", "pending backfill audio limit reached"}},
                 frame.first_sample, replay_mode_ ? "replay" : "live");
      cancel_recognition_generation("ASR backfill assembly did not keep up");
      return;
    }
    if (pending.pending.empty()) pending.pending_first_sample = start_sample;
    pending.pending.insert(
        pending.pending.end(),
        frame.samples.begin() + static_cast<std::ptrdiff_t>(offset),
        frame.samples.end());
  }

  bool cancel_generation_for_overload{};
  for (auto it = active_recognizers_.begin(); it != active_recognizers_.end();) {
    auto& state = it->second;
    if (frame_end <= state.stream_start_sample) {
      ++it;
      continue;
    }
    const auto start_sample = std::max(frame.first_sample, state.stream_start_sample);
    const auto offset = static_cast<std::size_t>(start_sample - frame.first_sample);
    if (state.pending.empty()) state.pending_first_sample = start_sample;
    state.pending.insert(state.pending.end(), frame.samples.begin() +
        static_cast<std::ptrdiff_t>(offset), frame.samples.end());

    bool failed = false;
    while (state.pending.size() >= state.chunk_samples) {
      auto pcm = std::make_shared<std::vector<float>>(
          state.pending.begin(),
          state.pending.begin() + static_cast<std::ptrdiff_t>(state.chunk_samples));
      RecognitionChunk chunk{it->first, state.generation, state.pending_first_sample,
                             std::move(pcm), kProcessingSampleRate};
      const auto status = asr_->try_accept(std::move(chunk));
      if (status != RecognitionSubmitStatus::accepted) {
        emit_event("asr_overloaded", {{"utterance_id", it->first},
                                       {"origin", to_string(state.origin)},
                                       {"activation_id", state.activation_id},
                                       {"turn_index", state.turn_index},
                                       {"trigger_sample", state.trigger_sample},
                                       {"status", to_string(status)}}, frame.first_sample,
                   replay_mode_ ? "replay" : "live");
        asr_dropped_.fetch_add(1, std::memory_order_relaxed);
        failed = true;
        break;
      }
      emit_utterance_lifecycle({UtteranceLifecyclePhase::chunk, it->first,
                                state.generation, state.pending_first_sample,
                                 state.chunk_samples,
                                 replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                                 "streaming ASR audio chunk accepted",
                                 state.origin, state.activation_id,
                                 state.turn_index, state.trigger_sample});
      state.pending.erase(state.pending.begin(),
                          state.pending.begin() +
                              static_cast<std::ptrdiff_t>(state.chunk_samples));
      state.pending_first_sample += state.chunk_samples;
    }
    if (failed) {
      cancel_generation_for_overload = true;
      break;
    } else {
      ++it;
    }
  }
  if (cancel_generation_for_overload) {
    // A control request sent through the same full queue is not a reliable
    // cancellation mechanism. The generation watermark is lock-free and also
    // suppresses a final which may already be decoding.
    cancel_recognition_generation("ASR request queue or pending-audio limit reached");
  }
}

void VoiceFrontendRuntime::finalize_recognition(
    std::shared_ptr<const UtteranceCandidate> candidate) {
  if (!candidate) return;
  const auto utterance_id = candidate->utterance_id;
  if (!asr_) {
    const auto generation = recognition_generation_.load(std::memory_order_acquire);
    abandon_followup(candidate->origin, candidate->activation_id,
                     candidate->turn_index, candidate_end_sample(*candidate),
                     "recognizer unavailable at finalize");
    erase_command_snapshot(utterance_id, generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              generation, candidate_end_sample(*candidate), 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              "recognizer unavailable at finalize",
                              candidate->origin, candidate->activation_id,
                              candidate->turn_index, candidate->trigger_sample});
    return;
  }
  const auto it = active_recognizers_.find(utterance_id);
  const auto generation = it == active_recognizers_.end()
                              ? recognition_generation_.load(std::memory_order_acquire)
                              : it->second.generation;
  active_recognizers_.erase(utterance_id);
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  // Do not spend the last queue slot on a provisional tail: exact-final starts
  // a fresh stream from candidate PCM and is the only executable transcript.
  const auto status = asr_->try_finalize(
      {utterance_id, generation, source, candidate});
  if (status != RecognitionSubmitStatus::accepted) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    emit_event("asr_finalize_dropped", {{"utterance_id", utterance_id},
                                         {"origin", to_string(candidate->origin)},
                                         {"activation_id", candidate->activation_id},
                                         {"turn_index", candidate->turn_index},
                                         {"trigger_sample", candidate->trigger_sample},
                                         {"status", to_string(status)}},
               candidate_end_sample(*candidate),
               source);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              generation, candidate_end_sample(*candidate),
                              candidate->pcm.size(), source,
                              "exact-final submission rejected: " +
                                  std::string(to_string(status)),
                              candidate->origin, candidate->activation_id,
                              candidate->turn_index, candidate->trigger_sample});
    if (status == RecognitionSubmitStatus::queue_full ||
        status == RecognitionSubmitStatus::pending_audio_full) {
      cancel_recognition_generation("ASR exact-final submission overloaded");
    } else {
      erase_command_snapshot(utterance_id, generation);
    }
    abandon_followup(candidate->origin, candidate->activation_id,
                     candidate->turn_index, candidate_end_sample(*candidate),
                     "exact-final submission rejected: " +
                         std::string(to_string(status)));
  } else {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::finalize, utterance_id,
                              generation, candidate->source_spans.empty()
                                              ? candidate_end_sample(*candidate)
                                              : candidate->source_spans.front().start,
                              candidate->pcm.size(), source,
                              "authoritative exact-final decode queued",
                              candidate->origin, candidate->activation_id,
                              candidate->turn_index, candidate->trigger_sample});
  }
}

void VoiceFrontendRuntime::cancel_recognition(const std::string& utterance_id,
                                               std::string reason) {
  const auto it = active_recognizers_.find(utterance_id);
  const auto pending = pending_recognition_starts_.find(utterance_id);
  const auto generation = it != active_recognizers_.end()
                              ? it->second.generation
                              : (pending != pending_recognition_starts_.end()
                                     ? pending->second.generation
                                     : recognition_generation_.load(
                                           std::memory_order_acquire));
  std::optional<UtteranceCommandSnapshot> command_snapshot;
  {
    const auto key = command_snapshot_key(utterance_id, generation);
    std::scoped_lock command_lock(command_mutex_);
    if (const auto snapshot = command_snapshots_.find(key);
        snapshot != command_snapshots_.end()) {
      command_snapshot = snapshot->second;
    } else if (const auto completed = completed_command_snapshots_.find(key);
               completed != completed_command_snapshots_.end()) {
      command_snapshot = completed->second;
    }
  }
  if (command_snapshot) {
    abandon_followup(command_snapshot->origin, command_snapshot->activation_id,
                     command_snapshot->turn_index,
                     ring_ ? ring_->tail() : 0, reason);
  }
  if (pending != pending_recognition_starts_.end()) {
    pending_recognition_starts_.erase(pending);
  }
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                            generation, 0, 0, source, reason,
                            command_snapshot ? command_snapshot->origin
                                             : UtteranceOrigin::keyword,
                            command_snapshot ? command_snapshot->activation_id
                                             : std::string{},
                            command_snapshot ? command_snapshot->turn_index : 0,
                            command_snapshot ? command_snapshot->trigger_sample : 0});
  if (it == active_recognizers_.end() || !asr_) {
    erase_command_snapshot(utterance_id, generation);
    return;
  }
  const auto status = asr_->try_cancel({utterance_id, generation, reason});
  active_recognizers_.erase(it);
  erase_command_snapshot(utterance_id, generation);
  if (status == RecognitionSubmitStatus::queue_full ||
      status == RecognitionSubmitStatus::pending_audio_full) {
    cancel_recognition_generation("ASR cancellation queue overloaded");
  }
}

void VoiceFrontendRuntime::cancel_recognition_generation(std::string reason) {
  const auto cancelled = recognition_generation_.fetch_add(1, std::memory_order_acq_rel);
  if (asr_) asr_->cancel_generation(cancelled);
  const auto utterance_count = active_recognizers_.size() +
                               pending_recognition_starts_.size();
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  std::vector<UtteranceCommandSnapshot> command_snapshots;
  {
    std::scoped_lock command_lock(command_mutex_);
    command_snapshots.reserve(command_snapshots_.size() +
                              completed_command_snapshots_.size());
    for (const auto& [key, snapshot] : command_snapshots_) {
      static_cast<void>(key);
      command_snapshots.push_back(snapshot);
    }
    for (const auto& [key, snapshot] : completed_command_snapshots_) {
      static_cast<void>(key);
      command_snapshots.push_back(snapshot);
    }
  }
  for (const auto& snapshot : command_snapshots) {
    abandon_followup(snapshot.origin, snapshot.activation_id,
                     snapshot.turn_index, ring_ ? ring_->tail() : 0, reason);
  }
  for (const auto& [utterance_id, state] : active_recognizers_) {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              state.generation, 0, 0, source, reason,
                              state.origin, state.activation_id,
                              state.turn_index, state.trigger_sample});
  }
  for (const auto& [utterance_id, state] : pending_recognition_starts_) {
    if (!active_recognizers_.contains(utterance_id)) {
      emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                                state.generation, 0, 0, source, reason,
                                state.origin, state.activation_id,
                                state.turn_index, state.trigger_sample});
    }
  }
  if (utterance_count != 0) {
    emit_event("asr_generation_cancelled", {{"generation", cancelled},
                                              {"reason", std::move(reason)},
                                              {"utterances", utterance_count}});
  }
  active_recognizers_.clear();
  pending_recognition_starts_.clear();
  {
    std::scoped_lock command_lock(command_mutex_);
    command_snapshots_.clear();
    completed_command_snapshots_.clear();
    completed_command_snapshot_order_.clear();
  }
}

void VoiceFrontendRuntime::erase_command_snapshot(const std::string& utterance_id,
                                                   std::uint64_t generation) {
  const auto key = command_snapshot_key(utterance_id, generation);
  std::scoped_lock command_lock(command_mutex_);
  command_snapshots_.erase(key);
  completed_command_snapshots_.erase(key);
  std::erase(completed_command_snapshot_order_, key);
}

void VoiceFrontendRuntime::emit_utterance_lifecycle(UtteranceLifecycleEvent event) {
  const auto timestamp = event.first_sample;
  const auto source = event.source;
  emit_event("utterance_lifecycle",
             {{"phase", to_string(event.phase)},
              {"utterance_id", event.utterance_id},
              {"generation", event.generation},
               {"origin", to_string(event.origin)},
               {"activation_id", event.activation_id},
               {"turn_index", event.turn_index},
               {"trigger_sample", event.trigger_sample},
               {"first_sample", event.first_sample},
              {"sample_count", event.sample_count},
              {"detail", event.detail}},
             timestamp, source);
}

void VoiceFrontendRuntime::handle_recognition_result(RecognitionResult result) {
  if (result.generation != recognition_generation_.load(std::memory_order_acquire)) return;
  bool emit_partials{};
  {
    std::scoped_lock lock(config_mutex_);
    emit_partials = config_.asr.emit_partials;
  }
  std::optional<UtteranceCommandSnapshot> command_snapshot;
  {
    std::scoped_lock command_lock(command_mutex_);
    const auto key = command_snapshot_key(result.utterance_id, result.generation);
    if (const auto it = command_snapshots_.find(key); it != command_snapshots_.end()) {
      command_snapshot = it->second;
      if (result.kind == RecognitionResultKind::final) {
        const bool already_cached = completed_command_snapshots_.contains(key);
        completed_command_snapshots_.insert_or_assign(key, it->second);
        if (!already_cached) completed_command_snapshot_order_.push_back(key);
        command_snapshots_.erase(it);
        while (completed_command_snapshot_order_.size() >
               kCompletedCommandSnapshotCapacity) {
          completed_command_snapshots_.erase(completed_command_snapshot_order_.front());
          completed_command_snapshot_order_.pop_front();
        }
      } else if (result.kind == RecognitionResultKind::error &&
                 result.exact_final_attempt) {
        command_snapshots_.erase(it);
        completed_command_snapshots_.erase(key);
        std::erase(completed_command_snapshot_order_, key);
      }
    } else if (result.kind == RecognitionResultKind::final) {
      if (const auto completed = completed_command_snapshots_.find(key);
          completed != completed_command_snapshots_.end()) {
        command_snapshot = completed->second;
      }
    }
  }
  const std::string source = result.source == "replay" ? "replay" : "live";
  const auto type = std::string("asr_") + to_string(result.kind);
  const auto origin = result.candidate
                          ? result.candidate->origin
                          : (command_snapshot ? command_snapshot->origin
                                              : UtteranceOrigin::keyword);
  const auto activation_id = result.candidate
                                 ? result.candidate->activation_id
                                 : (command_snapshot
                                        ? command_snapshot->activation_id
                                        : std::string{});
  const auto turn_index = result.candidate
                              ? result.candidate->turn_index
                              : (command_snapshot ? command_snapshot->turn_index : 0);
  const auto trigger_sample = result.candidate
                                  ? result.candidate->trigger_sample
                                  : (command_snapshot
                                         ? command_snapshot->trigger_sample
                                         : 0);
  auto payload = nlohmann::json{{"utterance_id", result.utterance_id},
                                {"origin", to_string(origin)},
                                {"activation_id", activation_id},
                                {"turn_index", turn_index},
                                {"trigger_sample", trigger_sample},
                                {"generation", result.generation},
                                {"revision", result.revision},
                                {"text", result.hypothesis.text},
                                {"tokens", result.hypothesis.tokens},
                                {"is_final", result.kind == RecognitionResultKind::final},
                                {"exact_final_attempt", result.exact_final_attempt},
                                {"exact_final", result.exact_final},
                                {"latency_ms", result.latency_ms},
                                {"inference_ms", result.inference_ms},
                                {"rtf", result.rtf},
                                {"audio_start_sample", result.audio_start_sample},
                                 {"audio_end_sample", result.audio_end_sample},
                                 {"detail", result.detail}};
  if (command_snapshot) {
    payload["command_config_revision"] = command_snapshot->config_revision;
  }
  if (result.kind != RecognitionResultKind::partial || emit_partials) {
    emit_event(type, payload, result.audio_end_sample, source);
  }
  if (result.kind == RecognitionResultKind::partial) {
    asr_partials_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (result.kind == RecognitionResultKind::cancelled) {
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    active_recognizers_.erase(result.utterance_id);
    pending_recognition_starts_.erase(result.utterance_id);
    return;
  }
  if (result.kind == RecognitionResultKind::error) {
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    active_recognizers_.erase(result.utterance_id);
    pending_recognition_starts_.erase(result.utterance_id);
    if (result.exact_final_attempt) {
      const auto resolution_sample = result.candidate
                                         ? candidate_end_sample(*result.candidate)
                                         : (ring_ ? ring_->tail()
                                                  : result.audio_end_sample);
      abandon_followup(origin, activation_id, turn_index,
                       resolution_sample,
                       result.detail.empty() ? "exact-final ASR error"
                                             : result.detail);
    }
    return;
  }
  if (result.kind != RecognitionResultKind::final) return;
  asr_finals_.fetch_add(1, std::memory_order_relaxed);

  CommandParseContext context;
  context.runtime_session_id = runtime_session_id_;
  context.utterance_id = result.utterance_id;
  context.origin = origin;
  context.activation_id = activation_id;
  context.turn_index = turn_index;
  context.trigger_sample = trigger_sample;
  context.wake_position =
      result.candidate && result.candidate->position
          ? to_string(*result.candidate->position)
          : "";
  context.wake_word = result.candidate ? result.candidate->keyword : "";
  context.config_revision =
      command_snapshot ? command_snapshot->config_revision : 0;
  context.recognition_generation = result.generation;
  context.final_revision = result.revision;
  context.timestamp_sample =
      result.candidate ? candidate_end_sample(*result.candidate)
                       : result.audio_end_sample;
  context.source =
      source == "live" ? ExecutionSource::live : ExecutionSource::replay;
  context.execution_mode =
      source == "live" ? ExecutionMode::live : ExecutionMode::dry_run;

  std::string parse_stage{"precondition"};
  bool parse_debug_emitted{};
  const auto reject = [&](std::string reason) {
    if (!parse_debug_emitted && source == "replay") {
      emit_event("parse_debug",
                 parse_debug_failure(result.hypothesis.text, context,
                                     "automatic", parse_stage,
                                     "precondition_failed", reason),
                 result.audio_end_sample, source);
      parse_debug_emitted = true;
    }
    command_rejections_.fetch_add(1, std::memory_order_relaxed);
    auto rejection = nlohmann::json{{"utterance_id", result.utterance_id},
                                    {"origin", to_string(origin)},
                                    {"activation_id", activation_id},
                                    {"turn_index", turn_index},
                                    {"trigger_sample", trigger_sample},
                                    {"text", result.hypothesis.text},
                                    {"reason", reason}};
    if (command_snapshot) {
      rejection["config_revision"] = command_snapshot->config_revision;
    }
    emit_event("command_rejected", std::move(rejection),
               result.audio_end_sample, source);
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    abandon_followup(origin, activation_id, turn_index,
                     result.candidate
                         ? candidate_end_sample(*result.candidate)
                         : result.audio_end_sample,
                     std::move(reason));
  };
  parse_stage = "exact_final";
  if (!result.exact_final || !result.candidate) {
    return reject("result is not an exact final decode");
  }
  parse_stage = "parser_snapshot";
  if (!command_snapshot || !command_snapshot->parser) {
    return reject("command parser snapshot is unavailable");
  }
  if (!command_snapshot->commands_enabled && source != "replay") {
    return reject("command execution is disabled");
  }
  parse_stage = "candidate";
  if (result.candidate->truncated || result.candidate->timed_out ||
      result.candidate->discontinuity) {
    return reject("candidate is truncated, timed out, or discontinuous");
  }

  parse_stage = "parser";
  auto parsed = command_snapshot->parser->parse(result.hypothesis.text, context);
  if (!parsed.ok()) {
    const auto code =
        parsed.error ? parsed.error->code : "rule_not_matched";
    const auto message =
        parsed.error ? parsed.error->message
                     : "text did not match a command rule";
    const auto byte_offset =
        parsed.error ? parsed.error->byte_offset : std::size_t{};
    if (source == "replay") {
      emit_event("parse_debug",
                 parse_debug_failure(result.hypothesis.text, context,
                                     "automatic", parse_stage, code, message,
                                     byte_offset),
                 result.audio_end_sample, source);
      parse_debug_emitted = true;
    }
    return reject(parsed.error ? parsed.error->code + ": " + parsed.error->message
                               : "text did not match a command rule");
  }

  if (source == "replay") {
    emit_event("parse_debug",
               parse_debug_success(result.hypothesis.text, context,
                                   *parsed.plan, "automatic"),
               result.audio_end_sample, source);
    parse_debug_emitted = true;
  }
  auto plan = std::move(*parsed.plan);
  if (!command_snapshot->commands_enabled) {
    return reject("command execution is disabled");
  }
  SubmitResult submitted;
  bool activation_eligible = true;
  {
    // Serialize the final generation check with pipeline resets. Whichever
    // side obtains the frame-boundary lock first defines whether this final is
    // executable or stale.
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    if (result.generation != recognition_generation_.load(std::memory_order_acquire)) return;
    activation_eligible = plan.origin != UtteranceOrigin::followup ||
                          (segmenter_ && segmenter_->followup_plan_eligible(
                                             plan.activation_id,
                                             plan.turn_index));
    if (activation_eligible) {
      submitted = action_executor_
                    ? action_executor_->try_submit(
                          std::move(plan),
                          [this, timestamp = result.audio_end_sample,
                           source](const CommandPlan& accepted) {
                            command_plans_.fetch_add(1, std::memory_order_relaxed);
                            emit_event("command_plan", command_plan_json(accepted), timestamp,
                                       source);
                            for (const auto& action : accepted.actions) {
                              emit_event("action_queued",
                                         {{"command_id", accepted.command_id},
                                          {"utterance_id", accepted.utterance_id},
                                          {"origin", to_string(accepted.origin)},
                                          {"activation_id", accepted.activation_id},
                                          {"turn_index", accepted.turn_index},
                                          {"trigger_sample", accepted.trigger_sample},
                                          {"action_id", action.action_id},
                                          {"sequence", action.sequence},
                                          {"type", to_string(action.type)}},
                                         timestamp, source);
                            }
                            if (segmenter_ && segmenter_->refresh_activation(
                                                  accepted.activation_id,
                                                  accepted.turn_index,
                                                  accepted.timestamp_sample)) {
                              auto events = segmenter_->take_activation_events();
                              const auto refresh_sample = events.empty()
                                                              ? accepted.timestamp_sample
                                                              : events.back().at_sample;
                              emit_activation_events(std::move(events), source);
                              emit_activation_state_if_changed(
                                  refresh_sample, source);
                            }
                          })
                    : SubmitResult{SubmitStatus::stopped,
                                   "action executor unavailable"};
    }
  }
  if (!activation_eligible) {
    return reject("follow-up started outside the validated activation window");
  }
  if (!submitted.accepted()) {
    emit_event("action_submit_failed",
               {{"utterance_id", result.utterance_id},
                {"origin", to_string(origin)},
                {"activation_id", activation_id},
                {"turn_index", turn_index},
                {"status", to_string(submitted.status)},
                {"error", submitted.error}},
               result.audio_end_sample, source);
    return reject("action submit failed: " + submitted.error);
  }
}

void VoiceFrontendRuntime::handle_action_result(const ActionResult& result) {
  const auto status = std::string(to_string(result.status));
  if (result.status == ActionStatus::started) {
    emit_event("action_started", action_result_json(result), result.timestamp_sample,
               to_string(result.source));
    return;
  }
  if (result.status == ActionStatus::succeeded || result.status == ActionStatus::noop ||
      result.status == ActionStatus::dry_run || result.status == ActionStatus::deduplicated) {
    actions_succeeded_.fetch_add(1, std::memory_order_relaxed);
  } else {
    actions_failed_.fetch_add(1, std::memory_order_relaxed);
  }
  emit_event("action_" + status, action_result_json(result), result.timestamp_sample,
             to_string(result.source));
}

void VoiceFrontendRuntime::handle_plan_started(const CommandPlan& plan) {
  AnnouncementsConfig announcements;
  {
    std::scoped_lock config_lock(config_mutex_);
    announcements = config_.announcements;
  }

  auto request = make_plan_announcement(
      plan, plan.command_id + ":plan-started");
  announcements_requested_.fetch_add(1, std::memory_order_relaxed);
  if (!announcements.enabled) {
    auto payload = announcement_request_json(request, "suppressed");
    {
      std::scoped_lock announcement_lock(announcement_state_mutex_);
      (request.source == ExecutionSource::live ? latest_live_announcement_
                                                : latest_replay_announcement_) = payload;
    }
    emit_event("announcement", std::move(payload),
               request.timestamp_sample, to_string(request.source));
    return;
  }

  // This synchronous structured event is the user-visible announcement. It
  // is emitted by the plan-start observer before execute_plan can publish the
  // first action_started result. Backend delivery remains fully asynchronous.
  auto payload = announcement_request_json(request, "requested");
  {
    std::scoped_lock announcement_lock(announcement_state_mutex_);
    (request.source == ExecutionSource::live ? latest_live_announcement_
                                              : latest_replay_announcement_) = payload;
  }
  emit_event("announcement", std::move(payload), request.timestamp_sample,
             to_string(request.source));
  const auto submitted = announcement_dispatcher_
                             ? announcement_dispatcher_->try_submit(request)
                             : AnnouncementSubmitResult{
                                   AnnouncementSubmitStatus::stopped,
                                   "announcement dispatcher unavailable"};
  if (submitted.accepted()) return;

  announcements_dropped_.fetch_add(1, std::memory_order_relaxed);
  auto diagnostic = announcement_request_json(
      request, std::string("dropped:") + to_string(submitted.status));
  diagnostic["error"] = submitted.error;
  {
    std::scoped_lock announcement_lock(announcement_state_mutex_);
    (request.source == ExecutionSource::live ? latest_live_announcement_
                                              : latest_replay_announcement_) = diagnostic;
  }
  emit_event("announcement_dropped", std::move(diagnostic),
             request.timestamp_sample, to_string(request.source));
}

void VoiceFrontendRuntime::handle_announcement_result(
    const AnnouncementResult& result) {
  // The log backend's authoritative audit event was already published
  // synchronously by handle_plan_started. Omitting its instantaneous worker
  // lifecycle keeps replay event order independent of thread scheduling.
  if (result.backend == "log") return;

  if (result.status == AnnouncementStatus::playback_failed) {
    announcements_failed_.fetch_add(1, std::memory_order_relaxed);
  }
  auto lifecycle_payload = announcement_result_json(result);
  {
    std::scoped_lock announcement_lock(announcement_state_mutex_);
    (result.source == ExecutionSource::live ? latest_live_announcement_
                                             : latest_replay_announcement_) =
        lifecycle_payload;
  }
  emit_event(std::string("announcement_") + to_string(result.status),
             std::move(lifecycle_payload), result.timestamp_sample,
             to_string(result.source));

  if (result.source != ExecutionSource::live) return;
  std::scoped_lock pipeline_lock(pipeline_mutex_);
  AnnouncementsConfig announcements;
  {
    // Match config.apply's pipeline -> config lock order so a hot policy
    // update cannot race a playback gate start or terminal transition.
    std::scoped_lock config_lock(config_mutex_);
    announcements = config_.announcements;
  }

  if (result.status == AnnouncementStatus::playback_started) {
    if (!announcements.enabled || announcements.barge_in) return;
    announcement_guard_until_sample_.store(0, std::memory_order_release);
    announcement_kws_reset_pending_.store(false, std::memory_order_release);
    announcement_playback_active_.store(true, std::memory_order_release);
    const auto at_sample = ring_ ? ring_->tail() : result.timestamp_sample;
    announcement_playback_start_sample_ = at_sample;
    // Preserve a turn that already owns a candidate, but do not let an
    // unrelated speech interval which merely began before playback later
    // become a suffix/embedded keyword candidate across TTS audio.
    if (vad_speech_active_ &&
        (!segmenter_ || segmenter_->pending_count() == 0)) {
      suppress_current_vad_interval_ = true;
    }
    if (segmenter_ && segmenter_->begin_activation_playback(at_sample)) {
      emit_activation_state_if_changed(at_sample, "live");
    }
    return;
  }

  // Gate ownership is established at playback_started. A hot barge-in or
  // enabled change must not prevent the matching terminal callback from
  // releasing the segmenter hold.
  if (!announcement_playback_active_.exchange(false,
                                               std::memory_order_acq_rel)) {
    return;
  }
  const auto at_sample = ring_ ? ring_->tail() : result.timestamp_sample;
  if (announcements.enabled && !announcements.barge_in) {
    const auto tail_guard_samples =
        static_cast<std::uint64_t>(announcements.tail_guard_ms) *
        kProcessingSampleRate / 1000;
    announcement_guard_until_sample_.store(at_sample + tail_guard_samples,
                                            std::memory_order_release);
    announcement_kws_reset_pending_.store(true, std::memory_order_release);
  } else {
    announcement_guard_until_sample_.store(0, std::memory_order_release);
    announcement_kws_reset_pending_.store(false, std::memory_order_release);
  }
  const auto playback_samples =
      at_sample >= announcement_playback_start_sample_
          ? at_sample - announcement_playback_start_sample_
          : 0;
  announcement_playback_start_sample_ = 0;
  if (segmenter_) {
    if (segmenter_->finish_activation_playback(playback_samples, at_sample)) {
      emit_activation_events(segmenter_->take_activation_events(), "live");
      emit_activation_state_if_changed(at_sample, "live");
    }
  }
}

void VoiceFrontendRuntime::emit_activation_events(
    std::vector<ActivationTransition> events, std::string source) {
  if (source.empty()) {
    source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  }
  for (auto& event : events) {
    switch (event.kind) {
      case ActivationTransitionKind::started:
        activations_started_.fetch_add(1, std::memory_order_relaxed);
        break;
      case ActivationTransitionKind::refreshed:
        activations_refreshed_.fetch_add(1, std::memory_order_relaxed);
        break;
      case ActivationTransitionKind::expired:
        activations_expired_.fetch_add(1, std::memory_order_relaxed);
        microphone_audio_activations_.erase(event.activation_id);
        break;
      case ActivationTransitionKind::cancelled:
        activations_cancelled_.fetch_add(1, std::memory_order_relaxed);
        microphone_audio_activations_.erase(event.activation_id);
        break;
    }
    const bool active = event.kind == ActivationTransitionKind::started ||
                        event.kind == ActivationTransitionKind::refreshed;
    const auto state = event.kind == ActivationTransitionKind::started
                           ? "keyword_turn"
                           : (event.kind == ActivationTransitionKind::refreshed
                                  ? "armed_idle"
                                  : "dormant");
    emit_event(std::string("activation_") + to_string(event.kind),
               {{"active", active},
                {"state", state},
                {"activation_id", event.activation_id},
                {"turn_index", event.turn_index},
                {"at_sample", event.at_sample},
                {"idle_deadline_sample", event.idle_deadline_sample},
                {"hard_deadline_sample", event.hard_deadline_sample},
                {"reason", event.reason}},
               event.at_sample, source);
  }
}

void VoiceFrontendRuntime::emit_activation_state(std::string source) {
  nlohmann::json state = activation_json(ActivationSnapshot{});
  std::uint64_t timestamp{};
  {
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    if (segmenter_) {
      last_published_activation_state_ = segmenter_->activation();
      state = activation_json(last_published_activation_state_);
    }
    if (ring_) timestamp = ring_->tail();
  }
  if (source.empty()) {
    source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  }
  emit_event("activation_state", std::move(state), timestamp,
             std::move(source));
}

void VoiceFrontendRuntime::emit_activation_state_if_changed(
    std::uint64_t timestamp_sample, const std::string& source) {
  if (!segmenter_) return;
  const auto& current = segmenter_->activation();
  const bool changed =
      current.state != last_published_activation_state_.state ||
      current.activation_id != last_published_activation_state_.activation_id ||
      current.turn_index != last_published_activation_state_.turn_index ||
      current.idle_deadline_sample !=
          last_published_activation_state_.idle_deadline_sample ||
      current.hard_deadline_sample !=
          last_published_activation_state_.hard_deadline_sample ||
      current.playback_hold != last_published_activation_state_.playback_hold;
  if (!changed) return;
  last_published_activation_state_ = current;
  emit_event("activation_state", activation_json(current), timestamp_sample,
             source);
}

void VoiceFrontendRuntime::abandon_followup(
    UtteranceOrigin origin, const std::string& activation_id,
    std::uint32_t turn_index, std::uint64_t at_sample, std::string reason) {
  if (origin != UtteranceOrigin::followup || !segmenter_ ||
      activation_id.empty()) {
    return;
  }
  if (segmenter_->abandon_followup(activation_id, turn_index, at_sample,
                                   std::move(reason))) {
    emit_activation_events(segmenter_->take_activation_events());
    emit_activation_state_if_changed(
        at_sample,
        replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
  }
}

bool VoiceFrontendRuntime::announcement_capture_blocked(
    std::uint64_t at_sample) const {
  {
    std::scoped_lock config_lock(config_mutex_);
    if (!config_.announcements.enabled || config_.announcements.barge_in) {
      return false;
    }
  }
  if (announcement_playback_active_.load(std::memory_order_acquire)) return true;
  const auto guard_until =
      announcement_guard_until_sample_.load(std::memory_order_acquire);
  return guard_until != 0 && at_sample < guard_until;
}

void VoiceFrontendRuntime::reset_pipeline(bool discontinuity,
                                          bool reset_preprocessor,
                                          bool reset_keyword) {
  if (candidate_assembler_) candidate_assembler_->cancel_pending();
  if (microphone_candidate_assembler_) {
    microphone_candidate_assembler_->cancel_pending();
  }
  const auto next = ring_ ? ring_->tail() : 0;
  if (ring_) ring_->reset(next);
  if (preprocessor_ && discontinuity && reset_preprocessor) preprocessor_->reset();
  if (keyword_preprocessor_ && reset_keyword) keyword_preprocessor_->reset();
  if (vad_) vad_->reset();
  if (reset_keyword) {
    if (microphone_ring_) {
      microphone_ring_->reset(microphone_ring_->tail());
    }
    if (kws_) kws_->reset(next);
    if (microphone_kws_) microphone_kws_->reset(next);
    pending_microphone_keywords_.clear();
    last_runtime_keyword_.clear();
    last_runtime_keyword_sample_ = 0;
    microphone_candidate_fallback_after_sample_ = 0;
    microphone_audio_activations_.clear();
  }
  if (segmenter_) {
    segmenter_->reset(discontinuity);
    emit_activation_events(segmenter_->take_activation_events());
    emit_activation_state_if_changed(
        next, replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
  }
  announcement_playback_active_.store(false, std::memory_order_release);
  announcement_guard_until_sample_.store(0, std::memory_order_release);
  announcement_kws_reset_pending_.store(false, std::memory_order_release);
  announcement_playback_start_sample_ = 0;
  vad_speech_active_ = false;
  suppress_current_vad_interval_ = false;
  if (discontinuity) {
    cancel_recognition_generation("audio pipeline reset");
    discontinuities_.fetch_add(1, std::memory_order_relaxed);
  }
}

void VoiceFrontendRuntime::emit_event(std::string type, nlohmann::json payload,
                                      std::uint64_t timestamp_sample, std::string source) {
  if (type != "telemetry" &&
      benchmark_capture_enabled_.load(std::memory_order_acquire)) {
    std::scoped_lock lock(benchmark_events_mutex_);
    if (benchmark_capture_enabled_.load(std::memory_order_relaxed)) {
      benchmark_events_.push_back({{"type", type},
                                   {"timestamp_sample", timestamp_sample},
                                   {"source", source},
                                   {"payload", payload}});
    }
  }
  if (recorder_.active() && type != "telemetry") {
    recorder_.try_enqueue(RecordEvent{{{"type", type}, {"timestamp_sample", timestamp_sample},
                                       {"source", source}, {"payload", payload}}});
  }
  debug_.publish(std::move(type), std::move(payload), timestamp_sample, std::move(source));
}

void VoiceFrontendRuntime::queue_capture_event(std::string type, std::string message) {
  std::scoped_lock lock(capture_events_mutex_);
  const auto separator = message.find(':');
  const auto stream = separator == std::string::npos ? std::string("unknown")
                                                      : message.substr(0, separator);
  capture_capabilities_[stream] = {{"last_event", type}, {"message", message}};
  capture_events_.emplace_back(std::move(type), std::move(message));
}

void VoiceFrontendRuntime::drain_capture_events() {
  std::deque<std::pair<std::string, std::string>> events;
  {
    std::scoped_lock lock(capture_events_mutex_);
    events.swap(capture_events_);
  }
  for (auto& [type, message] : events) emit_event(std::move(type), {{"message", std::move(message)}});
}

nlohmann::json VoiceFrontendRuntime::handle_command(const nlohmann::json& command) {
  std::scoped_lock control_lock(control_mutex_);
  const auto action = command.value("action", "");
  if (action == "state.get") {
    nlohmann::json activation_state = activation_json(ActivationSnapshot{});
    std::uint64_t activation_sample{};
    {
      std::scoped_lock pipeline_lock(pipeline_mutex_);
      if (segmenter_) {
        last_published_activation_state_ = segmenter_->activation();
        activation_state = activation_json(last_published_activation_state_);
      }
      if (ring_) activation_sample = ring_->tail();
    }
    const auto mode = replay_mode_.load(std::memory_order_acquire)
                          ? std::string("replay")
                          : std::string("live");
    nlohmann::json latest_announcement = nullptr;
    {
      std::scoped_lock announcement_lock(announcement_state_mutex_);
      latest_announcement = mode == "live" ? latest_live_announcement_
                                            : latest_replay_announcement_;
    }
    emit_event("config_state", {{"config", config_store_.to_public_json(config_)}});
    emit_event("sessions", session_list());
    emit_event("recording_state",
               {{"active", recorder_.active()},
                {"session", recorder_.session_path().string()},
                {"started_at_ms", recorder_.started_at_unix_ms()},
                {"incomplete", recorder_.incomplete()}});
    emit_event("runtime_mode", {{"mode", mode}});
    emit_event("activation_state", activation_state, activation_sample, mode);
    emit_event("runtime_state", {{"mode", mode},
                                  {"activation", activation_state},
                                  {"latest_announcement", latest_announcement}},
               activation_sample, mode);
    if (replay_) emit_event("replay_state", replay_->state(), 0, "replay");
    return {{"ok", true}, {"metrics", metrics_json()},
            {"activation_state", std::move(activation_state)},
            {"latest_announcement", std::move(latest_announcement)}};
  }
  if (action == "replay.parse_debug") {
    if (!replay_mode_.load(std::memory_order_acquire)) {
      return {{"ok", false},
              {"error", "parse debugging is available only in replay mode"}};
    }
    const auto text = command.value("text", "");
    const auto wake_word = command.value("wake_word", "");
    if (text.size() > 4096 || wake_word.size() > 256) {
      return {{"ok", false}, {"error", "parse debug input is too long"}};
    }

    std::shared_ptr<const CommandParser> parser;
    std::uint64_t config_revision{};
    bool commands_enabled{};
    {
      std::scoped_lock command_lock(command_mutex_);
      parser = command_parser_;
      config_revision = command_config_revision_;
      commands_enabled = commands_enabled_;
    }
    if (!parser) {
      return {{"ok", false}, {"error", "command parser is unavailable"}};
    }

    CommandParseContext context;
    context.runtime_session_id = runtime_session_id_;
    context.utterance_id =
        "replay-parse-debug-" +
        std::to_string(replay_parse_debug_sequence_.fetch_add(
            1, std::memory_order_relaxed));
    context.origin = wake_word.empty() ? UtteranceOrigin::followup
                                       : UtteranceOrigin::keyword;
    context.wake_position = wake_word.empty() ? "" : "manual";
    context.wake_word = wake_word;
    context.config_revision = config_revision;
    context.recognition_generation =
        recognition_generation_.load(std::memory_order_acquire);
    context.final_revision = 1;
    context.source = ExecutionSource::replay;
    context.execution_mode = ExecutionMode::dry_run;

    const auto parsed = parser->parse(text, context);
    auto debug =
        parsed.ok()
            ? parse_debug_success(text, context, *parsed.plan, "manual")
            : parse_debug_failure(
                  text, context, "manual", "parser",
                  parsed.error ? parsed.error->code : "rule_not_matched",
                  parsed.error ? parsed.error->message
                               : "text did not match a command rule",
                  parsed.error ? parsed.error->byte_offset : std::size_t{});
    debug["commands_enabled"] = commands_enabled;
    debug["safe_only"] = true;
    emit_event("parse_debug", debug, 0, "replay");
    return {{"ok", true}, {"parse", std::move(debug)}};
  }
  if (action == "recording.start") {
    if (replay_mode_.load(std::memory_order_acquire)) {
      return {{"ok", false},
              {"error", "return to live mode before starting a recording"}};
    }
    if (recorder_.active()) {
      return {{"ok", false}, {"error", "recording is already active"}};
    }
    const auto model_file = [](const std::filesystem::path& path) {
      nlohmann::json value{{"path", path.string()}, {"present", std::filesystem::is_regular_file(path)}};
      if (value["present"].get<bool>()) value["sha256"] = sha256_file(path);
      return value;
    };
    nlohmann::json capture_capabilities;
    {
      std::scoped_lock capture_lock(capture_events_mutex_);
      capture_capabilities = capture_capabilities_;
    }
    auto manifest = nlohmann::json{
        {"config", config_store_.to_manifest_json(config_)},
        {"models", {{"sherpa_onnx_version", "1.13.2"},
                    {"kws", {{"encoder", model_file(config_.kws.encoder)},
                             {"decoder", model_file(config_.kws.decoder)},
                             {"joiner", model_file(config_.kws.joiner)},
                             {"tokens", model_file(config_.kws.tokens)},
                             {"keywords", model_file(config_.kws.keywords)}}},
                    {"vad", model_file(config_.vad.model)}}},
        {"asr_model", {{"implementation", config_.asr.implementation},
                       {"encoder", model_file(config_.asr.encoder)},
                       {"decoder", model_file(config_.asr.decoder)},
                       {"tokens", model_file(config_.asr.tokens)}}},
        {"aec", {{"webrtc_compiled", webrtc_aec3_compiled()},
                 {"status", preprocess_json(preprocess_diagnostics_snapshot())}}},
        {"capture_capabilities", std::move(capture_capabilities)},
        {"clock", {{"qpc_unit", "100ns"}, {"processed_sample_rate", kProcessingSampleRate}}},
        {"initial_metrics", metrics_json()},
        {"kws_status", kws_->status()}, {"vad_status", vad_->status()}};
    const auto path = recorder_.start(config_.recording, manifest);
    emit_event("recording_state",
               {{"active", true},
                {"session", path.string()},
                {"started_at_ms", recorder_.started_at_unix_ms()},
                {"incomplete", false}});
    return {{"ok", true}, {"session", path.string()}};
  }
  if (action == "recording.stop") {
    if (!recorder_.active()) {
      return {{"ok", false}, {"error", "recording is not active"}};
    }
    recorder_.stop(metrics_json());
    emit_event("recording_state",
               {{"active", false},
                {"session", recorder_.session_path().string()},
                {"started_at_ms", recorder_.started_at_unix_ms()},
                {"incomplete", recorder_.incomplete()}});
    emit_event("sessions", session_list());
    return {{"ok", true}};
  }
  if (action == "config.apply") {
    AppConfig candidate;
    {
      std::scoped_lock lock(config_mutex_);
      candidate = config_;
    }
    const auto validation = config_store_.apply_patch(candidate, command.value("patch", nlohmann::json::object()));
    if (!validation.ok()) {
      emit_event("config_rejected", {{"errors", validation.errors}});
      return {{"ok", false}, {"error", validation.errors.front()}, {"errors", validation.errors}};
    }
    AppConfig current;
    {
      std::scoped_lock lock(config_mutex_);
      current = config_;
    }
    std::vector<std::string> restart_required;
    const auto changed = [&restart_required](bool value, const char* field) {
      if (value) restart_required.emplace_back(field);
    };
    changed(candidate.audio.microphone_device != current.audio.microphone_device,
            "audio.microphone_device");
    changed(candidate.audio.loopback_device != current.audio.loopback_device,
            "audio.loopback_device");
    changed(candidate.preprocess.implementation != current.preprocess.implementation,
            "preprocess.implementation");
    changed(candidate.preprocess.aec_enabled != current.preprocess.aec_enabled,
            "preprocess.aec_enabled");
    changed(candidate.aec.processing_rate_hz != current.aec.processing_rate_hz,
            "aec.processing_rate_hz");
    changed(candidate.aec.request_raw_capture != current.aec.request_raw_capture,
            "aec.request_raw_capture");
    changed(candidate.aec.request_post_volume_loopback != current.aec.request_post_volume_loopback,
            "aec.request_post_volume_loopback");
    changed(candidate.kws.encoder != current.kws.encoder, "kws.encoder");
    changed(candidate.kws.decoder != current.kws.decoder, "kws.decoder");
    changed(candidate.kws.joiner != current.kws.joiner, "kws.joiner");
    changed(candidate.kws.tokens != current.kws.tokens, "kws.tokens");
    changed(candidate.kws.keywords != current.kws.keywords, "kws.keywords");
    changed(candidate.kws.provider != current.kws.provider, "kws.provider");
    changed(candidate.kws.num_threads != current.kws.num_threads, "kws.num_threads");
    changed(candidate.vad.model != current.vad.model, "vad.model");
    changed(candidate.vad.provider != current.vad.provider, "vad.provider");
    changed(candidate.vad.num_threads != current.vad.num_threads, "vad.num_threads");
    changed(candidate.asr.enabled != current.asr.enabled, "asr.enabled");
    changed(candidate.asr.encoder != current.asr.encoder, "asr.encoder");
    changed(candidate.asr.decoder != current.asr.decoder, "asr.decoder");
    changed(candidate.asr.tokens != current.asr.tokens, "asr.tokens");
    changed(candidate.asr.provider != current.asr.provider, "asr.provider");
    changed(candidate.asr.num_threads != current.asr.num_threads, "asr.num_threads");
    changed(candidate.commands.action_timeout_ms != current.commands.action_timeout_ms,
            "commands.action_timeout_ms");
    changed(candidate.commands.queue_capacity != current.commands.queue_capacity,
            "commands.queue_capacity");
    changed(candidate.segmentation.assembly_queue_capacity !=
                current.segmentation.assembly_queue_capacity,
            "segmentation.assembly_queue_capacity");
    changed(candidate.announcements.backend != current.announcements.backend,
            "announcements.backend");
    changed(candidate.announcements.queue_capacity !=
                current.announcements.queue_capacity,
            "announcements.queue_capacity");
    const bool save = command.value("save", false);
    if (!restart_required.empty() && !save) {
      const std::vector<std::string> errors{
          "cold parameter changes require 'apply and save', then a process restart"};
      emit_event("config_rejected", {{"errors", errors}, {"restart_required", restart_required}});
      return {{"ok", false}, {"error", errors.front()}, {"errors", errors},
              {"restart_required", restart_required}};
    }

    AppConfig active = candidate;
    active.audio = current.audio;
    active.preprocess = current.preprocess;
    active.aec.processing_rate_hz = current.aec.processing_rate_hz;
    active.aec.request_raw_capture = current.aec.request_raw_capture;
    active.aec.request_post_volume_loopback = current.aec.request_post_volume_loopback;
    active.ring = current.ring;
    active.recording = current.recording;
    active.segmentation.assembly_queue_capacity =
        current.segmentation.assembly_queue_capacity;
    active.announcements = candidate.announcements;
    active.announcements.backend = current.announcements.backend;
    active.announcements.queue_capacity =
        current.announcements.queue_capacity;
    active.kws = current.kws;
    active.kws.threshold = candidate.kws.threshold;
    active.kws.boosting_score = candidate.kws.boosting_score;
    active.vad = current.vad;
    active.vad.threshold = candidate.vad.threshold;
    active.vad.min_speech_ms = candidate.vad.min_speech_ms;
    active.vad.min_silence_ms = candidate.vad.min_silence_ms;
    active.vad.max_speech_ms = candidate.vad.max_speech_ms;
    active.asr = current.asr;
    active.asr.emit_partials = candidate.asr.emit_partials;
    active.asr.feed_chunk_ms = candidate.asr.feed_chunk_ms;
    active.commands = candidate.commands;
    // These shape the long-lived Windows backend/executor and are applied on
    // the next process start. Grammar and enablement remain true hot settings.
    active.commands.action_timeout_ms = current.commands.action_timeout_ms;
    active.commands.queue_capacity = current.commands.queue_capacity;
    active.web = current.web;
    active.web.telemetry_hz = candidate.web.telemetry_hz;

    const bool aec_changed =
        active.aec.enabled != current.aec.enabled ||
        active.aec.high_pass_filter != current.aec.high_pass_filter ||
        active.aec.noise_suppression != current.aec.noise_suppression ||
        active.aec.gain_control != current.aec.gain_control ||
        active.aec.alignment_wait_ms != current.aec.alignment_wait_ms ||
        active.aec.target_render_buffer_ms != current.aec.target_render_buffer_ms ||
        active.aec.max_render_buffer_ms != current.aec.max_render_buffer_ms ||
        active.aec.drift_window_ms != current.aec.drift_window_ms ||
        active.aec.max_drift_ppm != current.aec.max_drift_ppm ||
        active.aec.hard_resync_error_ms != current.aec.hard_resync_error_ms ||
        active.aec.delay_offset_ms != current.aec.delay_offset_ms ||
        active.aec.microphone_channel_index != current.aec.microphone_channel_index ||
        active.aec.auto_delay_enabled != current.aec.auto_delay_enabled ||
        active.aec.auto_delay_min_ms != current.aec.auto_delay_min_ms ||
        active.aec.auto_delay_max_ms != current.aec.auto_delay_max_ms ||
        active.aec.auto_delay_window_ms != current.aec.auto_delay_window_ms ||
        active.aec.auto_delay_update_ms != current.aec.auto_delay_update_ms ||
        active.aec.auto_delay_min_correlation != current.aec.auto_delay_min_correlation ||
        active.aec.stats_hz != current.aec.stats_hz;
    const bool kws_changed = active.kws.threshold != current.kws.threshold ||
                             active.kws.boosting_score != current.kws.boosting_score;
    const bool vad_changed = active.vad.threshold != current.vad.threshold ||
                             active.vad.min_speech_ms != current.vad.min_speech_ms ||
                             active.vad.min_silence_ms != current.vad.min_silence_ms ||
                             active.vad.max_speech_ms != current.vad.max_speech_ms;
    const bool segmentation_changed =
        active.segmentation.wake_guard_ms != current.segmentation.wake_guard_ms ||
        active.segmentation.pre_roll_ms != current.segmentation.pre_roll_ms ||
        active.segmentation.post_roll_ms != current.segmentation.post_roll_ms ||
        active.segmentation.endpoint_silence_ms != current.segmentation.endpoint_silence_ms ||
        active.segmentation.max_candidate_ms != current.segmentation.max_candidate_ms ||
        active.segmentation.min_command_speech_ms !=
            current.segmentation.min_command_speech_ms ||
        active.segmentation.embedded_join_silence_ms !=
            current.segmentation.embedded_join_silence_ms;
    const bool activation_changed =
        active.activation.enabled != current.activation.enabled ||
        active.activation.idle_timeout_ms != current.activation.idle_timeout_ms ||
        active.activation.hard_limit_ms != current.activation.hard_limit_ms;
    const bool release_announcement_gate =
        current.announcements.enabled &&
        !current.announcements.barge_in &&
        (!active.announcements.enabled || active.announcements.barge_in);
    const bool parser_changed =
        active.commands.default_volume_step_percent !=
            current.commands.default_volume_step_percent ||
        active.commands.max_spoken_volume_step_percent !=
            current.commands.max_spoken_volume_step_percent ||
        active.commands.max_actions_per_utterance !=
            current.commands.max_actions_per_utterance ||
        active.commands.play_phrases != current.commands.play_phrases ||
        active.commands.pause_phrases != current.commands.pause_phrases ||
        active.commands.volume_up_phrases != current.commands.volume_up_phrases ||
        active.commands.volume_down_phrases != current.commands.volume_down_phrases ||
        active.commands.connectors != current.commands.connectors;
    const bool pipeline_changed = aec_changed || kws_changed || vad_changed ||
                                  segmentation_changed || activation_changed;

    std::unique_ptr<IAudioPreprocessor> new_preprocessor;
    std::unique_ptr<IAudioPreprocessor> new_keyword_preprocessor;
    if (aec_changed) {
      const auto timeline = make_preprocessor_timeline_config(active.aec);
      new_preprocessor = active.preprocess.aec_enabled && active.aec.enabled &&
                                 active.preprocess.implementation == "webrtc_aec3"
                             ? std::unique_ptr<IAudioPreprocessor>(
                                   std::make_unique<WebRtcAec3Preprocessor>(active.audio, timeline))
                             : std::unique_ptr<IAudioPreprocessor>(
                                   std::make_unique<BypassPreprocessor>(active.audio, timeline));
      new_keyword_preprocessor =
          std::make_unique<BypassPreprocessor>(active.audio, timeline);
    }
    auto new_vad = vad_changed ? create_vad(active.vad) : nullptr;
    auto new_kws = kws_changed ? create_keyword_spotter(active.kws) : nullptr;
    auto new_microphone_kws =
        kws_changed ? create_keyword_spotter(active.kws) : nullptr;
    std::shared_ptr<const CommandParser> new_parser;
    if (parser_changed) {
      new_parser = std::make_shared<const CommandParser>(
          static_cast<int>(active.commands.default_volume_step_percent),
          static_cast<int>(active.commands.max_spoken_volume_step_percent),
          command_grammar(active.commands));
    }
    if (save) config_store_.save_overrides(candidate);
    std::size_t cancelled_candidates{};
    {
      std::scoped_lock pipeline_lock(pipeline_mutex_);
      {
        std::scoped_lock config_lock(config_mutex_);
        config_ = active;
      }
      if (release_announcement_gate &&
          announcement_playback_active_.exchange(
              false, std::memory_order_acq_rel)) {
        const auto at_sample = ring_ ? ring_->tail() : 0;
        const auto playback_samples =
            at_sample >= announcement_playback_start_sample_
                ? at_sample - announcement_playback_start_sample_
                : 0;
        announcement_playback_start_sample_ = 0;
        announcement_guard_until_sample_.store(0, std::memory_order_release);
        announcement_kws_reset_pending_.store(false,
                                               std::memory_order_release);
        if (segmenter_ && segmenter_->finish_activation_playback(
                              playback_samples, at_sample)) {
          emit_activation_events(segmenter_->take_activation_events());
          emit_activation_state_if_changed(
              at_sample,
              replay_mode_.load(std::memory_order_acquire) ? "replay"
                                                           : "live");
        }
      }
      if (pipeline_changed) {
        if (new_preprocessor) preprocessor_ = std::move(new_preprocessor);
        if (new_keyword_preprocessor) {
          keyword_preprocessor_ = std::move(new_keyword_preprocessor);
        }
        if (new_vad) vad_ = std::move(new_vad);
        if (new_kws) kws_ = std::move(new_kws);
        if (new_microphone_kws) {
          microphone_kws_ = std::move(new_microphone_kws);
        }
        cancelled_candidates = segmenter_->pending_count();
        segmenter_->reconfigure(config_.segmentation, config_.activation);
        emit_activation_events(segmenter_->take_activation_events());
        emit_activation_state_if_changed(
            ring_ ? ring_->tail() : 0,
            replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
        if (candidate_assembler_) candidate_assembler_->cancel_pending();
        if (microphone_candidate_assembler_) {
          microphone_candidate_assembler_->cancel_pending();
        }
        const auto next = ring_->tail();
        if (microphone_ring_) {
          microphone_ring_->reset(microphone_ring_->tail());
        }
        kws_->reset(next);
        microphone_kws_->reset(next);
        pending_microphone_keywords_.clear();
        last_runtime_keyword_.clear();
        last_runtime_keyword_sample_ = 0;
        microphone_candidate_fallback_after_sample_ = 0;
        microphone_audio_activations_.clear();
        vad_->reset();
        vad_speech_active_ = false;
        suppress_current_vad_interval_ = false;
        cancel_recognition_generation("audio pipeline config hot swap");
      }
      // Keeping this exchange under the frame-boundary lock makes the config
      // revision and immutable parser snapshot visible as one utterance epoch.
      std::scoped_lock command_lock(command_mutex_);
      if (new_parser) command_parser_ = std::move(new_parser);
      command_config_revision_ = active.revision;
      commands_enabled_ = active.commands.enabled;
    }
    if (cancelled_candidates != 0) {
      emit_event("candidate_cancelled",
                 {{"reason", "config_hot_swap"}, {"count", cancelled_candidates}});
    }
    emit_event("config_applied", {{"config", config_store_.to_public_json(active)},
                                   {"saved", save}, {"restart_required", restart_required},
                                   {"pipeline_reset", pipeline_changed},
                                   {"command_parser_swapped", parser_changed},
                                   {"saved_config", config_store_.to_public_json(candidate)}});
    return {{"ok", true}, {"config", config_store_.to_public_json(active)},
            {"saved", save}, {"restart_required", restart_required}};
  }
  if (action == "replay.open") {
    if (recorder_.active()) {
      return {{"ok", false},
              {"error", "stop the active recording before opening replay"}};
    }
    auto requested = std::filesystem::path(command.value("session", ""));
    if (requested.empty()) throw std::invalid_argument("session is required");
    std::filesystem::path session_root;
    {
      std::scoped_lock config_lock(config_mutex_);
      session_root = config_.recording.session_root;
    }
    std::error_code path_error;
    const auto canonical_root =
        std::filesystem::weakly_canonical(session_root, path_error);
    if (path_error) throw std::invalid_argument("recording root is unavailable");
    auto session = requested.is_absolute() ? requested : session_root / requested;
    session = std::filesystem::weakly_canonical(session, path_error);
    if (path_error || session.parent_path() != canonical_root ||
        !valid_session_name(session.filename().string()) ||
        !inspect_session_directory(session)) {
      return {{"ok", false},
              {"error", "selected session has no replayable audio"}};
    }
    stop_captures();
    if (replay_) {
      replay_->stop();
      replay_.reset();
    }
    {
      std::scoped_lock callback_lock(replay_callback_mutex_);
      replay_mode_.store(false, std::memory_order_release);
    }
    stop_processing();
    reset_audio_queues();
    {
      std::scoped_lock lock(pipeline_mutex_);
      view_sample_origin_.reset();
      next_telemetry_sample_ = 0;
      reset_pipeline(true);
    }
    replay_mode_.store(true, std::memory_order_release);
    start_processing();
    if (!replay_) {
      replay_ = std::make_unique<ReplayController>(
          [this](AudioPacket packet) {
            enqueue_replay_packet(std::move(packet));
          },
          [this](nlohmann::json state) {
            emit_event("replay_state", std::move(state), 0, "replay");
          });
    }
    replay_->open(session);
    emit_event("runtime_mode", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
    return {{"ok", true}};
  }
  if (action == "live.resume") {
    if (!replay_mode_.load(std::memory_order_acquire)) {
      emit_event("runtime_mode", {{"mode", "live"}});
      return {{"ok", true}, {"already_live", true}};
    }
    if (replay_) {
      replay_->stop();
      replay_.reset();
    }
    {
      std::scoped_lock callback_lock(replay_callback_mutex_);
      replay_mode_.store(false, std::memory_order_release);
    }
    stop_processing();
    reset_audio_queues();
    {
      std::scoped_lock lock(pipeline_mutex_);
      view_sample_origin_.reset();
      next_telemetry_sample_ = 0;
      reset_pipeline(true);
    }
    start_processing();
    start_captures();
    emit_event("runtime_mode", {{"mode", "live"}});
    return {{"ok", true}};
  }
  if (action == "replay.play" && replay_) {
    replay_->play();
    return {{"ok", true}};
  }
  if (action == "replay.pause" && replay_) {
    replay_->pause();
    return {{"ok", true}};
  }
  if (action == "replay.speed" && replay_) {
    replay_->set_speed(command.value("speed", 1.0));
    return {{"ok", true}};
  }
  if (action == "replay.seek" && replay_) {
    const auto before = replay_->state();
    const bool resume_after_seek = before.value("playing", false);
    replay_->pause();
    if (!replay_->wait_until_quiescent(std::chrono::seconds(2))) {
      return {{"ok", false}, {"error", "replay did not become quiescent before seek"}};
    }
    {
      // No old callback is in flight, and holding this mutex prevents a newly
      // scheduled callback from crossing the queue/pipeline reset boundary.
      std::scoped_lock callback_lock(replay_callback_mutex_);
      stop_processing();
      reset_audio_queues();
      {
        std::scoped_lock pipeline_lock(pipeline_mutex_);
        view_sample_origin_.reset();
        next_telemetry_sample_ = 0;
        reset_pipeline(true);
      }
      replay_->seek_seconds(command.value("seconds", 0.0), resume_after_seek);
      start_processing();
    }
    return {{"ok", true}};
  }
  return {{"ok", false}, {"error", "unknown or unavailable action: " + action}};
}

nlohmann::json VoiceFrontendRuntime::session_list() const {
  std::filesystem::path session_root;
  {
    std::scoped_lock config_lock(config_mutex_);
    session_root = config_.recording.session_root;
  }
  std::vector<nlohmann::json> valid_sessions;
  std::uint64_t excluded_sessions{};
  std::error_code error;
  if (std::filesystem::is_directory(session_root, error)) {
    for (const auto& entry :
         std::filesystem::directory_iterator(session_root, error)) {
      if (error) break;
      if (!entry.is_directory(error)) continue;
      if (auto session = inspect_session_directory(entry.path())) {
        valid_sessions.push_back(std::move(*session));
      } else {
        ++excluded_sessions;
      }
    }
  }
  std::ranges::sort(valid_sessions, std::greater{},
                    [](const nlohmann::json& session) {
                      return session.value("name", "");
                    });
  auto sessions = nlohmann::json::array();
  for (auto& session : valid_sessions) {
    sessions.push_back(std::move(session));
  }
  return {{"sessions", std::move(sessions)},
          {"excluded_sessions", excluded_sessions}};
}

std::optional<DebugServer::FileResource>
VoiceFrontendRuntime::session_audio_resource(
    const nlohmann::json& request) const {
  const auto session_name = request.value("session", "");
  const auto stream = request.value("stream", "processed");
  if (!valid_session_name(session_name)) return std::nullopt;

  std::string filename;
  if (stream == "processed") {
    filename = "processed.wav";
  } else if (stream == "microphone") {
    filename = "mic.wav";
  } else if (stream == "loopback") {
    filename = "loopback.wav";
  } else {
    return std::nullopt;
  }

  std::filesystem::path session_root;
  {
    std::scoped_lock config_lock(config_mutex_);
    session_root = config_.recording.session_root;
  }
  std::error_code error;
  const auto canonical_root =
      std::filesystem::weakly_canonical(session_root, error);
  if (error) return std::nullopt;
  const auto session = std::filesystem::weakly_canonical(
      session_root / session_name, error);
  if (error || session.parent_path() != canonical_root) return std::nullopt;
  const auto info = inspect_session_directory(session);
  if (!info || !(*info)["streams"].contains(stream)) return std::nullopt;

  const auto audio = session / filename;
  if (!std::filesystem::is_regular_file(audio, error) || error) {
    return std::nullopt;
  }
  return DebugServer::FileResource{audio, "audio/wav"};
}

nlohmann::json VoiceFrontendRuntime::metrics_json() const {
  auto result = nlohmann::json{{"audio_packets", audio_packets_.load()}, {"processed_frames", processed_frames_.load()},
          {"kws_hits", kws_hits_.load()}, {"candidates", candidates_.load()},
          {"rejections", rejections_.load()}, {"audio_queue_drops", audio_queue_drops_.load()},
          {"audio_reset_backlog_drops", audio_reset_backlog_drops_.load()},
          {"discontinuities", discontinuities_.load()},
          {"telemetry_dropped", debug_.dropped() +
                                    telemetry_queue_drops_.load()},
          {"asr_partials", asr_partials_.load()}, {"asr_finals", asr_finals_.load()},
          {"asr_dropped", asr_dropped_.load()},
          {"assembly_dropped", assembly_dropped_.load()},
          {"assembly_outstanding",
           (candidate_assembler_ ? candidate_assembler_->outstanding() : 0) +
               (microphone_candidate_assembler_
                    ? microphone_candidate_assembler_->outstanding()
                    : 0)},
          {"command_plans", command_plans_.load()},
          {"command_rejections", command_rejections_.load()},
          {"activations_started", activations_started_.load()},
          {"activations_refreshed", activations_refreshed_.load()},
          {"activations_expired", activations_expired_.load()},
          {"activations_cancelled", activations_cancelled_.load()},
          {"followup_turns", followup_turns_.load()},
          {"announcements_requested", announcements_requested_.load()},
          {"announcements_dropped", announcements_dropped_.load()},
          {"announcements_failed", announcements_failed_.load()},
          {"actions_succeeded", actions_succeeded_.load()}, {"actions_failed", actions_failed_.load()},
          {"recording_incomplete", recorder_.incomplete()}};
  if (preprocessor_) result["aec"] = preprocess_json(preprocess_diagnostics_snapshot());
  if (asr_) {
    const auto stats = asr_->stats();
    result["asr"] = {{"available", asr_->available()},
                     {"state", to_string(asr_->state())}, {"status", asr_->status()},
                     {"queue_depth", asr_->queue_depth()}, {"submitted", stats.submitted},
                     {"pending_requests", asr_->pending_requests()},
                     {"pending_audio_samples", asr_->pending_audio_samples()},
                     {"queue_full", stats.dropped_queue_full},
                     {"pending_audio_full", stats.dropped_pending_audio_full},
                     {"active_stream_limit", stats.rejected_active_stream_limit},
                     {"errors", stats.errors},
                     {"stale_results", stats.suppressed_stale_results}};
  }
  if (action_executor_) result["action_queue_depth"] = action_executor_->queue_size();
  if (announcement_dispatcher_) {
    result["announcement_queue_depth"] = announcement_dispatcher_->queue_size();
  }
  return result;
}

PreprocessDiagnostics VoiceFrontendRuntime::preprocess_diagnostics_snapshot() const {
  // TimelineEngine is single-threaded by design. All external readers take
  // the same frame-boundary lock used by PushPacket/TryPopFrame so copying its
  // non-atomic diagnostics cannot race the real-time processing thread.
  std::scoped_lock lock(pipeline_mutex_);
  return preprocessor_ ? preprocessor_->Diagnostics() : PreprocessDiagnostics{};
}

VoiceFrontendRuntime::SignalSummary VoiceFrontendRuntime::summarize(
    std::span<const float> values) {
  if (values.empty()) return {};
  SignalSummary result{values.front(), values.front(), 0.0F};
  double squares{};
  for (const auto value : values) {
    result.min = std::min(result.min, value);
    result.max = std::max(result.max, value);
    squares += static_cast<double>(value) * value;
  }
  result.rms = static_cast<float>(std::sqrt(squares / values.size()));
  return result;
}

VoiceFrontendRuntime::SignalSummary VoiceFrontendRuntime::summarize(const NormalizedFrame& frame) {
  return summarize(frame.samples);
}

}  // namespace dvo
