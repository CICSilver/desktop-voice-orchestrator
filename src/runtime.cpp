#include "dvo/runtime.h"

#include "dvo/benchmark_report.h"

#include <algorithm>
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
#include "dvo/windows_action_backend.h"

namespace dvo {
namespace {

// The executor retains 4096 terminal action IDs. Keeping at most 512 plans
// guarantees that all eight actions of every replayable snapshot still fit in
// the deduplication ledger.
constexpr std::size_t kCompletedCommandSnapshotCapacity = 4096 / kMaxCommandActions;

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
    actions.push_back({{"action_id", action.action_id}, {"sequence", action.sequence},
                       {"type", to_string(action.type)},
                       {"volume_delta_percent", action.volume_delta_percent}});
  }
  return {{"command_id", plan.command_id}, {"runtime_session_id", plan.runtime_session_id},
          {"utterance_id", plan.utterance_id},
          {"raw_text", plan.raw_text}, {"normalized_text", plan.normalized_text},
          {"parser_version", plan.parser_version}, {"config_revision", plan.config_revision},
          {"recognition_generation", plan.recognition_generation},
          {"final_revision", plan.final_revision},
          {"source", to_string(plan.source)},
          {"execution_mode", to_string(plan.execution_mode)}, {"actions", std::move(actions)}};
}

nlohmann::json action_result_json(const ActionResult& result) {
  return {{"command_id", result.command_id}, {"utterance_id", result.utterance_id},
          {"action_id", result.action_id}, {"sequence", result.sequence},
          {"type", to_string(result.type)}, {"status", to_string(result.status)},
          {"source", to_string(result.source)},
          {"adapter", result.adapter}, {"target_id", result.target_id},
          {"error_code", result.error_code}, {"message", result.message},
          {"requested_volume_delta_percent", result.requested_volume_delta_percent},
          {"volume_before", result.volume_before}, {"volume_requested", result.volume_requested},
          {"volume_after", result.volume_after}, {"mute_before", result.mute_before},
          {"mute_after", result.mute_after}, {"clamped", result.clamped},
          {"verified", result.verified}, {"duration_ms", result.duration.count()}};
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
  return {{"state", to_string(diagnostics.state)}, {"enabled", diagnostics.aec_requested},
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
           {"resampler", {{"speex_compiled", diagnostics.speexdsp_compiled},
                           {"microphone_speex", diagnostics.microphone_resampler_speex},
                           {"render_speex", diagnostics.render_resampler_speex},
                           {"render_rate_updates", diagnostics.render_resampler_rate_updates},
                           {"synthetic_samples_replaced",
                            diagnostics.render_synthetic_samples_replaced},
                           {"failures", diagnostics.resampler_failures},
                           {"last_error_code", diagnostics.last_resampler_error_code},
                           {"last_input_expected",
                            diagnostics.last_resampler_input_expected},
                           {"last_input_consumed",
                            diagnostics.last_resampler_input_consumed}}},
          {"render_fifo_ms", diagnostics.render_buffered_samples * 1000.0 /
                                 static_cast<double>(kProcessingSampleRate)},
          {"render_target_ms", diagnostics.target_render_buffer_samples * 1000.0 /
                                   static_cast<double>(kProcessingSampleRate)},
          {"processing_average_ms", diagnostics.average_processing_time_us / 1000.0},
          {"processing_max_ms", diagnostics.max_processing_time_us / 1000.0},
          {"erle_db", diagnostics.echo_return_loss_enhancement_db},
          {"erl_db", diagnostics.echo_return_loss_db},
          {"residual_echo_likelihood", diagnostics.residual_echo_likelihood},
          {"estimated_delay_ms", diagnostics.estimated_delay_ms},
          {"reset_count", diagnostics.resets}, {"fallback_frames", diagnostics.fallback_frames},
          {"last_reset_reason", to_string(diagnostics.last_reset_reason)}};
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
  const auto timeline = make_preprocessor_timeline_config(config_.aec);
  if (config_.preprocess.aec_enabled && config_.aec.enabled &&
      config_.preprocess.implementation == "webrtc_aec3") {
    preprocessor_ = std::make_unique<WebRtcAec3Preprocessor>(config_.audio, timeline);
  } else {
    preprocessor_ = std::make_unique<BypassPreprocessor>(config_.audio, timeline);
  }
  vad_ = create_vad(config_.vad);
  kws_ = create_keyword_spotter(config_.kws);
  segmenter_ = std::make_unique<UtteranceSegmenter>(config_.segmentation, *ring_);
  candidate_assembler_ = std::make_unique<CandidateAssembler>(
      *ring_, config_.segmentation.assembly_queue_capacity);
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
  action_executor_ = std::make_unique<OrderedActionExecutor>(
      create_windows_action_backend(
          {config_.audio.loopback_device, config_.commands.action_timeout_ms}),
      ActionExecutorConfig{config_.commands.queue_capacity, 4096, ExecutionMode::live},
      [this](const ActionResult& result) { handle_action_result(result); });
  active_recognizers_.clear();
  pending_recognition_starts_.clear();
  recognition_generation_.fetch_add(1, std::memory_order_acq_rel);
  reset_audio_queues();
  telemetry_queue_ = std::make_unique<SpscQueue<TelemetrySample>>(128);
  telemetry_queue_drops_.store(0, std::memory_order_release);
  telemetry_enabled_.store(false, std::memory_order_release);
  next_telemetry_sample_ = 0;
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
    debug_.start(config_.web, [this](const auto& command) { return handle_command(command); });
    telemetry_enabled_.store(true, std::memory_order_release);
  }
  start_processing();
  start_captures();
  emit_event("runtime_started", {{"mode", "live"}, {"vad", vad_->status()}, {"kws", kws_->status()}});
  emit_event("aec_status", preprocess_json(preprocess_diagnostics_snapshot()));
  emit_event("asr_status", {{"state", to_string(asr_->state())},
                            {"status", asr_->status()}});
  emit_event("runtime_mode", {{"mode", "live"}});
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
    debug_.start(config_.web, [this](const auto& command) { return handle_command(command); });
    telemetry_enabled_.store(true, std::memory_order_release);
  }
  start_processing();
  replay_ = std::make_unique<ReplayController>(
      [this](AudioPacket packet) { enqueue_replay_packet(std::move(packet)); });
  replay_->open(session);
  replay_->set_speed(speed);
  replay_->play();
  emit_event("runtime_started", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
  emit_event("aec_status", preprocess_json(preprocess_diagnostics_snapshot()), 0, "replay");
  emit_event("asr_status", {{"state", to_string(asr_->state())},
                            {"status", asr_->status()}}, 0, "replay");
  emit_event("runtime_mode", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
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
        (candidate_assembler_ && candidate_assembler_->outstanding() != 0)) {
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
                     candidate_assembler_->outstanding() == 0);
    break;
  }

  const bool asr_load_finished =
      asr_ ? asr_->wait_until_loaded(remaining()) : true;
  const bool asr_idle = asr_ ? asr_->wait_until_idle(remaining()) : true;
  const bool actions_idle =
      action_executor_ ? action_executor_->wait_until_idle(remaining()) : true;
  auto metrics = metrics_json();
  metrics["elapsed_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started).count();
  metrics["session"] = session.string();
  metrics["benchmark_wait"] = {{"replay_finished", replay_finished},
                               {"pipeline_idle", pipeline_idle},
                               {"asr_load_finished", asr_load_finished},
                               {"asr_idle", asr_idle},
                               {"actions_idle", actions_idle},
                               {"timed_out", !(replay_finished && pipeline_idle &&
                                                asr_load_finished && asr_idle &&
                                                actions_idle)},
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
  stop_async_services();
  recorder_.stop(metrics_json());
}

void VoiceFrontendRuntime::stop_async_services() {
  if (candidate_assembler_) candidate_assembler_->stop();
  candidate_assembler_.reset();
  const auto generation = recognition_generation_.load(std::memory_order_acquire);
  if (asr_) asr_->cancel_generation(generation);
  active_recognizers_.clear();
  pending_recognition_starts_.clear();
  asr_.reset();
  if (action_executor_) action_executor_->stop(false);
  action_executor_.reset();
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
        reset_pipeline(true, false);
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
        emit_event("audio_timeline_reset",
                   {{"stream", "processed"},
                    {"reason", "normalized_frame_discontinuity"}},
                   frame.first_sample,
                   replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
        report_audio_queue_boundary("processed",
                                    PreprocessResetReason::discontinuity);
        reset_pipeline(true, false);
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
  segmenter_->set_vad_state(vad_update.speech, frame.first_sample + frame.samples.size());
  for (const auto& interval : vad_update.completed) segmenter_->add_vad_interval(interval);

  if (auto hit = kws_->accept(frame)) {
    const auto guard = static_cast<std::uint64_t>(config_.segmentation.wake_guard_ms) * kProcessingSampleRate / 1000;
    hit->wake_span.start = hit->wake_span.start > guard ? hit->wake_span.start - guard : 0;
    hit->wake_span.end += guard;
    kws_hits_.fetch_add(1, std::memory_order_relaxed);
    emit_event("kws_hit", kws_json(*hit), hit->detected_at_sample,
               replay_mode_ ? "replay" : "live");
    if (auto start = segmenter_->add_kws_hit(std::move(*hit))) {
      begin_recognition(*start);
    }
  }

  feed_recognizers(frame);

  auto segmented = segmenter_->advance_for_assembly(
      frame.first_sample + frame.samples.size());
  for (auto& request : segmented.assemblies) {
    const auto utterance_id = request.candidate.utterance_id;
    if (candidate_assembler_ && candidate_assembler_->try_submit(std::move(request))) {
      emit_event("candidate_assembly_queued",
                 {{"utterance_id", utterance_id},
                  {"outstanding", candidate_assembler_->outstanding()}},
                 frame.first_sample, replay_mode_ ? "replay" : "live");
    } else {
      constexpr auto reason = "candidate assembly queue is full";
      assembly_dropped_.fetch_add(1, std::memory_order_relaxed);
      rejections_.fetch_add(1, std::memory_order_relaxed);
      emit_event("candidate_rejected",
                 {{"utterance_id", utterance_id}, {"reason", reason}},
                 frame.first_sample, replay_mode_ ? "replay" : "live");
      cancel_recognition(utterance_id, reason);
    }
  }
  for (const auto& rejection : segmented.rejections) {
    rejections_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate_rejected", {{"utterance_id", rejection.utterance_id},
                                       {"reason", rejection.reason}}, frame.first_sample,
               replay_mode_ ? "replay" : "live");
    cancel_recognition(rejection.utterance_id, rejection.reason);
  }

  if (telemetry_enabled_.load(std::memory_order_relaxed) &&
      frame.first_sample >= next_telemetry_sample_) {
    TelemetrySample telemetry;
    telemetry.sample = frame.first_sample;
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

bool VoiceFrontendRuntime::drain_audio_assemblies() {
  if (!candidate_assembler_) return false;
  bool drained{};
  AudioAssemblyResult result;
  while (candidate_assembler_->try_pop(result)) {
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
  return drained;
}

void VoiceFrontendRuntime::complete_backfill(BackfillAssemblyResult result) {
  const auto pending_it = pending_recognition_starts_.find(result.utterance_id);
  if (pending_it == pending_recognition_starts_.end()) {
    erase_command_snapshot(result.utterance_id, result.recognition_generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                              result.utterance_id, result.recognition_generation,
                              result.wake_end_sample, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              result.rejection.empty()
                                  ? "recognition start no longer pending"
                                  : result.rejection});
    emit_event("asr_backfill_stale",
               {{"utterance_id", result.utterance_id},
                {"reason", result.rejection.empty()
                               ? "recognition start no longer pending"
                               : result.rejection}},
               result.wake_end_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }
  auto pending = std::move(pending_it->second);
  pending_recognition_starts_.erase(pending_it);

  if (!result.rejection.empty() ||
      result.recognition_generation != pending.generation ||
      pending.generation != recognition_generation_.load(std::memory_order_acquire)) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(result.utterance_id, pending.generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                              result.utterance_id, pending.generation,
                              result.wake_end_sample, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              result.rejection.empty() ? "stale recognition generation"
                                                       : result.rejection});
    emit_event("asr_backfill_rejected",
               {{"utterance_id", result.utterance_id},
                {"reason", result.rejection.empty() ? "stale recognition generation"
                                                     : result.rejection}},
               result.wake_end_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  const auto recognizer_state = asr_ ? asr_->state()
                                     : StreamingRecognizerState::unavailable;
  if (!asr_ || recognizer_state == StreamingRecognizerState::unavailable ||
      recognizer_state == StreamingRecognizerState::stopped) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(result.utterance_id, pending.generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                              result.utterance_id, pending.generation,
                              result.wake_end_sample, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              "recognizer unavailable before stream begin"});
    emit_event("asr_unavailable",
               {{"utterance_id", result.utterance_id},
                {"state", to_string(recognizer_state)},
                {"status", asr_ ? asr_->status() : "recognizer unavailable"}},
               result.wake_end_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  auto backfill = std::make_shared<std::vector<float>>(std::move(result.pcm));
  RecognitionBegin begin;
  begin.utterance_id = result.utterance_id;
  begin.generation = pending.generation;
  begin.source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  begin.left_backfill_first_sample = backfill->empty() ? pending.wake_end_sample
                                                       : result.first_sample;
  begin.left_backfill = std::move(backfill);
  const auto submitted = asr_->try_begin(std::move(begin));
  if (submitted != RecognitionSubmitStatus::accepted) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(result.utterance_id, pending.generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel,
                              result.utterance_id, pending.generation,
                              result.wake_end_sample, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              "recognizer rejected begin: " +
                                  std::string(to_string(submitted))});
    emit_event("asr_overloaded",
               {{"utterance_id", result.utterance_id},
                {"status", to_string(submitted)}},
               result.wake_end_sample,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  ActiveRecognitionFeed active;
  active.generation = pending.generation;
  active.wake_end_sample = pending.wake_end_sample;
  active.pending_first_sample = pending.pending_first_sample;
  active.chunk_samples = pending.chunk_samples;
  active.pending = std::move(pending.pending);
  active.pending.reserve(std::max(active.pending.capacity(), active.chunk_samples * 2));
  active_recognizers_.insert_or_assign(result.utterance_id, std::move(active));
  emit_event("asr_queued",
             {{"utterance_id", result.utterance_id},
              {"generation", pending.generation},
              {"state", to_string(recognizer_state)},
              {"command_config_revision", pending.command_config_revision},
              {"left_spans", pending.left_span_count},
              {"backfill_truncated", result.truncated}},
             result.wake_end_sample,
             replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
}

void VoiceFrontendRuntime::complete_candidate(CandidateAssemblyResult result) {
  if (!result.candidate) {
    rejections_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate_rejected",
               {{"utterance_id", result.utterance_id},
                {"reason", result.rejection}},
               0, replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    cancel_recognition(result.utterance_id, result.rejection);
    return;
  }

  candidates_.fetch_add(1, std::memory_order_relaxed);
  auto shared = std::make_shared<const UtteranceCandidate>(
      std::move(*result.candidate));
  const auto timestamp = shared->wake_span.end;
  emit_event("candidate", candidate_json(*shared), timestamp,
             replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
  if (recorder_.active()) recorder_.try_enqueue(RecordCandidate{shared});
  finalize_recognition(std::move(shared));
}

void VoiceFrontendRuntime::begin_recognition(const UtteranceStart& start) {
  const auto generation = recognition_generation_.load(std::memory_order_acquire);
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  emit_utterance_lifecycle({UtteranceLifecyclePhase::begin, start.utterance_id,
                            generation, start.wake_span.end, 0, source,
                            "keyword-associated utterance created"});
  if (!asr_ || !config_.asr.enabled) {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, start.utterance_id,
                              generation, start.wake_span.end, 0, source,
                              "streaming recognition is disabled"});
    return;
  }
  UtteranceCommandSnapshot command_snapshot;
  {
    std::scoped_lock command_lock(command_mutex_);
    command_snapshot = {command_parser_, command_config_revision_, commands_enabled_};
    const auto key = command_snapshot_key(start.utterance_id, generation);
    completed_command_snapshots_.erase(key);
    std::erase(completed_command_snapshot_order_, key);
    command_snapshots_.insert_or_assign(key, command_snapshot);
  }
  const auto recognizer_state = asr_->state();
  if (recognizer_state == StreamingRecognizerState::unavailable ||
      recognizer_state == StreamingRecognizerState::stopped) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(start.utterance_id, generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, start.utterance_id,
                              generation, start.wake_span.end, 0, source,
                              "recognizer unavailable at utterance begin"});
    emit_event("asr_unavailable", {{"utterance_id", start.utterance_id},
                                    {"state", to_string(recognizer_state)},
                                    {"status", asr_->status()}}, start.wake_span.end,
               replay_mode_ ? "replay" : "live");
    return;
  }
  const auto starting_streams = active_recognizers_.size() +
                                pending_recognition_starts_.size();
  if (starting_streams >= config_.asr.max_active_streams) {
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(start.utterance_id, generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, start.utterance_id,
                              generation, start.wake_span.end, 0, source,
                              "active recognizer stream limit reached"});
    emit_event("asr_stream_rejected",
               {{"utterance_id", start.utterance_id},
                {"reason", "active_stream_limit"},
                {"active_streams", starting_streams},
                {"max_active_streams", config_.asr.max_active_streams}},
               start.wake_span.end,
               replay_mode_.load(std::memory_order_acquire) ? "replay" : "live");
    return;
  }

  BackfillAssemblyRequest request;
  request.utterance_id = start.utterance_id;
  request.source_spans = start.provisional_left_spans;
  request.trailing_silence_samples =
      static_cast<std::size_t>(config_.segmentation.embedded_join_silence_ms) *
      kProcessingSampleRate / 1000;
  request.wake_end_sample = start.wake_span.end;
  request.recognition_generation = generation;
  if (!candidate_assembler_ || !candidate_assembler_->try_submit(std::move(request))) {
    assembly_dropped_.fetch_add(1, std::memory_order_relaxed);
    asr_dropped_.fetch_add(1, std::memory_order_relaxed);
    erase_command_snapshot(start.utterance_id, generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, start.utterance_id,
                              generation, start.wake_span.end, 0, source,
                              "audio backfill assembly queue is full"});
    emit_event("asr_backfill_rejected",
               {{"utterance_id", start.utterance_id},
                {"reason", "audio assembly queue is full"}},
               start.wake_span.end,
               replay_mode_ ? "replay" : "live");
    return;
  }

  PendingRecognitionStart pending;
  pending.generation = generation;
  pending.wake_end_sample = start.wake_span.end;
  pending.chunk_samples = std::max<std::size_t>(
      kFrameSamples, static_cast<std::size_t>(config_.asr.feed_chunk_ms) *
                         kProcessingSampleRate / 1000);
  const auto pending_limit =
      static_cast<std::size_t>(config_.asr.max_pending_audio_ms) *
      kProcessingSampleRate / 1000;
  pending.pending.reserve(std::min(pending_limit, pending.chunk_samples * 2));
  pending.command_config_revision = command_snapshot.config_revision;
  pending.left_span_count = start.provisional_left_spans.size();
  pending_recognition_starts_.insert_or_assign(start.utterance_id, pending);
  emit_event("asr_backfill_queued",
             {{"utterance_id", start.utterance_id},
              {"generation", generation},
              {"command_config_revision", command_snapshot.config_revision},
              {"left_spans", start.provisional_left_spans.size()},
              {"assembly_outstanding", candidate_assembler_->outstanding()}},
             start.wake_span.end, replay_mode_ ? "replay" : "live");
}

void VoiceFrontendRuntime::feed_recognizers(const NormalizedFrame& frame) {
  if (!asr_ || (active_recognizers_.empty() &&
                pending_recognition_starts_.empty())) return;
  const auto frame_end = frame.first_sample + frame.samples.size();

  const auto pending_limit = static_cast<std::size_t>(config_.asr.max_pending_audio_ms) *
                             kProcessingSampleRate / 1000;
  for (auto& [utterance_id, pending] : pending_recognition_starts_) {
    if (frame_end <= pending.wake_end_sample) continue;
    const auto start_sample = std::max(frame.first_sample, pending.wake_end_sample);
    const auto offset = static_cast<std::size_t>(start_sample - frame.first_sample);
    const auto incoming = frame.samples.size() - offset;
    if (pending.pending.size() + incoming > pending_limit) {
      asr_dropped_.fetch_add(1, std::memory_order_relaxed);
      emit_event("asr_overloaded",
                 {{"utterance_id", utterance_id},
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
    if (frame_end <= state.wake_end_sample) {
      ++it;
      continue;
    }
    const auto start_sample = std::max(frame.first_sample, state.wake_end_sample);
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
                                "streaming ASR audio chunk accepted"});
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
    erase_command_snapshot(utterance_id, generation);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              generation, candidate->wake_span.end, 0,
                              replay_mode_.load(std::memory_order_acquire) ? "replay" : "live",
                              "recognizer unavailable at finalize"});
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
                                         {"status", to_string(status)}}, 0,
               source);
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              generation, candidate->wake_span.end,
                              candidate->pcm.size(), source,
                              "exact-final submission rejected: " +
                                  std::string(to_string(status))});
    if (status == RecognitionSubmitStatus::queue_full ||
        status == RecognitionSubmitStatus::pending_audio_full) {
      cancel_recognition_generation("ASR exact-final submission overloaded");
    } else {
      erase_command_snapshot(utterance_id, generation);
    }
  } else {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::finalize, utterance_id,
                              generation, candidate->source_spans.empty()
                                              ? candidate->wake_span.end
                                              : candidate->source_spans.front().start,
                              candidate->pcm.size(), source,
                              "authoritative exact-final decode queued"});
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
  if (pending != pending_recognition_starts_.end()) {
    pending_recognition_starts_.erase(pending);
  }
  const auto source = replay_mode_.load(std::memory_order_acquire) ? "replay" : "live";
  emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                            generation, 0, 0, source, reason});
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
  for (const auto& [utterance_id, state] : active_recognizers_) {
    emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                              state.generation, 0, 0, source, reason});
  }
  for (const auto& [utterance_id, state] : pending_recognition_starts_) {
    if (!active_recognizers_.contains(utterance_id)) {
      emit_utterance_lifecycle({UtteranceLifecyclePhase::cancel, utterance_id,
                                state.generation, 0, 0, source, reason});
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
      } else if (result.kind == RecognitionResultKind::cancelled ||
                 result.kind == RecognitionResultKind::error) {
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
  auto payload = nlohmann::json{{"utterance_id", result.utterance_id},
                                {"generation", result.generation},
                                {"revision", result.revision},
                                {"text", result.hypothesis.text},
                                {"tokens", result.hypothesis.tokens},
                                {"is_final", result.kind == RecognitionResultKind::final},
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
  if (result.kind != RecognitionResultKind::final) return;
  asr_finals_.fetch_add(1, std::memory_order_relaxed);

  const auto reject = [&](std::string reason) {
    command_rejections_.fetch_add(1, std::memory_order_relaxed);
    auto rejection = nlohmann::json{{"utterance_id", result.utterance_id},
                                    {"text", result.hypothesis.text},
                                    {"reason", std::move(reason)}};
    if (command_snapshot) {
      rejection["config_revision"] = command_snapshot->config_revision;
    }
    emit_event("command_rejected", std::move(rejection),
               result.audio_end_sample, source);
  };
  if (!result.exact_final || !result.candidate) return reject("result is not an exact final decode");
  if (!command_snapshot || !command_snapshot->parser) {
    return reject("command parser snapshot is unavailable");
  }
  if (!command_snapshot->commands_enabled) return reject("command execution is disabled");
  if (result.candidate->truncated || result.candidate->timed_out ||
      result.candidate->discontinuity) {
    return reject("candidate is truncated, timed out, or discontinuous");
  }

  CommandParseContext context;
  context.runtime_session_id = runtime_session_id_;
  context.utterance_id = result.utterance_id;
  context.wake_position = to_string(result.candidate->position);
  context.config_revision = command_snapshot->config_revision;
  context.recognition_generation = result.generation;
  context.final_revision = result.revision;
  context.timestamp_sample = result.candidate->wake_span.end;
  context.source = source == "live" ? ExecutionSource::live : ExecutionSource::replay;
  context.execution_mode = source == "live" ? ExecutionMode::live : ExecutionMode::dry_run;
  auto parsed = command_snapshot->parser->parse(result.hypothesis.text, context);
  if (!parsed.ok()) {
    return reject(parsed.error ? parsed.error->code + ": " + parsed.error->message
                               : "text did not match a command rule");
  }

  auto plan = std::move(*parsed.plan);
  SubmitResult submitted;
  {
    // Serialize the final generation check with pipeline resets. Whichever
    // side obtains the frame-boundary lock first defines whether this final is
    // executable or stale.
    std::scoped_lock pipeline_lock(pipeline_mutex_);
    if (result.generation != recognition_generation_.load(std::memory_order_acquire)) return;
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
                                          {"action_id", action.action_id},
                                          {"sequence", action.sequence},
                                          {"type", to_string(action.type)}},
                                         timestamp, source);
                            }
                          })
                    : SubmitResult{SubmitStatus::stopped,
                                   "action executor unavailable"};
  }
  if (!submitted.accepted()) {
    emit_event("action_submit_failed",
               {{"utterance_id", result.utterance_id},
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

void VoiceFrontendRuntime::reset_pipeline(bool discontinuity, bool reset_preprocessor) {
  if (candidate_assembler_) candidate_assembler_->cancel_pending();
  const auto next = ring_ ? ring_->tail() : 0;
  if (ring_) ring_->reset(next);
  if (preprocessor_ && discontinuity && reset_preprocessor) preprocessor_->reset();
  if (vad_) vad_->reset();
  if (kws_) kws_->reset(next);
  if (segmenter_) segmenter_->reset(discontinuity);
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
    emit_event("config_state", {{"config", config_store_.to_public_json(config_)}});
    emit_event("sessions", session_list());
    emit_event("recording_state", {{"active", recorder_.active()}, {"session", recorder_.session_path().string()}});
    emit_event("runtime_mode", {{"mode", replay_mode_.load(std::memory_order_acquire) ? "replay" : "live"}});
    if (replay_) emit_event("replay_state", replay_->state(), 0, "replay");
    return {{"ok", true}, {"metrics", metrics_json()}};
  }
  if (action == "recording.start") {
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
    emit_event("recording_state", {{"active", true}, {"session", path.string()}});
    return {{"ok", true}, {"session", path.string()}};
  }
  if (action == "recording.stop") {
    recorder_.stop(metrics_json());
    emit_event("recording_state", {{"active", false}, {"session", recorder_.session_path().string()},
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
                                  segmentation_changed;

    std::unique_ptr<IAudioPreprocessor> new_preprocessor;
    if (aec_changed) {
      const auto timeline = make_preprocessor_timeline_config(active.aec);
      new_preprocessor = active.preprocess.aec_enabled && active.aec.enabled &&
                                 active.preprocess.implementation == "webrtc_aec3"
                             ? std::unique_ptr<IAudioPreprocessor>(
                                   std::make_unique<WebRtcAec3Preprocessor>(active.audio, timeline))
                             : std::unique_ptr<IAudioPreprocessor>(
                                   std::make_unique<BypassPreprocessor>(active.audio, timeline));
    }
    auto new_vad = vad_changed ? create_vad(active.vad) : nullptr;
    auto new_kws = kws_changed ? create_keyword_spotter(active.kws) : nullptr;
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
      if (pipeline_changed) {
        if (new_preprocessor) preprocessor_ = std::move(new_preprocessor);
        if (new_vad) vad_ = std::move(new_vad);
        if (new_kws) kws_ = std::move(new_kws);
        cancelled_candidates = segmenter_->pending_count();
        segmenter_->reconfigure(config_.segmentation);
        if (candidate_assembler_) candidate_assembler_->cancel_pending();
        const auto next = ring_->tail();
        kws_->reset(next);
        vad_->reset();
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
    const auto session = std::filesystem::path(command.value("session", ""));
    if (session.empty()) throw std::invalid_argument("session is required");
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
      reset_pipeline(true);
    }
    replay_mode_.store(true, std::memory_order_release);
    start_processing();
    if (!replay_) replay_ = std::make_unique<ReplayController>(
        [this](AudioPacket packet) { enqueue_replay_packet(std::move(packet)); });
    replay_->open(session);
    emit_event("runtime_mode", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
    emit_event("replay_state", replay_->state(), 0, "replay");
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
      reset_pipeline(true);
    }
    start_processing();
    start_captures();
    emit_event("runtime_mode", {{"mode", "live"}});
    return {{"ok", true}};
  }
  if (action == "replay.play" && replay_) { replay_->play(); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
  if (action == "replay.pause" && replay_) { replay_->pause(); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
  if (action == "replay.speed" && replay_) { replay_->set_speed(command.value("speed", 1.0)); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
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
        reset_pipeline(true);
      }
      replay_->seek_seconds(command.value("seconds", 0.0), resume_after_seek);
      start_processing();
    }
    emit_event("replay_state", replay_->state(), 0, "replay");
    return {{"ok", true}};
  }
  return {{"ok", false}, {"error", "unknown or unavailable action: " + action}};
}

nlohmann::json VoiceFrontendRuntime::session_list() const {
  nlohmann::json sessions = nlohmann::json::array();
  std::error_code error;
  if (std::filesystem::is_directory(config_.recording.session_root, error)) {
    for (const auto& entry : std::filesystem::directory_iterator(config_.recording.session_root, error)) {
      if (entry.is_directory() && std::filesystem::exists(entry.path() / "manifest.json")) {
        sessions.push_back({{"name", entry.path().filename().string()}, {"path", entry.path().string()}});
      }
    }
  }
  return {{"sessions", sessions}};
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
          {"assembly_outstanding", candidate_assembler_
                                       ? candidate_assembler_->outstanding()
                                       : 0},
          {"command_plans", command_plans_.load()},
          {"command_rejections", command_rejections_.load()},
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
