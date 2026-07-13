#include "dvo/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "dvo/json_utils.h"
#include "dvo/hash.h"

namespace dvo {

VoiceFrontendRuntime::VoiceFrontendRuntime(AppConfig config, ConfigStore store)
    : config_(std::move(config)), config_store_(std::move(store)),
      recorder_(std::max<std::size_t>(256,
          ((config_.recording.queue_capacity_ms + config_.audio.frame_ms - 1) /
           config_.audio.frame_ms) * 3 + 256)),
      debug_(8192) {}

VoiceFrontendRuntime::~VoiceFrontendRuntime() { stop(); }

void VoiceFrontendRuntime::initialize_pipeline() {
  std::scoped_lock lock(pipeline_mutex_);
  const auto capacity = static_cast<std::size_t>(config_.ring.duration_ms) * kProcessingSampleRate / 1000;
  ring_ = std::make_unique<TimedRingBuffer>(capacity);
  preprocessor_ = std::make_unique<BypassPreprocessor>(config_.audio);
  vad_ = create_vad(config_.vad);
  kws_ = create_keyword_spotter(config_.kws);
  segmenter_ = std::make_unique<UtteranceSegmenter>(config_.segmentation, *ring_);
  const auto queue_capacity = std::max<std::size_t>(
      1, (config_.audio.queue_capacity_ms + config_.audio.frame_ms - 1) / config_.audio.frame_ms);
  microphone_queue_ = std::make_unique<SpscQueue<AudioPacket>>(queue_capacity);
  loopback_queue_ = std::make_unique<SpscQueue<AudioPacket>>(queue_capacity);
  next_telemetry_sample_ = 0;
}

void VoiceFrontendRuntime::start_live(bool enable_web) {
  stop();
  initialize_pipeline();
  running_.store(true, std::memory_order_release);
  replay_mode_.store(false, std::memory_order_release);
  if (enable_web && config_.web.enabled) {
    debug_.start(config_.web, [this](const auto& command) { return handle_command(command); });
  }
  start_processing();
  start_captures();
  emit_event("runtime_started", {{"mode", "live"}, {"vad", vad_->status()}, {"kws", kws_->status()}});
  emit_event("config_state", {{"config", config_store_.to_public_json(config_)}});
  emit_event("sessions", session_list());
}

void VoiceFrontendRuntime::start_replay(const std::filesystem::path& session, bool enable_web,
                                        double speed) {
  stop();
  initialize_pipeline();
  running_.store(true, std::memory_order_release);
  replay_mode_.store(true, std::memory_order_release);
  if (enable_web && config_.web.enabled) {
    debug_.start(config_.web, [this](const auto& command) { return handle_command(command); });
  }
  start_processing();
  replay_ = std::make_unique<ReplayController>([this](AudioPacket packet) { enqueue_packet(std::move(packet)); });
  replay_->open(session);
  replay_->set_speed(speed);
  replay_->play();
  emit_event("runtime_started", {{"mode", "replay"}, {"session", session.string()}}, 0, "replay");
  emit_event("config_state", {{"config", config_store_.to_public_json(config_)}}, 0, "replay");
  emit_event("sessions", session_list(), 0, "replay");
}

nlohmann::json VoiceFrontendRuntime::run_benchmark(const std::filesystem::path& session) {
  const auto started = std::chrono::steady_clock::now();
  start_replay(session, false, 0.0);
  while (running() && replay_) {
    const auto state = replay_->state();
    const bool replay_done = !state.value("playing", false) &&
                             state.value("cursor", 0ULL) >= state.value("entries", 0ULL);
    if (replay_done && microphone_queue_->size() == 0 && loopback_queue_->size() == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  auto metrics = metrics_json();
  metrics["elapsed_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started).count();
  metrics["session"] = session.string();
  stop();
  return metrics;
}

void VoiceFrontendRuntime::start_processing() {
  processing_thread_ = std::jthread([this](std::stop_token stop) { processing_loop(stop); });
}

void VoiceFrontendRuntime::start_captures() {
  const auto event = [this](std::string type, std::string message) {
    queue_capture_event(std::move(type), std::move(message));
  };
  microphone_capture_.start(
      {AudioStreamKind::microphone, config_.audio.microphone_device, config_.audio.follow_default_device},
      [this](AudioPacket packet) { enqueue_packet(std::move(packet)); }, event);
  loopback_capture_.start(
      {AudioStreamKind::loopback, config_.audio.loopback_device, config_.audio.follow_default_device},
      [this](AudioPacket packet) { enqueue_packet(std::move(packet)); }, event);
}

void VoiceFrontendRuntime::stop_captures() {
  microphone_capture_.stop();
  loopback_capture_.stop();
}

void VoiceFrontendRuntime::stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel) && !processing_thread_.joinable()) return;
  stop_captures();
  if (replay_) { replay_->stop(); replay_.reset(); }
  if (processing_thread_.joinable()) { processing_thread_.request_stop(); processing_thread_.join(); }
  recorder_.stop(metrics_json());
  debug_.stop();
}

void VoiceFrontendRuntime::enqueue_packet(AudioPacket packet) {
  audio_packets_.fetch_add(1, std::memory_order_relaxed);
  auto& queue = packet.stream == AudioStreamKind::microphone ? *microphone_queue_ : *loopback_queue_;
  auto& overflow = packet.stream == AudioStreamKind::microphone ? microphone_overflow_ : loopback_overflow_;
  if (replay_mode_.load(std::memory_order_acquire)) {
    while (running_.load(std::memory_order_acquire) && !queue.try_push(packet)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return;
  }
  if (!queue.try_push(std::move(packet))) {
    overflow.store(true, std::memory_order_release);
    audio_queue_drops_.fetch_add(1, std::memory_order_relaxed);
  }
}

void VoiceFrontendRuntime::processing_loop(std::stop_token stop) {
  while (!stop.stop_requested()) {
    drain_capture_events();
    bool worked = false;
    AudioPacket loopback;
    while (loopback_queue_->try_pop(loopback)) {
      worked = true;
      if (loopback_overflow_.exchange(false)) loopback.discontinuity = true;
      latest_loopback_summary_ = summarize(loopback.samples);
      if (recorder_.active()) recorder_.try_enqueue(loopback);
      latest_loopback_ = std::move(loopback);
    }
    AudioPacket microphone;
    if (microphone_queue_->try_pop(microphone)) {
      worked = true;
      if (microphone_overflow_.exchange(false)) microphone.discontinuity = true;
      process_microphone(std::move(microphone));
    }
    if (!worked) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void VoiceFrontendRuntime::process_microphone(AudioPacket packet) {
  const auto mic_summary = summarize(packet.samples);
  if (recorder_.active()) recorder_.try_enqueue(packet);
  std::scoped_lock lock(pipeline_mutex_);
  const auto processed = preprocessor_->process(packet, latest_loopback_);
  if (processed.reset_required || packet.discontinuity) reset_pipeline(true);
  for (const auto& frame : processed.frames) process_frame(frame, mic_summary, latest_loopback_summary_);
}

void VoiceFrontendRuntime::process_frame(const NormalizedFrame& frame,
                                         const SignalSummary& microphone,
                                         const SignalSummary& loopback) {
  ring_->push(frame.first_sample, frame.samples);
  processed_frames_.fetch_add(1, std::memory_order_relaxed);
  if (recorder_.active()) recorder_.try_enqueue(frame);

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
    segmenter_->add_kws_hit(std::move(*hit));
  }

  auto segmented = segmenter_->advance(frame.first_sample + frame.samples.size());
  for (auto& candidate : segmented.candidates) {
    candidates_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate", candidate_json(candidate), frame.first_sample,
               replay_mode_ ? "replay" : "live");
    if (recorder_.active()) recorder_.try_enqueue(RecordCandidate{std::move(candidate)});
  }
  for (const auto& rejection : segmented.rejections) {
    rejections_.fetch_add(1, std::memory_order_relaxed);
    emit_event("candidate_rejected", {{"reason", rejection}}, frame.first_sample,
               replay_mode_ ? "replay" : "live");
  }

  if (frame.first_sample >= next_telemetry_sample_) {
    const auto processed_summary = summarize(frame);
    emit_event("telemetry",
               {{"sample", frame.first_sample},
                {"microphone", {{"min", microphone.min}, {"max", microphone.max}, {"rms", microphone.rms}}},
                {"loopback", {{"min", loopback.min}, {"max", loopback.max}, {"rms", loopback.rms}}},
                {"processed", {{"min", processed_summary.min}, {"max", processed_summary.max}, {"rms", processed_summary.rms}}},
                {"vad", vad_update.speech}, {"queue_depth", microphone_queue_->size()},
                {"telemetry_dropped", debug_.dropped()}, {"audio_queue_drops", audio_queue_drops_.load()}},
               frame.first_sample, replay_mode_ ? "replay" : "live");
    const auto interval = std::max<std::uint64_t>(kFrameSamples,
        kProcessingSampleRate / std::max<std::uint32_t>(1, config_.web.telemetry_hz));
    next_telemetry_sample_ = frame.first_sample + interval;
  }
}

void VoiceFrontendRuntime::reset_pipeline(bool discontinuity) {
  const auto next = ring_ ? ring_->tail() : 0;
  if (ring_) ring_->reset(next);
  if (preprocessor_ && discontinuity) preprocessor_->reset();
  if (vad_) vad_->reset();
  if (kws_) kws_->reset(next);
  if (segmenter_) segmenter_->reset(discontinuity);
  if (discontinuity) discontinuities_.fetch_add(1, std::memory_order_relaxed);
}

void VoiceFrontendRuntime::emit_event(std::string type, nlohmann::json payload,
                                      std::uint64_t timestamp_sample, std::string source) {
  if (recorder_.active() && type != "telemetry") {
    recorder_.try_enqueue(RecordEvent{{{"type", type}, {"timestamp_sample", timestamp_sample},
                                       {"source", source}, {"payload", payload}}});
  }
  debug_.publish(std::move(type), std::move(payload), timestamp_sample, std::move(source));
}

void VoiceFrontendRuntime::queue_capture_event(std::string type, std::string message) {
  std::scoped_lock lock(capture_events_mutex_);
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
  const auto action = command.value("action", "");
  if (action == "state.get") {
    emit_event("config_state", {{"config", config_store_.to_public_json(config_)}});
    emit_event("sessions", session_list());
    emit_event("recording_state", {{"active", recorder_.active()}, {"session", recorder_.session_path().string()}});
    if (replay_) emit_event("replay_state", replay_->state(), 0, "replay");
    return {{"ok", true}, {"metrics", metrics_json()}};
  }
  if (action == "recording.start") {
    const auto model_file = [](const std::filesystem::path& path) {
      nlohmann::json value{{"path", path.string()}, {"present", std::filesystem::is_regular_file(path)}};
      if (value["present"].get<bool>()) value["sha256"] = sha256_file(path);
      return value;
    };
    auto manifest = nlohmann::json{
        {"config", config_store_.to_manifest_json(config_)},
        {"models", {{"sherpa_onnx_version", "1.13.2"},
                    {"kws", {{"encoder", model_file(config_.kws.encoder)},
                             {"decoder", model_file(config_.kws.decoder)},
                             {"joiner", model_file(config_.kws.joiner)},
                             {"tokens", model_file(config_.kws.tokens)},
                             {"keywords", model_file(config_.kws.keywords)}}},
                    {"vad", model_file(config_.vad.model)}}},
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
    active.ring = current.ring;
    active.recording = current.recording;
    active.kws = current.kws;
    active.kws.threshold = candidate.kws.threshold;
    active.kws.boosting_score = candidate.kws.boosting_score;
    active.vad = current.vad;
    active.vad.threshold = candidate.vad.threshold;
    active.vad.min_speech_ms = candidate.vad.min_speech_ms;
    active.vad.min_silence_ms = candidate.vad.min_silence_ms;
    active.vad.max_speech_ms = candidate.vad.max_speech_ms;
    active.web = current.web;
    active.web.telemetry_hz = candidate.web.telemetry_hz;

    auto new_vad = create_vad(active.vad);
    auto new_kws = create_keyword_spotter(active.kws);
    if (save) config_store_.save_overrides(candidate);
    std::size_t cancelled_candidates{};
    {
      std::scoped_lock pipeline_lock(pipeline_mutex_);
      std::scoped_lock config_lock(config_mutex_);
      config_ = active;
      vad_ = std::move(new_vad);
      kws_ = std::move(new_kws);
      cancelled_candidates = segmenter_->pending_count();
      segmenter_->reconfigure(config_.segmentation);
      const auto next = ring_->tail();
      kws_->reset(next);
      vad_->reset();
    }
    if (cancelled_candidates != 0) {
      emit_event("candidate_cancelled",
                 {{"reason", "config_hot_swap"}, {"count", cancelled_candidates}});
    }
    emit_event("config_applied", {{"config", config_store_.to_public_json(active)},
                                   {"saved", save}, {"restart_required", restart_required},
                                   {"saved_config", config_store_.to_public_json(candidate)}});
    return {{"ok", true}, {"config", config_store_.to_public_json(active)},
            {"saved", save}, {"restart_required", restart_required}};
  }
  if (action == "replay.open") {
    const auto session = std::filesystem::path(command.value("session", ""));
    if (session.empty()) throw std::invalid_argument("session is required");
    stop_captures();
    replay_mode_.store(true, std::memory_order_release);
    {
      std::scoped_lock lock(pipeline_mutex_);
      reset_pipeline(false);
    }
    if (!replay_) replay_ = std::make_unique<ReplayController>(
        [this](AudioPacket packet) { enqueue_packet(std::move(packet)); });
    replay_->open(session);
    emit_event("replay_state", replay_->state(), 0, "replay");
    return {{"ok", true}};
  }
  if (action == "replay.play" && replay_) { replay_->play(); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
  if (action == "replay.pause" && replay_) { replay_->pause(); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
  if (action == "replay.speed" && replay_) { replay_->set_speed(command.value("speed", 1.0)); emit_event("replay_state", replay_->state(), 0, "replay"); return {{"ok", true}}; }
  if (action == "replay.seek" && replay_) {
    std::scoped_lock lock(pipeline_mutex_);
    reset_pipeline(false);
    replay_->seek_seconds(command.value("seconds", 0.0));
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
  return {{"audio_packets", audio_packets_.load()}, {"processed_frames", processed_frames_.load()},
          {"kws_hits", kws_hits_.load()}, {"candidates", candidates_.load()},
          {"rejections", rejections_.load()}, {"audio_queue_drops", audio_queue_drops_.load()},
          {"discontinuities", discontinuities_.load()}, {"telemetry_dropped", debug_.dropped()},
          {"recording_incomplete", recorder_.incomplete()}};
}

VoiceFrontendRuntime::SignalSummary VoiceFrontendRuntime::summarize(const std::vector<float>& values) {
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
  return summarize(std::vector<float>(frame.samples.begin(), frame.samples.end()));
}

}  // namespace dvo
