#include "dvo/recorder.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "dvo/json_utils.h"

namespace dvo {
namespace {

std::string session_name() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_s(&local, &time);
  const auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  std::ostringstream result;
  result << std::put_time(&local, "%Y%m%d-%H%M%S") << '-' << std::hex << (ticks & 0xffffff);
  return result.str();
}

}  // namespace

SessionRecorder::SessionRecorder(std::size_t queue_capacity) : queue_(queue_capacity) {
  thread_ = std::jthread([this](std::stop_token stop) {
    while (!stop.stop_requested()) {
      RecordItem item;
      if (!queue_.try_pop(item)) {
        std::unique_lock lock(state_mutex_);
        state_cv_.wait_for(lock, std::chrono::milliseconds(20));
        continue;
      }
      inflight_.fetch_add(1, std::memory_order_acq_rel);
      try { std::visit([this](const auto& value) { write_item(value); }, item); }
      catch (...) { incomplete_.store(true, std::memory_order_release); }
      inflight_.fetch_sub(1, std::memory_order_acq_rel);
      state_cv_.notify_all();
    }
  });
}

SessionRecorder::~SessionRecorder() {
  stop();
  thread_.request_stop();
  state_cv_.notify_all();
}

std::filesystem::path SessionRecorder::start(const RecordingConfig& config,
                                             const nlohmann::json& manifest) {
  std::scoped_lock lock(state_mutex_);
  if (active_ || closing_) throw std::runtime_error("a recording session is already active");
  session_path_ = config.session_root / session_name();
  std::filesystem::create_directories(session_path_ / "candidates");
  timeline_.open(session_path_ / "timeline.ndjson", std::ios::binary | std::ios::trunc);
  events_.open(session_path_ / "events.ndjson", std::ios::binary | std::ios::trunc);
  if (!timeline_ || !events_) throw std::runtime_error("cannot create recording session files");
  manifest_ = manifest;
  manifest_["session_id"] = session_path_.filename().string();
  manifest_["complete"] = false;
  manifest_["format"] = "dvo-session-v1";
  mic_stats_ = {};
  loopback_stats_ = {};
  processed_stats_ = {};
  dropped_items_.store(0, std::memory_order_release);
  incomplete_.store(false, std::memory_order_release);
  active_.store(true, std::memory_order_release);
  return session_path_;
}

void SessionRecorder::stop(const nlohmann::json& final_metrics) {
  if (!active_.exchange(false, std::memory_order_acq_rel) && !closing_) return;
  closing_.store(true, std::memory_order_release);
  state_cv_.notify_all();
  std::unique_lock lock(state_mutex_);
  state_cv_.wait(lock, [this] { return queue_.size() == 0 && inflight_.load() == 0; });
  if (!final_metrics.empty()) manifest_["final_metrics"] = final_metrics;
  close_files();
  closing_.store(false, std::memory_order_release);
}

bool SessionRecorder::try_enqueue(RecordItem item) {
  if (!active()) return false;
  std::unique_lock lock(enqueue_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    dropped_items_.fetch_add(1, std::memory_order_relaxed);
    incomplete_.store(true, std::memory_order_release);
    return false;
  }
  if (!queue_.try_push(std::move(item))) {
    dropped_items_.fetch_add(1, std::memory_order_relaxed);
    incomplete_.store(true, std::memory_order_release);
    return false;
  }
  state_cv_.notify_one();
  return true;
}

std::filesystem::path SessionRecorder::session_path() const {
  std::scoped_lock lock(state_mutex_);
  return session_path_;
}

void SessionRecorder::run() {
  // Kept as a private ABI anchor; the worker body lives in the jthread lambda so
  // the stop token is never read through a concurrently assigned jthread object.
}

void SessionRecorder::write_item(const AudioPacket& packet) {
  std::scoped_lock lock(state_mutex_);
  std::unique_ptr<FloatWavWriter>* target =
      packet.stream == AudioStreamKind::microphone ? &mic_ : &loopback_;
  const auto filename = packet.stream == AudioStreamKind::microphone ? "mic.wav" : "loopback.wav";
  if (!*target) {
    *target = std::make_unique<FloatWavWriter>();
    (*target)->open(session_path_ / filename, packet.format);
  }
  if ((*target)->format().sample_rate != packet.format.sample_rate ||
      (*target)->format().channels != packet.format.channels) {
    incomplete_.store(true, std::memory_order_release);
    return;
  }
  const auto offset = (*target)->frames_written();
  (*target)->write(packet.samples);
  auto& stats = packet.stream == AudioStreamKind::microphone ? mic_stats_ : loopback_stats_;
  stats.format = packet.format;
  stats.frames += packet.samples.size() / packet.format.channels;
  if (stats.packets == 0) stats.first_qpc_100ns = packet.qpc_100ns;
  stats.last_qpc_100ns = packet.qpc_100ns;
  ++stats.packets;
  if (packet.synthetic) ++stats.synthetic_packets;
  timeline_ << nlohmann::json{{"kind", to_string(packet.stream)}, {"offset_frames", offset},
                              {"frame_count", packet.samples.size() / packet.format.channels},
                              {"qpc_100ns", packet.qpc_100ns}, {"device_position", packet.device_position},
                              {"silent", packet.silent}, {"discontinuity", packet.discontinuity},
                              {"synthetic", packet.synthetic}}.dump() << '\n';
}

void SessionRecorder::write_item(const NormalizedFrame& frame) {
  std::scoped_lock lock(state_mutex_);
  if (!processed_) {
    processed_ = std::make_unique<FloatWavWriter>();
    processed_->open(session_path_ / "processed.wav", {kProcessingSampleRate, 1});
  }
  const auto offset = processed_->frames_written();
  processed_->write(frame.samples);
  processed_stats_.format = {kProcessingSampleRate, 1};
  processed_stats_.frames += frame.samples.size();
  if (processed_stats_.packets == 0) processed_stats_.first_qpc_100ns = frame.qpc_100ns;
  processed_stats_.last_qpc_100ns = frame.qpc_100ns;
  ++processed_stats_.packets;
  timeline_ << nlohmann::json{{"kind", "processed"}, {"offset_frames", offset},
                              {"frame_count", frame.samples.size()}, {"first_sample", frame.first_sample},
                              {"qpc_100ns", frame.qpc_100ns}, {"discontinuity", frame.discontinuity}}.dump() << '\n';
}

void SessionRecorder::write_item(const RecordEvent& event) {
  std::scoped_lock lock(state_mutex_);
  events_ << event.value.dump() << '\n';
}

void SessionRecorder::write_item(const RecordCandidate& item) {
  std::scoped_lock lock(state_mutex_);
  FloatWavWriter writer;
  writer.open(session_path_ / "candidates" / (item.value.utterance_id + ".wav"),
              {item.value.sample_rate, 1});
  writer.write(item.value.pcm);
  writer.close();
  events_ << nlohmann::json{{"type", "candidate"}, {"payload", candidate_json(item.value)}}.dump() << '\n';
}

void SessionRecorder::close_files() {
  if (mic_) mic_->close();
  if (loopback_) loopback_->close();
  if (processed_) processed_->close();
  mic_.reset();
  loopback_.reset();
  processed_.reset();
  timeline_.close();
  events_.close();
  if (!session_path_.empty()) {
    const auto stream_json = [](const StreamStats& stats) {
      return nlohmann::json{{"present", stats.packets != 0},
                            {"sample_rate", stats.format.sample_rate},
                            {"channels", stats.format.channels},
                            {"frames", stats.frames}, {"packets", stats.packets},
                            {"first_qpc_100ns", stats.first_qpc_100ns},
                            {"last_qpc_100ns", stats.last_qpc_100ns},
                            {"synthetic_packets", stats.synthetic_packets}};
    };
    manifest_["streams"] = {{"microphone", stream_json(mic_stats_)},
                             {"loopback", stream_json(loopback_stats_)},
                             {"processed", stream_json(processed_stats_)}};
    manifest_["recording_queue_drops"] = dropped_items_.load(std::memory_order_acquire);
    manifest_["complete"] = !incomplete();
    std::ofstream manifest(session_path_ / "manifest.json", std::ios::binary | std::ios::trunc);
    manifest << manifest_.dump(2);
  }
}

}  // namespace dvo
