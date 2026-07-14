#include "dvo/replay.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "dvo/wav_file.h"

namespace dvo {

ReplayController::ReplayController(PacketCallback callback) : callback_(std::move(callback)) {
  thread_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

ReplayController::~ReplayController() { stop(); }

void ReplayController::open(const std::filesystem::path& session) {
  std::scoped_lock lock(mutex_);
  playing_ = false;
  dispatching_ = false;
  session_ = session;
  cursor_ = 0;
  seek_target_.reset();
  load_timeline();
}

void ReplayController::play() {
  std::scoped_lock lock(mutex_);
  if (entries_.empty()) throw std::runtime_error("no replay session is open");
  playing_ = true;
  cv_.notify_all();
}

void ReplayController::pause() {
  std::scoped_lock lock(mutex_);
  playing_ = false;
}

void ReplayController::seek_seconds(double seconds) {
  std::scoped_lock lock(mutex_);
  if (entries_.empty()) return;
  const auto target = entries_.front().qpc_100ns + static_cast<std::uint64_t>(std::max(0.0, seconds) * 10000000.0);
  const auto target_cursor = static_cast<std::size_t>(std::lower_bound(entries_.begin(), entries_.end(), target,
      [](const TimelineEntry& entry, std::uint64_t value) { return entry.qpc_100ns < value; }) - entries_.begin());
  resume_after_seek_ = playing_;
  cursor_ = 0;
  seek_target_ = target_cursor;
  playing_ = true;  // Recompute state from the beginning at unlimited speed.
  cv_.notify_all();
}

void ReplayController::seek_seconds(double seconds, bool resume_after_seek) {
  std::scoped_lock lock(mutex_);
  if (entries_.empty()) return;
  const auto target = entries_.front().qpc_100ns +
                      static_cast<std::uint64_t>(std::max(0.0, seconds) * 10000000.0);
  const auto target_cursor = static_cast<std::size_t>(
      std::lower_bound(entries_.begin(), entries_.end(), target,
                       [](const TimelineEntry& entry, std::uint64_t value) {
                         return entry.qpc_100ns < value;
                       }) -
      entries_.begin());
  resume_after_seek_ = resume_after_seek;
  cursor_ = 0;
  seek_target_ = target_cursor;
  playing_ = true;
  cv_.notify_all();
}

void ReplayController::set_speed(double speed) {
  if (speed != 0.0 && speed != 0.5 && speed != 1.0 && speed != 2.0) {
    throw std::invalid_argument("replay speed must be 0, 0.5, 1 or 2");
  }
  std::scoped_lock lock(mutex_);
  speed_ = speed;
}

bool ReplayController::wait_until_finished(std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  return cv_.wait_for(lock, timeout, [this] {
    return !playing_ && !dispatching_ && cursor_ >= entries_.size();
  });
}

bool ReplayController::wait_until_quiescent(std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  return cv_.wait_for(lock, timeout, [this] { return !dispatching_; });
}

void ReplayController::stop() {
  if (thread_.joinable()) {
    thread_.request_stop();
    cv_.notify_all();
    thread_.join();
  }
}

nlohmann::json ReplayController::state() const {
  std::scoped_lock lock(mutex_);
  double duration_seconds{};
  double position_seconds{};
  if (!entries_.empty()) {
    duration_seconds = static_cast<double>(entries_.back().qpc_100ns - entries_.front().qpc_100ns) /
                       10000000.0;
    const auto index = std::min(cursor_, entries_.size() - 1);
    position_seconds = static_cast<double>(entries_[index].qpc_100ns - entries_.front().qpc_100ns) /
                       10000000.0;
  }
  return {{"session", session_.string()}, {"playing", playing_},
          {"dispatching", dispatching_}, {"speed", speed_},
          {"cursor", cursor_}, {"entries", entries_.size()}, {"seeking", seek_target_.has_value()},
          {"duration_seconds", duration_seconds}, {"position_seconds", position_seconds}};
}

void ReplayController::load_timeline() {
  entries_.clear();
  std::ifstream input(session_ / "timeline.ndjson", std::ios::binary);
  if (!input) throw std::runtime_error("session timeline.ndjson is missing");
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto value = nlohmann::json::parse(line);
    const auto kind = value.value("kind", "");
    if (kind != "microphone" && kind != "loopback") continue;
    entries_.push_back({kind == "microphone" ? AudioStreamKind::microphone : AudioStreamKind::loopback,
                        value.at("offset_frames").get<std::uint64_t>(),
                        value.at("frame_count").get<std::uint64_t>(),
                        value.at("qpc_100ns").get<std::uint64_t>(),
                        value.value("arrival_qpc_100ns", value.at("qpc_100ns").get<std::uint64_t>()),
                        value.value("device_position", 0ULL), value.value("stream_epoch", 0ULL),
                        value.value("sequence", 0ULL), value.value("silent", false),
                        value.value("discontinuity", false), value.value("timestamp_error", false),
                        value.value("synthetic", false)});
  }
  std::stable_sort(entries_.begin(), entries_.end(),
                   [](const auto& a, const auto& b) { return a.qpc_100ns < b.qpc_100ns; });
}

void ReplayController::run(std::stop_token stop) {
  std::unique_ptr<FloatWavReader> mic;
  std::unique_ptr<FloatWavReader> loopback;
  std::filesystem::path reader_session;
  std::uint64_t previous_qpc{};
  while (!stop.stop_requested()) {
    TimelineEntry entry;
    std::filesystem::path session;
    double speed{};
    {
      std::unique_lock lock(mutex_);
      cv_.wait_for(lock, std::chrono::milliseconds(20), [this, stop] {
        return stop.stop_requested() || (playing_ && cursor_ < entries_.size());
      });
      if (stop.stop_requested()) break;
      if (!playing_ || cursor_ >= entries_.size()) continue;
      dispatching_ = true;
      entry = entries_[cursor_++];
      session = session_;
      speed = seek_target_ ? 0.0 : speed_;
      if (seek_target_ && cursor_ >= *seek_target_) {
        seek_target_.reset();
        playing_ = resume_after_seek_;
      }
      if (cursor_ >= entries_.size()) playing_ = false;
    }
    try {
      if (session != reader_session) {
        mic.reset();
        loopback.reset();
        reader_session = session;
        previous_qpc = 0;
      }
      auto& reader = entry.kind == AudioStreamKind::microphone ? mic : loopback;
      if (!reader) reader = std::make_unique<FloatWavReader>(
          session / (entry.kind == AudioStreamKind::microphone ? "mic.wav" : "loopback.wav"));
      if (speed > 0.0 && previous_qpc != 0 && entry.qpc_100ns > previous_qpc) {
        const auto delay = static_cast<std::uint64_t>((entry.qpc_100ns - previous_qpc) / speed);
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay * 100));
      }
      previous_qpc = entry.qpc_100ns;
      AudioPacket packet;
      packet.stream = entry.kind;
      packet.format = reader->format();
      packet.samples = reader->read_frames(entry.offset_frames, entry.frame_count);
      packet.qpc_100ns = entry.qpc_100ns;
      packet.arrival_qpc_100ns = entry.arrival_qpc_100ns;
      packet.device_position = entry.device_position;
      packet.stream_epoch = entry.stream_epoch;
      packet.sequence = entry.sequence;
      packet.silent = entry.silent;
      packet.discontinuity = entry.discontinuity;
      packet.timestamp_error = entry.timestamp_error;
      packet.synthetic = entry.synthetic;
      callback_(std::move(packet));
      {
        std::scoped_lock lock(mutex_);
        dispatching_ = false;
        cv_.notify_all();
      }
    } catch (...) {
      std::scoped_lock lock(mutex_);
      playing_ = false;
      dispatching_ = false;
      cv_.notify_all();
    }
  }
}

}  // namespace dvo
