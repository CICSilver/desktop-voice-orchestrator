#include "dvo/wake_probe.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include "dvo/command_parser.h"

namespace dvo {

SampleSpan estimate_wake_span(SampleSpan speech, std::size_t begin, std::size_t end,
                              std::size_t length) {
  if (length == 0 || end <= begin || speech.end <= speech.start) return speech;
  const auto duration = speech.end - speech.start;
  const auto at = [&](std::size_t index) {
    return speech.start + duration * std::min(index, length) / length;
  };
  return {at(begin), std::max(at(begin) + 1, at(end))};
}

WakeProbe::WakeProbe(EngineLoader loader, const TimedRingBuffer& ring,
                     std::vector<std::string> wake_words, std::size_t queue_capacity)
    : loader_(std::move(loader)),
      ring_(ring),
      wake_words_(std::move(wake_words)),
      queue_capacity_(std::max<std::size_t>(1, queue_capacity)) {
  worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

WakeProbe::~WakeProbe() {
  worker_.request_stop();
  cv_.notify_all();
}

bool WakeProbe::try_submit(WakeProbeRequest request) {
  {
    std::scoped_lock lock(mutex_);
    if (requests_.size() >= queue_capacity_) return false;
    requests_.push_back(std::move(request));
  }
  cv_.notify_all();
  return true;
}

bool WakeProbe::try_pop(WakeProbeResult& result) {
  std::scoped_lock lock(mutex_);
  if (results_.empty()) return false;
  result = std::move(results_.front());
  results_.pop_front();
  return true;
}

bool WakeProbe::idle() const {
  std::scoped_lock lock(mutex_);
  return requests_.empty() && !busy_;
}

void WakeProbe::clear() {
  std::scoped_lock lock(mutex_);
  requests_.clear();
  results_.clear();
  ++epoch_;
}

std::string WakeProbe::status() const {
  std::scoped_lock lock(mutex_);
  return status_;
}

void WakeProbe::run(std::stop_token stop) {
  try {
    engine_ = loader_ ? loader_() : nullptr;
    std::scoped_lock lock(mutex_);
    status_ = engine_ ? "ready: " + engine_->name() : "unavailable: no engine";
  } catch (const std::exception& error) {
    std::scoped_lock lock(mutex_);
    status_ = std::string{"unavailable: "} + error.what();
  } catch (...) {
    std::scoped_lock lock(mutex_);
    status_ = "unavailable";
  }

  while (!stop.stop_requested()) {
    WakeProbeRequest request;
    std::uint64_t epoch{};
    {
      std::unique_lock lock(mutex_);
      if (!cv_.wait(lock, stop, [this] { return !requests_.empty(); })) break;
      request = std::move(requests_.front());
      requests_.pop_front();
      busy_ = true;
      epoch = epoch_;
    }
    auto result = decode(request);
    {
      std::scoped_lock lock(mutex_);
      busy_ = false;
      if (epoch == epoch_) results_.push_back(std::move(result));
    }
  }
}

WakeProbeResult WakeProbe::decode(const WakeProbeRequest& request) {
  WakeProbeResult result;
  result.id = request.id;
  result.speech = request.speech;
  if (!engine_) {
    result.error = "wake probe engine unavailable";
    return result;
  }
  try {
    const auto slice = ring_.slice_cooperative(request.audio);
    if (slice.samples.empty()) {
      result.error = "probe audio is no longer in the ring buffer";
      return result;
    }
    const auto started = std::chrono::steady_clock::now();
    result.text = engine_->decode(slice.samples, kProcessingSampleRate).text;
    result.decode_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    for (const auto& wake : wake_words_) {
      const auto match = find_wake_in_text(result.text, wake);
      if (!match) continue;
      result.matched = true;
      result.exact = match->exact;
      result.keyword = wake;
      result.wake_span =
          estimate_wake_span(request.speech, match->begin, match->end, match->length);
      break;
    }
  } catch (const std::exception& error) {
    result.error = error.what();
  } catch (...) {
    result.error = "wake probe decode failed";
  }
  return result;
}

}  // namespace dvo
