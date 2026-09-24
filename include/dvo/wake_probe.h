#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/ring_buffer.h"
#include "dvo/streaming_recognizer.h"

namespace dvo {

// Second-stage wake detection. While no activation is running, speech
// segments that produced no KWS hit are decoded by an offline recognizer and
// searched for the wake word. A match is reported with the wake span
// estimated from where the word sits in the text; the runtime then feeds it
// through the ordinary keyword path, so candidate assembly, the exact-final
// decode and strict parsing are unchanged.
struct WakeProbeRequest {
  std::uint64_t id{};
  SampleSpan speech;  // VAD speech span, processing sample clock
  SampleSpan audio;   // span actually decoded (speech plus margins)
};

struct WakeProbeResult {
  std::uint64_t id{};
  SampleSpan speech;
  bool matched{};
  bool exact{};
  std::string keyword;
  std::optional<SampleSpan> wake_span;
  // Recognized text. Probes run on ordinary conversation, so the runtime
  // publishes it only for replayed sessions, never for live audio.
  std::string text;
  double decode_ms{};
  std::string error;
};

// Maps the wake word's character position in the text onto the speech span,
// assuming a uniform speaking rate.
[[nodiscard]] SampleSpan estimate_wake_span(SampleSpan speech, std::size_t begin,
                                            std::size_t end, std::size_t length);

class WakeProbe {
 public:
  using EngineLoader = std::function<std::unique_ptr<IOfflineAsrEngine>()>;

  // The ring must outlive the probe. The engine is loaded on the worker
  // thread so construction never blocks the audio path.
  WakeProbe(EngineLoader loader, const TimedRingBuffer& ring,
            std::vector<std::string> wake_words, std::size_t queue_capacity = 2);
  ~WakeProbe();
  WakeProbe(const WakeProbe&) = delete;
  WakeProbe& operator=(const WakeProbe&) = delete;

  // Never blocks; returns false when the bounded queue is full.
  [[nodiscard]] bool try_submit(WakeProbeRequest request);
  [[nodiscard]] bool try_pop(WakeProbeResult& result);
  // No request is queued or being decoded (results may await try_pop).
  [[nodiscard]] bool idle() const;
  // Drops queued requests and undelivered results, e.g. on a pipeline reset.
  void clear();
  [[nodiscard]] std::string status() const;

 private:
  void run(std::stop_token stop);
  [[nodiscard]] WakeProbeResult decode(const WakeProbeRequest& request);

  EngineLoader loader_;
  const TimedRingBuffer& ring_;
  std::vector<std::string> wake_words_;
  std::size_t queue_capacity_;
  std::unique_ptr<IOfflineAsrEngine> engine_;
  mutable std::mutex mutex_;
  std::condition_variable_any cv_;
  std::deque<WakeProbeRequest> requests_;
  std::deque<WakeProbeResult> results_;
  bool busy_{};
  std::uint64_t epoch_{};
  std::string status_{"loading"};
  std::jthread worker_;
};

}  // namespace dvo
