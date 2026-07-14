#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dvo {

inline constexpr std::uint32_t kProcessingSampleRate = 16000;
inline constexpr std::size_t kFrameSamples = 160;

enum class AudioStreamKind { microphone, loopback, processed };

struct AudioFormat {
  std::uint32_t sample_rate{};
  std::uint16_t channels{};
};

struct AudioPacket {
  AudioStreamKind stream{AudioStreamKind::microphone};
  AudioFormat format{};
  std::vector<float> samples;
  // QPC timestamp of the first sample as reported by WASAPI. The unit is
  // 100 ns, matching IAudioCaptureClient::GetBuffer.
  std::uint64_t qpc_100ns{};
  // QPC timestamp at which the packet entered the application. Keeping this
  // separate from qpc_100ns makes WebRTC's stream-delay calculation
  // reproducible and allows deterministic replay to use a virtual clock.
  std::uint64_t arrival_qpc_100ns{};
  std::uint64_t device_position{};
  // Incremented whenever an endpoint is reopened. A different epoch means
  // device_position and all stateful preprocessing must be re-anchored.
  std::uint64_t stream_epoch{};
  // Monotonic within an epoch. Zero is also the legacy/default value, so gap
  // detection is enabled after the first non-zero sequence is observed.
  std::uint64_t sequence{};
  bool silent{};
  bool discontinuity{};
  // AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR. A packet with this flag must not be
  // used to update clock/drift estimates.
  bool timestamp_error{};
  // Loopback devices do not deliver packets while the render endpoint is idle.
  // Such gaps are filled with format-correct silence and remain distinguishable
  // in the recording timeline and deterministic replay.
  bool synthetic{};
};

struct NormalizedFrame {
  std::uint64_t first_sample{};
  std::uint64_t qpc_100ns{};
  std::array<float, kFrameSamples> samples{};
  bool discontinuity{};
};

struct SampleSpan {
  std::uint64_t start{};
  std::uint64_t end{};

  [[nodiscard]] std::uint64_t size() const { return end > start ? end - start : 0; }
  [[nodiscard]] bool empty() const { return end <= start; }
  [[nodiscard]] bool overlaps(const SampleSpan& other) const {
    return start < other.end && other.start < end;
  }
  bool operator==(const SampleSpan&) const = default;
};

struct KwsHit {
  std::string keyword;
  std::vector<std::string> tokens;
  std::vector<std::uint64_t> token_samples;
  SampleSpan wake_span;
  std::uint64_t detected_at_sample{};
};

struct VadInterval {
  SampleSpan span;
};

enum class WakePosition { prefix, suffix, embedded };

struct UtteranceCandidate {
  std::string utterance_id;
  WakePosition position{WakePosition::prefix};
  std::string keyword;
  std::vector<std::string> tokens;
  SampleSpan wake_span;
  std::vector<SampleSpan> source_spans;
  std::vector<float> pcm;
  std::uint32_t sample_rate{kProcessingSampleRate};
  std::string boundary_source{"vad"};
  bool truncated{};
  bool timed_out{};
  bool discontinuity{};
};

[[nodiscard]] inline const char* to_string(AudioStreamKind kind) {
  switch (kind) {
    case AudioStreamKind::microphone: return "microphone";
    case AudioStreamKind::loopback: return "loopback";
    case AudioStreamKind::processed: return "processed";
  }
  return "unknown";
}

[[nodiscard]] inline const char* to_string(WakePosition position) {
  switch (position) {
    case WakePosition::prefix: return "prefix";
    case WakePosition::suffix: return "suffix";
    case WakePosition::embedded: return "embedded";
  }
  return "unknown";
}

}  // namespace dvo
