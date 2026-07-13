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
  std::uint64_t qpc_100ns{};
  std::uint64_t device_position{};
  bool silent{};
  bool discontinuity{};
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
