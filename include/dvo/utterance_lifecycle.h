#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dvo {

enum class UtteranceLifecyclePhase {
  begin,
  chunk,
  finalize,
  cancel,
};

struct UtteranceLifecycleEvent {
  UtteranceLifecyclePhase phase{UtteranceLifecyclePhase::begin};
  std::string utterance_id;
  std::uint64_t generation{};
  std::uint64_t first_sample{};
  std::size_t sample_count{};
  std::string source{"live"};
  std::string detail;
};

[[nodiscard]] inline const char* to_string(UtteranceLifecyclePhase phase) {
  switch (phase) {
    case UtteranceLifecyclePhase::begin: return "begin";
    case UtteranceLifecyclePhase::chunk: return "chunk";
    case UtteranceLifecyclePhase::finalize: return "finalize";
    case UtteranceLifecyclePhase::cancel: return "cancel";
  }
  return "unknown";
}

}  // namespace dvo
