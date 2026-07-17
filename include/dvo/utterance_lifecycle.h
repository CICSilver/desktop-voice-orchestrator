#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "dvo/audio_types.h"

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
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
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
