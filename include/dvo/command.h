#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dvo {

inline constexpr std::size_t kMaxCommandActions = 8;
inline constexpr const char* kCommandParserVersion = "rules.zh.v1";

enum class ActionType {
  media_play,
  media_pause,
  master_volume_adjust,
};

enum class ExecutionSource {
  live,
  replay,
  benchmark,
};

enum class ExecutionMode {
  live,
  dry_run,
};

enum class ActionStatus {
  started,
  succeeded,
  noop,
  failed,
  indeterminate,
  deduplicated,
  dry_run,
  cancelled,
};

struct PlannedAction {
  std::string action_id;
  std::uint32_t sequence{};
  ActionType type{ActionType::media_play};
  // Only master_volume_adjust uses this field. It represents relative
  // Windows volume-slider percentage points, not an acoustic amplitude ratio.
  std::optional<int> volume_delta_percent;
};

struct CommandPlan {
  std::uint32_t schema_version{1};
  std::string command_id;
  std::string runtime_session_id;
  std::string utterance_id;
  std::string wake_position;
  std::string raw_text;
  std::string normalized_text;
  std::string parser_version{kCommandParserVersion};
  // Revision of the configuration snapshot captured when the utterance began.
  // A later hot update must not change the grammar used for this plan.
  std::uint64_t config_revision{};
  std::uint64_t recognition_generation{};
  std::uint64_t final_revision{};
  std::uint64_t timestamp_sample{};
  ExecutionSource source{ExecutionSource::live};
  ExecutionMode execution_mode{ExecutionMode::live};
  std::vector<PlannedAction> actions;
};

struct ActionResult {
  std::uint32_t schema_version{1};
  std::string command_id;
  std::string utterance_id;
  std::string action_id;
  std::uint32_t sequence{};
  ActionType type{ActionType::media_play};
  ExecutionSource source{ExecutionSource::live};
  std::uint64_t timestamp_sample{};
  ActionStatus status{ActionStatus::failed};
  std::string adapter;
  std::string target_id;
  std::string error_code;
  std::string message;
  std::optional<int> requested_volume_delta_percent;
  std::optional<double> volume_before;
  std::optional<double> volume_requested;
  std::optional<double> volume_after;
  std::optional<bool> mute_before;
  std::optional<bool> mute_after;
  bool clamped{};
  bool verified{};
  std::chrono::milliseconds duration{};
};

[[nodiscard]] const char* to_string(ActionType type);
[[nodiscard]] const char* to_string(ExecutionSource source);
[[nodiscard]] const char* to_string(ExecutionMode mode);
[[nodiscard]] const char* to_string(ActionStatus status);

}  // namespace dvo
