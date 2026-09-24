#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "dvo/command.h"

namespace dvo {

struct CommandParseContext {
  std::string runtime_session_id;
  std::string utterance_id;
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
  std::string wake_position;
  std::string wake_word;
  std::uint64_t config_revision{};
  std::uint64_t recognition_generation{};
  std::uint64_t final_revision{};
  std::uint64_t timestamp_sample{};
  ExecutionSource source{ExecutionSource::live};
  ExecutionMode execution_mode{ExecutionMode::live};
};

struct CommandParseError {
  std::string code;
  std::string message;
  std::size_t byte_offset{};
};

struct CommandParseResult {
  std::optional<CommandPlan> plan;
  std::optional<CommandParseError> error;

  [[nodiscard]] bool ok() const { return plan.has_value(); }
};

struct CommandGrammar {
  std::vector<std::string> play_phrases{"播放音乐", "打开音乐"};
  std::vector<std::string> pause_phrases{"暂停音乐"};
  std::vector<std::string> volume_up_phrases{"增加音量"};
  std::vector<std::string> volume_down_phrases{"降低音量"};
  std::vector<std::string> connectors{"然后再", "然后", "再", "后", "接着", "并且", "以及", "和", "还有"};
  std::size_t max_actions_per_utterance{kMaxCommandActions};
};

class CommandParser {
 public:
  explicit CommandParser(int default_volume_delta_percent = 5,
                         int maximum_volume_delta_percent = 20,
                         CommandGrammar grammar = {});

  [[nodiscard]] CommandParseResult parse(
      std::string_view text, const CommandParseContext& context) const;

  [[nodiscard]] int default_volume_delta_percent() const {
    return default_volume_delta_percent_;
  }
  [[nodiscard]] int maximum_volume_delta_percent() const {
    return maximum_volume_delta_percent_;
  }
  [[nodiscard]] std::size_t max_actions_per_utterance() const {
    return max_actions_per_utterance_;
  }

 private:
  struct Impl;
  int default_volume_delta_percent_;
  int maximum_volume_delta_percent_;
  std::size_t max_actions_per_utterance_;
  std::shared_ptr<const Impl> impl_;
};

// Where a wake word sits in recognized text, counted in word characters
// (punctuation, whitespace and fillers ignored). The match rule is the one the
// parser uses: an exact occurrence anywhere, otherwise a near miss sharing all
// but one character, and only at the very start or end of the text.
struct WakeTextMatch {
  std::size_t begin{};
  std::size_t end{};
  std::size_t length{};
  bool exact{};
};
[[nodiscard]] std::optional<WakeTextMatch> find_wake_in_text(std::string_view text,
                                                            std::string_view wake_word);

}  // namespace dvo
