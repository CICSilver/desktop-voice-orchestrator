#include "dvo/command_parser.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dvo {
namespace {

struct NormalizedText {
  std::string value;
  std::optional<CommandParseError> error;
};

struct ParsedNumber {
  std::optional<int> value;
  std::size_t end{};
};

struct ParsedAction {
  ActionType type{ActionType::media_play};
  std::optional<int> delta;
  std::string canonical;
  std::size_t end{};
  std::optional<CommandParseError> error;
};

struct PhraseRule {
  std::string phrase;
  std::vector<std::string> omission_variants;
  ActionType type{ActionType::media_play};
  bool volume_increase{};
};

[[nodiscard]] bool is_space(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' || cp == U'\f' ||
         cp == U'\v' || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

[[nodiscard]] bool is_separator(char32_t cp) {
  switch (cp) {
    case U',':
    case U'.':
    case U'!':
    case U'?':
    case U'+':
    case U';':
    case U':':
    case U'，':
    case U'。':
    case U'！':
    case U'？':
    case U'、':
    case U'；':
    case U'：': return true;
    default: return false;
  }
}

void append_utf8(std::string& output, char32_t cp) {
  if (cp <= 0x7F) {
    output.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    output.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

[[nodiscard]] std::optional<std::pair<char32_t, std::size_t>> decode_one(
    std::string_view text, std::size_t offset) {
  if (offset >= text.size()) return std::nullopt;
  const auto first = static_cast<unsigned char>(text[offset]);
  if (first <= 0x7F) return std::pair{static_cast<char32_t>(first), std::size_t{1}};

  std::size_t length{};
  char32_t value{};
  char32_t minimum{};
  if ((first & 0xE0) == 0xC0) {
    length = 2;
    value = first & 0x1F;
    minimum = 0x80;
  } else if ((first & 0xF0) == 0xE0) {
    length = 3;
    value = first & 0x0F;
    minimum = 0x800;
  } else if ((first & 0xF8) == 0xF0) {
    length = 4;
    value = first & 0x07;
    minimum = 0x10000;
  } else {
    return std::nullopt;
  }
  if (offset + length > text.size()) return std::nullopt;
  for (std::size_t i = 1; i < length; ++i) {
    const auto next = static_cast<unsigned char>(text[offset + i]);
    if ((next & 0xC0) != 0x80) return std::nullopt;
    value = (value << 6) | (next & 0x3F);
  }
  if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
    return std::nullopt;
  }
  return std::pair{value, length};
}

[[nodiscard]] NormalizedText normalize(std::string_view input) {
  NormalizedText result;
  for (std::size_t offset = 0; offset < input.size();) {
    const auto decoded = decode_one(input, offset);
    if (!decoded) {
      result.error = CommandParseError{"invalid_utf8", "ASR text is not valid UTF-8", offset};
      return result;
    }
    auto [cp, length] = *decoded;
    if (is_space(cp)) {
      offset += length;
      continue;
    }
    if ((cp < 0x20 || cp == 0x7F) && !is_space(cp)) {
      result.error = CommandParseError{"invalid_character", "ASR text contains a control character",
                                       offset};
      return result;
    }
    if (cp >= 0xFF10 && cp <= 0xFF19) cp = U'0' + (cp - 0xFF10);
    if (cp == U'％') cp = U'%';
    if (is_separator(cp)) cp = U';';
    append_utf8(result.value, cp);
    offset += length;
  }
  return result;
}

[[nodiscard]] bool starts_with_at(std::string_view value, std::size_t offset,
                                  std::string_view prefix) {
  return offset <= value.size() && value.substr(offset).starts_with(prefix);
}

[[nodiscard]] bool is_numeric_codepoint(char32_t cp) {
  return (cp >= U'0' && cp <= U'9') || cp == U'零' || cp == U'一' || cp == U'二' ||
         cp == U'两' || cp == U'三' || cp == U'四' || cp == U'五' || cp == U'六' ||
         cp == U'七' || cp == U'八' || cp == U'九' || cp == U'十' || cp == U'百';
}

[[nodiscard]] std::optional<int> parse_integer_token(std::string_view token) {
  if (token.empty()) return std::nullopt;
  bool ascii = true;
  for (const auto ch : token) {
    if (ch < '0' || ch > '9') {
      ascii = false;
      break;
    }
  }
  if (ascii) {
    if (token.size() > 1 && token.front() == '0') return std::nullopt;
    int value{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) return std::nullopt;
    return value;
  }

  static constexpr std::array<std::pair<std::string_view, int>, 22> numbers{{
      {"零", 0}, {"一", 1}, {"二", 2}, {"两", 2}, {"三", 3}, {"四", 4},
      {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}, {"十", 10},
      {"十一", 11}, {"十二", 12}, {"十三", 13}, {"十四", 14}, {"十五", 15},
      {"十六", 16}, {"十七", 17}, {"十八", 18}, {"十九", 19}, {"二十", 20},
  }};
  for (const auto& [spelling, value] : numbers) {
    if (token == spelling) return value;
  }
  return std::nullopt;
}

[[nodiscard]] ParsedNumber scan_number(std::string_view text, std::size_t offset) {
  auto end = offset;
  while (end < text.size()) {
    const auto decoded = decode_one(text, end);
    if (!decoded || !is_numeric_codepoint(decoded->first)) break;
    end += decoded->second;
  }
  if (end == offset) return {{}, offset};
  return {parse_integer_token(text.substr(offset, end - offset)), end};
}

[[nodiscard]] ParsedAction parse_volume(std::string_view text, std::size_t matched_end,
                                        std::string_view base, bool increase,
                                        int default_delta, int maximum_delta) {
  ParsedAction result;
  result.type = ActionType::master_volume_adjust;
  result.end = matched_end;
  auto amount = default_delta;
  bool explicit_amount = false;

  if (starts_with_at(text, result.end, "百分之")) {
    explicit_amount = true;
    const auto number_start = result.end + std::string_view{"百分之"}.size();
    const auto number = scan_number(text, number_start);
    if (!number.value) {
      result.error = {"invalid_volume_amount", "volume percentage is not a supported integer",
                      number_start};
      return result;
    }
    amount = *number.value;
    result.end = number.end;
  } else {
    const auto number = scan_number(text, result.end);
    if (number.end != result.end && number.end < text.size() && text[number.end] == '%') {
      explicit_amount = true;
      if (!number.value) {
        result.error = {"invalid_volume_amount", "volume percentage is not a supported integer",
                        result.end};
        return result;
      }
      amount = *number.value;
      result.end = number.end + 1;
    }
  }

  if (explicit_amount && (amount < 1 || amount > maximum_delta)) {
    result.error = {"volume_amount_out_of_range",
                    "volume percentage must be between 1 and the configured maximum",
                    result.end};
    return result;
  }
  result.delta = increase ? amount : -amount;
  result.canonical = std::string(base) + std::to_string(amount) + "%";
  return result;
}

[[nodiscard]] ParsedAction parse_action(std::string_view text, std::size_t offset,
                                        int default_delta, int maximum_delta,
                                        const std::vector<PhraseRule>& rules,
                                        bool allow_trailing_prefix) {
  const auto parsed_rule = [&](const PhraseRule& rule, std::size_t end) {
    if (rule.type == ActionType::master_volume_adjust) {
      return parse_volume(text, end, rule.phrase, rule.volume_increase,
                          default_delta, maximum_delta);
    }
    return ParsedAction{rule.type, {}, rule.phrase, end, {}};
  };

  for (const auto& rule : rules) {
    if (!starts_with_at(text, offset, rule.phrase)) continue;
    return parsed_rule(rule, offset + rule.phrase.size());
  }

  // ASR commonly drops one Chinese character at an utterance boundary. Admit
  // only an exact one-codepoint omission of a configured phrase, and only when
  // the resulting action is unambiguous. This is deliberately narrower than
  // general edit-distance matching so wake-gated speech cannot become an
  // unrelated command by guesswork.
  const PhraseRule* fuzzy_rule{};
  std::size_t fuzzy_size{};
  bool ambiguous{};
  for (const auto& rule : rules) {
    for (const auto& variant : rule.omission_variants) {
      if (!starts_with_at(text, offset, variant)) continue;
      if (variant.size() < fuzzy_size) continue;
      if (variant.size() > fuzzy_size) {
        fuzzy_rule = &rule;
        fuzzy_size = variant.size();
        ambiguous = false;
        continue;
      }
      if (fuzzy_rule &&
          (fuzzy_rule->type != rule.type ||
           fuzzy_rule->volume_increase != rule.volume_increase)) {
        ambiguous = true;
      }
    }
  }
  if (ambiguous) {
    return {ActionType::media_play, {}, {}, offset,
            CommandParseError{"ambiguous_text",
                              "one-character ASR correction matches multiple commands",
                              offset}};
  }
  if (fuzzy_rule) return parsed_rule(*fuzzy_rule, offset + fuzzy_size);

  // Exact-final decoding can regress relative to the last streaming partial
  // when the utterance tail is weak. After at least one complete action and a
  // connector, accept an end-of-utterance prefix only when every matching
  // configured phrase has the same semantic action. For example,
  // "播放音乐然后再暂" deterministically completes to "暂停音乐", while
  // arbitrary residual text remains rejected.
  if (allow_trailing_prefix && offset < text.size()) {
    const auto remainder = text.substr(offset);
    const PhraseRule* prefix_rule{};
    bool prefix_ambiguous{};
    for (const auto& rule : rules) {
      if (!rule.phrase.starts_with(remainder)) continue;
      if (!prefix_rule) {
        prefix_rule = &rule;
        continue;
      }
      if (prefix_rule->type != rule.type ||
          prefix_rule->volume_increase != rule.volume_increase) {
        prefix_ambiguous = true;
      }
    }
    if (prefix_ambiguous) {
      return {ActionType::media_play, {}, {}, offset,
              CommandParseError{"ambiguous_text",
                                "trailing command prefix matches multiple actions",
                                offset}};
    }
    if (prefix_rule) return parsed_rule(*prefix_rule, text.size());
  }
  return {ActionType::media_play, {}, {}, offset,
          CommandParseError{"unknown_text", "text contains an unknown or incomplete command", offset}};
}

[[nodiscard]] std::optional<std::size_t> consume_connector(std::string_view text,
                                                            std::size_t offset,
                                                            const std::vector<std::string>& connectors) {
  for (const auto connector : connectors) {
    if (starts_with_at(text, offset, connector)) return offset + connector.size();
  }
  return std::nullopt;
}

[[nodiscard]] CommandParseResult failure(std::string code, std::string message,
                                         std::size_t offset) {
  return {{}, CommandParseError{std::move(code), std::move(message), offset}};
}

[[nodiscard]] std::string normalized_rule(std::string_view rule, std::string_view category) {
  const auto normalized = normalize(rule);
  if (normalized.error) {
    throw std::invalid_argument(std::string(category) + " contains invalid UTF-8 or control characters");
  }
  if (normalized.value.empty()) {
    throw std::invalid_argument(std::string(category) + " must not contain an empty value");
  }
  if (normalized.value.find(';') != std::string::npos) {
    throw std::invalid_argument(std::string(category) + " values must not contain punctuation");
  }
  return normalized.value;
}

[[nodiscard]] std::vector<std::string> omission_variants(std::string_view phrase) {
  std::vector<std::size_t> boundaries{0};
  for (std::size_t offset = 0; offset < phrase.size();) {
    const auto decoded = decode_one(phrase, offset);
    if (!decoded) return {};
    offset += decoded->second;
    boundaries.push_back(offset);
  }
  if (boundaries.size() <= 2) return {};

  std::vector<std::string> variants;
  std::unordered_set<std::string> unique;
  for (std::size_t i = 0; i + 1 < boundaries.size(); ++i) {
    auto variant = std::string(phrase.substr(0, boundaries[i]));
    variant.append(phrase.substr(boundaries[i + 1]));
    if (!variant.empty() && unique.insert(variant).second) {
      variants.push_back(std::move(variant));
    }
  }
  return variants;
}

void append_rules(const std::vector<std::string>& phrases, std::string_view category,
                  ActionType type, bool volume_increase,
                  std::vector<PhraseRule>& output,
                  std::unordered_map<std::string, std::string>& owners) {
  if (phrases.empty()) {
    throw std::invalid_argument(std::string(category) + " must contain at least one phrase");
  }
  for (const auto& phrase : phrases) {
    auto normalized = normalized_rule(phrase, category);
    const auto [owner, inserted] = owners.emplace(normalized, std::string(category));
    if (!inserted) {
      throw std::invalid_argument("command phrase '" + normalized + "' is duplicated in " +
                                  owner->second + " and " + std::string(category));
    }
    auto variants = omission_variants(normalized);
    output.push_back({std::move(normalized), std::move(variants), type, volume_increase});
  }
}

}  // namespace

struct CommandParser::Impl {
  std::vector<PhraseRule> rules;
  std::vector<std::string> connectors;
};

CommandParser::CommandParser(int default_volume_delta_percent,
                             int maximum_volume_delta_percent,
                             CommandGrammar grammar)
    : default_volume_delta_percent_(default_volume_delta_percent),
      maximum_volume_delta_percent_(maximum_volume_delta_percent),
      max_actions_per_utterance_(grammar.max_actions_per_utterance) {
  if (maximum_volume_delta_percent_ < 1 || maximum_volume_delta_percent_ > 20) {
    throw std::invalid_argument("maximum volume delta must be in [1, 20]");
  }
  if (default_volume_delta_percent_ < 1 ||
      default_volume_delta_percent_ > maximum_volume_delta_percent_) {
    throw std::invalid_argument("default volume delta must be in [1, maximum]");
  }
  if (max_actions_per_utterance_ == 0 || max_actions_per_utterance_ > kMaxCommandActions) {
    throw std::invalid_argument("maximum actions per utterance must be in [1, 8]");
  }

  auto implementation = std::make_shared<Impl>();
  std::unordered_map<std::string, std::string> owners;
  append_rules(grammar.play_phrases, "commands.phrases.play", ActionType::media_play,
               false, implementation->rules, owners);
  append_rules(grammar.pause_phrases, "commands.phrases.pause", ActionType::media_pause,
               false, implementation->rules, owners);
  append_rules(grammar.volume_up_phrases, "commands.phrases.volume_up",
               ActionType::master_volume_adjust, true, implementation->rules, owners);
  append_rules(grammar.volume_down_phrases, "commands.phrases.volume_down",
               ActionType::master_volume_adjust, false, implementation->rules, owners);
  std::ranges::sort(implementation->rules, {}, [](const PhraseRule& rule) {
    return rule.phrase.size();
  });
  std::ranges::reverse(implementation->rules);

  std::unordered_set<std::string> connector_values;
  for (const auto& connector : grammar.connectors) {
    auto normalized = normalized_rule(connector, "commands.connectors");
    if (!connector_values.insert(normalized).second) {
      throw std::invalid_argument("commands.connectors contains duplicate value '" + normalized + "'");
    }
    for (const auto& rule : implementation->rules) {
      if (rule.phrase.starts_with(normalized)) {
        throw std::invalid_argument("connector '" + normalized +
                                    "' is a prefix of command phrase '" + rule.phrase + "'");
      }
    }
    implementation->connectors.push_back(std::move(normalized));
  }
  std::ranges::sort(implementation->connectors, {}, &std::string::size);
  std::ranges::reverse(implementation->connectors);
  impl_ = std::move(implementation);
}

CommandParseResult CommandParser::parse(std::string_view text,
                                        const CommandParseContext& context) const {
  if (context.utterance_id.empty()) {
    return failure("invalid_context", "utterance_id is required", 0);
  }
  if (context.runtime_session_id.empty()) {
    return failure("invalid_context", "runtime_session_id is required", 0);
  }
  auto normalized = normalize(text);
  if (normalized.error) return {{}, normalized.error};
  // KWS can occasionally miss a repeated wake word while an activation is
  // already armed, causing the utterance to arrive through the follow-up
  // path. Strip the activation's confirmed wake word in either origin so an
  // otherwise exact command is not rejected solely because of that routing.
  if (!context.wake_word.empty()) {
    auto wake = normalize(context.wake_word);
    if (!wake.error) {
      while (!wake.value.empty() && wake.value.front() == '@') {
        wake.value.erase(wake.value.begin());
      }
      if (!wake.value.empty()) {
        const auto position = normalized.value.find(wake.value);
        if (position != std::string::npos) {
          auto erase_start = position;
          auto erase_size = wake.value.size();
          if (erase_start > 0 && normalized.value[erase_start - 1] == ';') {
            --erase_start;
            ++erase_size;
          } else if (erase_start + erase_size < normalized.value.size() &&
                     normalized.value[erase_start + erase_size] == ';') {
            ++erase_size;
          }
          normalized.value.erase(erase_start, erase_size);
        }
      }
    }
  }
  if (normalized.value.empty()) {
    return failure("empty_text", "ASR text does not contain a command", 0);
  }
  if (normalized.value.front() == ';') {
    return failure("empty_clause", "command text must not start with a separator", 0);
  }
  std::size_t position{};

  // If ASR loses the first action of a multi-command utterance, do not reject
  // a later action merely because its connector is now at the beginning. A
  // chain also tolerates boundary duplications such as "然后后再".
  while (const auto connector_end = consume_connector(normalized.value, position,
                                                        impl_->connectors)) {
    position = *connector_end;
    if (position == normalized.value.size()) {
      return failure("trailing_connector", "a connector must be followed by a command",
                     position);
    }
  }

  struct ActionDraft {
    ActionType type;
    std::optional<int> delta;
    std::string canonical;
  };
  std::vector<ActionDraft> drafts;

  while (position < normalized.value.size()) {
    const auto parsed = parse_action(normalized.value, position,
                                     default_volume_delta_percent_,
                                     maximum_volume_delta_percent_, impl_->rules,
                                     !drafts.empty());
    if (parsed.error) return {{}, parsed.error};
    drafts.push_back({parsed.type, parsed.delta, parsed.canonical});
    if (drafts.size() > max_actions_per_utterance_) {
      return failure("too_many_actions", "command plan exceeds the configured action limit", position);
    }
    position = parsed.end;
    if (position == normalized.value.size()) break;

    bool had_punctuation = false;
    if (normalized.value[position] == ';') {
      had_punctuation = true;
      ++position;
      if (position < normalized.value.size() && normalized.value[position] == ';') {
        return failure("empty_clause", "consecutive separators create an empty command clause",
                       position);
      }
      if (position == normalized.value.size()) break;  // one terminal punctuation is harmless
    }

    bool had_connector = false;
    while (const auto connector_end = consume_connector(normalized.value, position,
                                                          impl_->connectors)) {
      had_connector = true;
      position = *connector_end;
      if (position < normalized.value.size() && normalized.value[position] == ';') {
        ++position;
        if (position < normalized.value.size() && normalized.value[position] == ';') {
          return failure("empty_clause", "consecutive separators create an empty command clause",
                         position);
        }
      }
      if (position == normalized.value.size()) {
        return failure("trailing_connector", "a connector must be followed by another command",
                       position);
      }
    }

    // With neither punctuation nor a connector, adjacency is intentional and
    // parse_action() below must consume the next command in full.
    (void)had_punctuation;
    (void)had_connector;
  }

  CommandPlan plan;
  plan.command_id = "cmd:" + std::to_string(context.runtime_session_id.size()) + ":" +
                    context.runtime_session_id + ":" +
                    std::to_string(context.utterance_id.size()) + ":" +
                    context.utterance_id + ":g" +
                    std::to_string(context.recognition_generation) + ":r" +
                    std::to_string(context.final_revision) + ":c" +
                    std::to_string(context.config_revision) + ":" + kCommandParserVersion;
  plan.runtime_session_id = context.runtime_session_id;
  plan.utterance_id = context.utterance_id;
  plan.origin = context.origin;
  plan.activation_id = context.activation_id;
  plan.turn_index = context.turn_index;
  plan.trigger_sample = context.trigger_sample;
  plan.wake_position = context.wake_position;
  plan.raw_text = std::string(text);
  plan.parser_version = kCommandParserVersion;
  plan.config_revision = context.config_revision;
  plan.recognition_generation = context.recognition_generation;
  plan.final_revision = context.final_revision;
  plan.timestamp_sample = context.timestamp_sample;
  plan.source = context.source;
  plan.execution_mode = context.source == ExecutionSource::live
                            ? context.execution_mode
                            : ExecutionMode::dry_run;

  for (std::size_t i = 0; i < drafts.size(); ++i) {
    if (!plan.normalized_text.empty()) plan.normalized_text.push_back(';');
    plan.normalized_text += drafts[i].canonical;
    PlannedAction action;
    action.sequence = static_cast<std::uint32_t>(i + 1);
    action.action_id = plan.command_id + ":" + std::to_string(action.sequence);
    action.type = drafts[i].type;
    action.volume_delta_percent = drafts[i].delta;
    plan.actions.push_back(std::move(action));
  }
  return {std::move(plan), {}};
}

}  // namespace dvo
