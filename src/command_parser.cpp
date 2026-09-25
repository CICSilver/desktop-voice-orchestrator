#include "dvo/command_parser.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <span>
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

void strip_polite_prefix(std::string& value) {
  static constexpr std::array<std::string_view, 4> prefixes{
      "麻烦帮我", "请帮我", "帮我", "请"};
  for (const auto prefix : prefixes) {
    if (!value.starts_with(prefix)) continue;
    value.erase(0, prefix.size());
    if (!value.empty() && value.front() == ';') value.erase(0, 1);
    return;
  }
}

[[nodiscard]] bool contains_rule(const std::vector<PhraseRule>& rules,
                                 std::string_view phrase,
                                 ActionType type) {
  return std::ranges::any_of(rules, [&](const PhraseRule& rule) {
    return rule.type == type && rule.phrase == phrase;
  });
}

void correct_wake_confirmed_asr_confusions(
    std::string& value, const std::vector<PhraseRule>& rules,
    bool wake_confirmed_in_text) {
  if (!wake_confirmed_in_text) return;

  // These substitutions are deliberately sentence-wide and wake-confirmed:
  // they cannot turn arbitrary follow-up speech into a command, and they are
  // enabled only when the canonical pause phrase remains configured.
  if (contains_rule(rules, "暂停音乐", ActionType::media_pause) &&
      (value == "再连音乐" || value == "再听音乐")) {
    value = "暂停音乐";
  }
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

  std::optional<std::size_t> percent_number_start;
  if (starts_with_at(text, result.end, "百分之")) {
    percent_number_start = result.end + std::string_view{"百分之"}.size();
  } else if (starts_with_at(text, result.end, "百分")) {
    // Exact-final ASR sometimes drops the unstressed "之" ("百分十").
    const auto start = result.end + std::string_view{"百分"}.size();
    if (const auto decoded = decode_one(text, start);
        decoded && is_numeric_codepoint(decoded->first)) {
      percent_number_start = start;
    }
  }
  if (percent_number_start) {
    explicit_amount = true;
    const auto number_start = *percent_number_start;
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

// ---- tolerant readings -----------------------------------------------------
// Every reading below is only a candidate: it must still parse completely under
// the strict grammar, so none of them can turn arbitrary speech into a command.

[[nodiscard]] std::vector<char32_t> codepoints(std::string_view text) {
  std::vector<char32_t> result;
  for (std::size_t offset = 0; offset < text.size();) {
    const auto decoded = decode_one(text, offset);
    if (!decoded) break;  // normalize() has already rejected invalid UTF-8
    result.push_back(decoded->first);
    offset += decoded->second;
  }
  return result;
}

[[nodiscard]] std::string to_utf8(std::span<const char32_t> text) {
  std::string result;
  for (const auto cp : text) append_utf8(result, cp);
  return result;
}

// Hesitations carry no command meaning, and recognizers sometimes emit stray
// markup symbols (for example a leading '<').
[[nodiscard]] bool is_filler(char32_t cp) {
  switch (cp) {
    case U'啊': case U'呃': case U'嗯': case U'哦': case U'唉': case U'诶': case U'呀':
    case U'<': case U'>': case U'|': case U'*': case U'#': case U'~': case U'"':
    case U'\'': case U'(': case U')': case U'[': case U']': case U'{': case U'}':
    case U'-': case U'_': case U'=': case U'@': case U'/': case U'\\': case U'`':
    case U'^':
      return true;
    default:
      return false;
  }
}

// Drops fillers together with separators directly next to them ("增加音量，啊，
// 小助手"). Other punctuation is left alone so that genuinely empty clauses
// such as "播放音乐，，暂停音乐" stay rejectable.
[[nodiscard]] std::string remove_fillers(std::string_view text) {
  const auto cps = codepoints(text);
  std::vector<bool> removed(cps.size());
  for (std::size_t index = 0; index < cps.size(); ++index) removed[index] = is_filler(cps[index]);
  std::vector<char32_t> kept;
  for (std::size_t index = 0; index < cps.size(); ++index) {
    if (removed[index]) continue;
    const bool next_to_filler = (index > 0 && removed[index - 1]) ||
                                (index + 1 < cps.size() && removed[index + 1]);
    if (cps[index] == U';' && next_to_filler) continue;
    kept.push_back(cps[index]);
  }
  return to_utf8(kept);
}

// The separator between a wake word and the command belongs to the wake word.
[[nodiscard]] std::span<const char32_t> trim_separators(std::span<const char32_t> text) {
  while (!text.empty() && text.front() == U';') text = text.subspan(1);
  while (!text.empty() && text.back() == U';') text = text.first(text.size() - 1);
  return text;
}

// "暂停暂停音乐" -> "暂停音乐": a two-character word repeated back to back is
// a stutter, not two commands.
[[nodiscard]] std::string collapse_repeats(std::string_view text) {
  auto cps = codepoints(text);
  std::vector<char32_t> result;
  for (std::size_t index = 0; index < cps.size();) {
    if (index + 3 < cps.size() && cps[index] != U';' && cps[index + 1] != U';' &&
        cps[index] == cps[index + 2] && cps[index + 1] == cps[index + 3]) {
      index += 2;
      continue;
    }
    result.push_back(cps[index]);
    ++index;
  }
  return to_utf8(result);
}

[[nodiscard]] std::size_t common_subsequence(std::span<const char32_t> left,
                                             std::span<const char32_t> right) {
  std::vector<std::size_t> previous(right.size() + 1), current(right.size() + 1);
  for (const auto cp : left) {
    for (std::size_t j = 1; j <= right.size(); ++j) {
      current[j] = cp == right[j - 1] ? previous[j - 1] + 1
                                      : std::max(previous[j], current[j - 1]);
    }
    std::swap(previous, current);
  }
  return previous.back();
}

// Locates the confirmed wake word in recognized text: an exact occurrence
// anywhere, otherwise a near miss ("小叔手", "小助", "助手") at the very start
// or end of the utterance, where a prefix or suffix wake word sits. A near miss
// must share all but one character with the wake word in order.
//
// For a two-character wake word that leaves a single shared character, which
// many ordinary words satisfy ("小可爱", "小心"). Such a near miss must be
// exactly two characters long, and only a caller whose wake word is already
// confirmed acoustically (the parser, after KWS or the probe fired) may ask
// for one; the probe itself needs the exact wake word.
[[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> locate_wake(
    std::span<const char32_t> text, std::span<const char32_t> wake,
    bool short_near_miss) {
  if (wake.empty() || text.size() < wake.size() - 1) return std::nullopt;
  if (const auto found = std::ranges::search(text, wake); !found.empty()) {
    const auto begin = static_cast<std::size_t>(found.begin() - text.begin());
    return std::pair{begin, begin + wake.size()};
  }
  const bool short_wake = wake.size() < 3;
  if (wake.size() < 2 || (short_wake && !short_near_miss)) return std::nullopt;
  const auto min_length = short_wake ? wake.size() : wake.size() - 1;
  const auto max_length = short_wake ? wake.size() : wake.size() + 1;
  std::optional<std::pair<std::size_t, std::size_t>> best;
  std::size_t best_score{};
  for (std::size_t length = min_length; length <= max_length; ++length) {
    if (length > text.size()) break;
    for (const bool at_start : {true, false}) {
      const auto begin = at_start ? 0 : text.size() - length;
      const auto window = text.subspan(begin, length);
      if (std::ranges::find(window, U';') != window.end()) continue;
      const auto score = common_subsequence(window, wake);
      if (score + 1 < wake.size() || score <= best_score) continue;
      best_score = score;
      best = std::pair{begin, begin + length};
    }
  }
  return best;
}

[[nodiscard]] std::size_t word_length(std::span<const char32_t> text) {
  return static_cast<std::size_t>(
      std::ranges::count_if(text, [](char32_t cp) { return cp != U';'; }));
}

// Text on the far side of the wake word may be dropped only when it is short
// residue (recognizer noise or background speech) that does not itself parse
// as a command, so a real command on that side is never discarded silently.
constexpr std::size_t kMaxDroppedResidue = 3;

struct ActionDraft {
  ActionType type;
  std::optional<int> delta;
  std::string canonical;
};

struct DraftParse {
  std::vector<ActionDraft> drafts;
  std::optional<CommandParseError> error;
};

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

  std::string wake_value;
  if (!context.wake_word.empty()) {
    if (auto wake = normalize(context.wake_word); !wake.error) {
      wake_value = std::move(wake.value);
      while (!wake_value.empty() && wake_value.front() == '@') {
        wake_value.erase(wake_value.begin());
      }
    }
  }

  // The strict reading. KWS can occasionally miss a repeated wake word while
  // an activation is already armed, causing the utterance to arrive through
  // the follow-up path. Strip the activation's confirmed wake word in either
  // origin so an otherwise exact command is not rejected solely because of
  // that routing.
  std::string strict = normalized.value;
  bool wake_confirmed_in_text{};
  if (!wake_value.empty()) {
    const auto position = strict.find(wake_value);
    if (position != std::string::npos) {
      wake_confirmed_in_text = true;
      auto erase_start = position;
      auto erase_size = wake_value.size();
      if (erase_start > 0 && strict[erase_start - 1] == ';') {
        --erase_start;
        ++erase_size;
      } else if (erase_start + erase_size < strict.size() &&
                 strict[erase_start + erase_size] == ';') {
        ++erase_size;
      } else if (position > 0) {
        // A weak utterance tail can make exact-final ASR append one
        // spurious Chinese character after a suffix wake word (for
        // example "小助手两"). Remove only one complete UTF-8 codepoint,
        // only when it is the entire residual tail.
        const auto tail = position + wake_value.size();
        if (const auto decoded = decode_one(strict, tail);
            decoded && tail + decoded->second == strict.size()) {
          erase_size += decoded->second;
        }
      }
      strict.erase(erase_start, erase_size);
    }
  }
  // Polite lead-ins do not change command semantics.
  strip_polite_prefix(strict);
  correct_wake_confirmed_asr_confusions(strict, impl_->rules, wake_confirmed_in_text);

  const auto parse_drafts = [this](std::string_view value) -> DraftParse {
    const auto fail = [](std::string code, std::string message, std::size_t offset) {
      return DraftParse{{}, CommandParseError{std::move(code), std::move(message), offset}};
    };
    if (value.empty()) {
      return fail("empty_text", "ASR text does not contain a command", 0);
    }
    if (value.front() == ';') {
      return fail("empty_clause", "command text must not start with a separator", 0);
    }
    std::size_t position{};

    // If ASR loses the first action of a multi-command utterance, do not reject
    // a later action merely because its connector is now at the beginning. A
    // chain also tolerates boundary duplications such as "然后后再".
    while (const auto connector_end = consume_connector(value, position,
                                                          impl_->connectors)) {
      position = *connector_end;
      if (position == value.size()) {
        return fail("trailing_connector", "a connector must be followed by a command",
                       position);
      }
    }

    std::vector<ActionDraft> drafts;

    while (position < value.size()) {
      const auto parsed = parse_action(value, position,
                                       default_volume_delta_percent_,
                                       maximum_volume_delta_percent_, impl_->rules,
                                       !drafts.empty());
      if (parsed.error) return DraftParse{{}, parsed.error};
      drafts.push_back({parsed.type, parsed.delta, parsed.canonical});
      if (drafts.size() > max_actions_per_utterance_) {
        return fail("too_many_actions", "command plan exceeds the configured action limit", position);
      }
      position = parsed.end;
      if (position == value.size()) break;

      bool had_punctuation = false;
      if (value[position] == ';') {
        had_punctuation = true;
        ++position;
        if (position < value.size() && value[position] == ';') {
          return fail("empty_clause", "consecutive separators create an empty command clause",
                         position);
        }
        if (position == value.size()) break;  // one terminal punctuation is harmless
      }

      bool had_connector = false;
      while (const auto connector_end = consume_connector(value, position,
                                                            impl_->connectors)) {
        had_connector = true;
        position = *connector_end;
        if (position < value.size() && value[position] == ';') {
          ++position;
          if (position < value.size() && value[position] == ';') {
            return fail("empty_clause", "consecutive separators create an empty command clause",
                           position);
          }
        }
        if (position == value.size()) {
          return fail("trailing_connector", "a connector must be followed by another command",
                         position);
        }
      }

      // With neither punctuation nor a connector, adjacency is intentional and
      // parse_action() below must consume the next command in full.
      (void)had_punctuation;
      (void)had_connector;
    }
    return DraftParse{std::move(drafts), {}};
  };

  // Tolerant readings, tried in order only when the strict reading does not
  // parse: hesitations and stray symbols removed, stutters collapsed, and the
  // command taken from one side of the wake word when the other side is short
  // residue that is not itself a command. Each must still parse completely.
  std::vector<std::string> readings{strict};
  const auto add_reading = [&](std::string value, bool wake_confirmed) {
    value = remove_fillers(value);
    strip_polite_prefix(value);
    correct_wake_confirmed_asr_confusions(value, impl_->rules, wake_confirmed);
    if (value.empty() || std::ranges::find(readings, value) != readings.end()) return;
    readings.push_back(std::move(value));
  };
  add_reading(strict, wake_confirmed_in_text);
  add_reading(collapse_repeats(remove_fillers(strict)), wake_confirmed_in_text);
  if (!wake_value.empty()) {
    const auto cleaned = codepoints(remove_fillers(normalized.value));
    const auto wake = codepoints(wake_value);
    if (const auto span = locate_wake(cleaned, wake, true)) {
      const auto all = std::span<const char32_t>(cleaned);
      const auto before = trim_separators(all.first(span->first));
      const auto after = trim_separators(all.subspan(span->second));
      for (const auto& [kept, dropped] :
           {std::pair{before, after}, std::pair{after, before}}) {
        const auto dropped_length = word_length(dropped);
        if (dropped_length > kMaxDroppedResidue) continue;
        if (dropped_length > 0 && !parse_drafts(remove_fillers(to_utf8(dropped))).error) {
          continue;
        }
        add_reading(to_utf8(kept), true);
        add_reading(collapse_repeats(to_utf8(kept)), true);
      }
    }
  }

  std::optional<DraftParse> accepted;
  std::optional<CommandParseError> first_error;
  for (const auto& reading : readings) {
    auto parsed = parse_drafts(reading);
    if (!parsed.error) {
      accepted = std::move(parsed);
      break;
    }
    if (!first_error) first_error = std::move(parsed.error);
  }
  if (!accepted) return {{}, first_error};
  const auto& drafts = accepted->drafts;

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

std::optional<WakeTextMatch> find_wake_in_text(std::string_view text,
                                               std::string_view wake_word) {
  const auto normalized_text = normalize(text);
  auto normalized_wake = normalize(wake_word);
  if (normalized_text.error || normalized_wake.error) return std::nullopt;
  while (!normalized_wake.value.empty() && normalized_wake.value.front() == '@') {
    normalized_wake.value.erase(normalized_wake.value.begin());
  }
  const auto words_only = [](std::string_view value) {
    auto cps = codepoints(remove_fillers(value));
    std::erase(cps, U';');
    return cps;
  };
  const auto cps = words_only(normalized_text.value);
  const auto wake = words_only(normalized_wake.value);
  const auto span = locate_wake(cps, wake, false);
  if (!span) return std::nullopt;
  const bool exact = span->second - span->first == wake.size() &&
                     std::equal(wake.begin(), wake.end(), cps.begin() + span->first);
  return WakeTextMatch{span->first, span->second, cps.size(), exact};
}

}  // namespace dvo
