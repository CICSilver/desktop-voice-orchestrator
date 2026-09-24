#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/config.h"

namespace dvo {

// Labels written next to a recorded session by the guided collection page
// (web/collect.html). Take times are seconds from the start of the session's
// audio files, which is also where replay starts its sample clock.
struct EvaluationExpectedAction {
  std::string type;
  std::optional<int> volume_delta_percent;
};

struct EvaluationTake {
  std::string id;
  // "command": wake word plus command, must execute exactly expected_actions.
  // "negative": speech without the wake word, must not wake or execute.
  // "silence": no speech (ambient or music only), must not wake.
  std::string kind;
  // "ok" takes are scored; "discarded" (re-recorded) and "skipped" are not.
  std::string status;
  std::string prompt;
  std::string text;
  std::string wake_position;  // prefix | suffix | none
  std::vector<EvaluationExpectedAction> expected_actions;
  double start_s{};
  double end_s{};
  int attempt{1};
};

struct EvaluationLabels {
  std::string collection_id;
  std::string group_id;
  std::string group_title;
  std::string condition;  // quiet | music
  double distance_m{};
  std::string speaker;
  std::string wake_word;
  bool complete{};
  std::vector<EvaluationTake> takes;
};

// Throws std::invalid_argument when the document is not a supported labels
// file or a take is malformed.
[[nodiscard]] EvaluationLabels parse_evaluation_labels(const nlohmann::json& value);
[[nodiscard]] EvaluationLabels load_evaluation_labels(const std::filesystem::path& path);

// Character coverage of expected text by recognized text, ignoring
// punctuation and whitespace. head_missing/tail_missing count expected
// characters before the first and after the last aligned character; they are
// the signature of a candidate that was cut at its start or end.
struct TextCoverage {
  std::size_t expected_chars{};
  std::size_t matched_chars{};
  std::size_t head_missing{};
  std::size_t tail_missing{};
  std::string missing_head;
  std::string missing_tail;
};
[[nodiscard]] TextCoverage text_coverage(std::string_view expected,
                                         std::string_view recognized);

// Pipeline-independent reference results for one take window, decoded with
// fresh KWS/ASR streams straight from the session audio.
struct TakeReference {
  bool available{};
  bool kws_microphone{};
  bool kws_processed{};
  std::string asr_microphone;
  std::string asr_processed;
  std::optional<double> speech_dbfs;    // loud end of the microphone level
  std::optional<double> loopback_dbfs;  // mean render level (music check)
};

// Re-decodes of one pipeline candidate from the session audio. A character
// that appears only when the span is extended was cut by segmentation; one
// that stays missing was dropped by the recognizer. Edge levels are the
// loudest audio just outside the span relative to speech inside it.
struct CandidateReference {
  std::optional<double> head_db;
  std::optional<double> tail_db;
  std::optional<std::string> text;
  std::optional<std::string> text_head_extended;
  std::optional<std::string> text_tail_extended;
};

struct SessionReference {
  std::vector<TakeReference> takes;  // parallel to EvaluationLabels::takes
  std::map<std::string, CandidateReference> candidates;  // by utterance_id
  bool processed_audio{};
};

// Decodes reference results for every scored take and measures the edge
// levels of every replayed candidate listed in utterance_summary (the output
// of summarize_replayed_utterances).
[[nodiscard]] SessionReference compute_session_reference(
    const std::filesystem::path& session, const EvaluationLabels& labels,
    const AppConfig& config, const nlohmann::json& utterance_summary);

// Scores a replayed session against its labels. Pure: all audio analysis is
// supplied through reference.
[[nodiscard]] nlohmann::json evaluate_labeled_session(
    const EvaluationLabels& labels, const nlohmann::json& utterance_summary,
    const SessionReference& reference);

// Combines per-session reports into totals by condition, distance and wake
// position.
[[nodiscard]] nlohmann::json aggregate_evaluations(
    const std::vector<nlohmann::json>& session_reports);

// Human-readable Chinese summary for the console.
[[nodiscard]] std::string format_evaluation_summary(
    const std::vector<nlohmann::json>& session_reports,
    const nlohmann::json& aggregate);

}  // namespace dvo
