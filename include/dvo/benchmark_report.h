#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dvo {

// Reads the append-only event stream used by diagnostic sessions. Malformed
// lines are skipped and reported so that one damaged record does not make an
// otherwise useful benchmark unreadable.
[[nodiscard]] std::vector<nlohmann::json> read_benchmark_events(
    const std::filesystem::path& path, std::vector<std::string>* errors = nullptr);

// Compares the newly replayed, derived pipeline events against those stored in
// the source session and aggregates ASR timing plus the protected dry-run
// action sequence.
[[nodiscard]] nlohmann::json build_benchmark_comparison(
    const std::vector<nlohmann::json>& recorded,
    const std::vector<nlohmann::json>& replayed,
    std::uint32_t frame_samples);

// Folds a replayed event stream into one record per keyword hit and one per
// utterance (candidate span, ASR text, plan or rejection). Times are seconds
// from origin_sample, i.e. offsets into the session's audio files when
// origin_sample is the first replayed frame.
[[nodiscard]] nlohmann::json summarize_replayed_utterances(
    const std::vector<nlohmann::json>& events, std::uint64_t origin_sample);

}  // namespace dvo
