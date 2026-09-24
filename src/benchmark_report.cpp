#include "dvo/benchmark_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string_view>

namespace dvo {
namespace {

constexpr std::array<std::string_view, 6> kComparableEventTypes{
    "kws_hit", "candidate", "candidate_rejected", "asr_final",
    "command_plan", "command_rejected"};

bool comparable(std::string_view type) {
  return std::ranges::find(kComparableEventTypes, type) !=
         kComparableEventTypes.end();
}

// Event payloads use null for absent optional strings (for example the wake
// position of a follow-up candidate); json::value() would throw on those.
std::string string_value(const nlohmann::json& value, std::string_view key,
                         std::string_view fallback = {}) {
  if (!value.is_object()) return std::string(fallback);
  const auto found = value.find(key);
  return found != value.end() && found->is_string() ? found->get<std::string>()
                                                    : std::string(fallback);
}

std::string type_of(const nlohmann::json& event) {
  return string_value(event, "type");
}

const nlohmann::json& payload_of(const nlohmann::json& event) {
  static const nlohmann::json empty = nlohmann::json::object();
  if (!event.is_object()) return empty;
  const auto found = event.find("payload");
  return found != event.end() && found->is_object() ? *found : empty;
}

std::vector<std::string> comparable_sequence(
    const std::vector<nlohmann::json>& events) {
  std::vector<std::string> result;
  std::set<std::string> seen_candidates;
  for (const auto& event : events) {
    auto type = type_of(event);
    if (!comparable(type)) continue;
    if (type == "candidate") {
      const auto& payload = payload_of(event);
      const auto key = string_value(payload, "utterance_id") + ":" +
                       payload.value("wake_span", nlohmann::json::object()).dump() +
                       ":" +
                       payload.value("source_spans", nlohmann::json::array()).dump();
      if (!seen_candidates.insert(key).second) continue;
    }
    result.push_back(std::move(type));
  }
  return result;
}

std::size_t lcs_length(const std::vector<std::string>& left,
                       const std::vector<std::string>& right) {
  std::vector<std::size_t> previous(right.size() + 1);
  std::vector<std::size_t> current(right.size() + 1);
  for (const auto& left_value : left) {
    for (std::size_t index = 1; index <= right.size(); ++index) {
      current[index] = left_value == right[index - 1]
                           ? previous[index - 1] + 1
                           : std::max(previous[index], current[index - 1]);
    }
    std::swap(previous, current);
    std::fill(current.begin(), current.end(), 0);
  }
  return previous.back();
}

nlohmann::json counts(const std::vector<std::string>& sequence) {
  std::map<std::string, std::size_t> values;
  for (const auto& type : sequence) ++values[type];
  return values;
}

std::uint64_t unsigned_value(const nlohmann::json& value,
                             std::string_view key) {
  const auto found = value.find(key);
  return found != value.end() && found->is_number_unsigned()
             ? found->get<std::uint64_t>()
             : found != value.end() && found->is_number_integer()
                   ? static_cast<std::uint64_t>(
                         std::max<std::int64_t>(0, found->get<std::int64_t>()))
                   : 0;
}

nlohmann::json nullable_value(const nlohmann::json& value,
                              std::string_view key) {
  const auto found = value.find(key);
  return found == value.end() || found->is_null() ? nlohmann::json(nullptr)
                                                  : *found;
}

std::vector<std::uint64_t> candidate_boundaries(const nlohmann::json& payload) {
  std::vector<std::uint64_t> result;
  const auto append_span = [&result](const nlohmann::json& span) {
    if (!span.is_object()) return;
    result.push_back(unsigned_value(span, "start"));
    result.push_back(unsigned_value(span, "end"));
  };
  if (const auto wake = payload.find("wake_span"); wake != payload.end()) {
    append_span(*wake);
  }
  if (const auto spans = payload.find("source_spans");
      spans != payload.end() && spans->is_array()) {
    for (const auto& span : *spans) append_span(span);
  }
  return result;
}

std::vector<nlohmann::json> unique_candidates(
    const std::vector<nlohmann::json>& events) {
  std::vector<nlohmann::json> result;
  std::set<std::string> seen;
  for (const auto& event : events) {
    if (type_of(event) != "candidate") continue;
    const auto& payload = payload_of(event);
    const auto id = string_value(payload, "utterance_id");
    const auto key = id + ":" + string_value(payload, "position") + ":" +
                     payload.value("wake_span", nlohmann::json::object()).dump() +
                     ":" + payload.value("source_spans", nlohmann::json::array()).dump();
    if (seen.insert(key).second) result.push_back(payload);
  }
  return result;
}

nlohmann::json compare_candidates(const std::vector<nlohmann::json>& recorded,
                                  const std::vector<nlohmann::json>& replayed,
                                  std::uint32_t frame_samples) {
  const auto old_values = unique_candidates(recorded);
  const auto new_values = unique_candidates(replayed);
  const auto paired = std::min(old_values.size(), new_values.size());
  std::size_t exact{};
  std::size_t within_frame{};
  nlohmann::json details = nlohmann::json::array();
  for (std::size_t index = 0; index < paired; ++index) {
    const auto old_boundaries = candidate_boundaries(old_values[index]);
    const auto new_boundaries = candidate_boundaries(new_values[index]);
    std::uint64_t maximum_error{};
    const bool same_shape = old_boundaries.size() == new_boundaries.size();
    if (same_shape) {
      for (std::size_t boundary = 0; boundary < old_boundaries.size(); ++boundary) {
        const auto left = old_boundaries[boundary];
        const auto right = new_boundaries[boundary];
        maximum_error = std::max(maximum_error,
                                 left > right ? left - right : right - left);
      }
    }
    const bool position_equal = string_value(old_values[index], "position") ==
                                string_value(new_values[index], "position");
    const bool is_exact = same_shape && position_equal && maximum_error == 0;
    const bool is_within = same_shape && position_equal &&
                           maximum_error <= frame_samples;
    exact += is_exact ? 1U : 0U;
    within_frame += is_within ? 1U : 0U;
    details.push_back({{"index", index},
                       {"recorded_utterance_id",
                        string_value(old_values[index], "utterance_id")},
                       {"replayed_utterance_id",
                        string_value(new_values[index], "utterance_id")},
                       {"position_equal", position_equal},
                       {"boundary_shape_equal", same_shape},
                       {"max_boundary_error_samples", maximum_error},
                       {"exact", is_exact},
                       {"within_one_frame", is_within}});
  }
  return {{"recorded_count", old_values.size()},
          {"replayed_count", new_values.size()},
          {"paired_count", paired},
          {"exact_count", exact},
          {"within_one_frame_count", within_frame},
          {"all_within_one_frame",
           old_values.size() == new_values.size() && paired == within_frame},
          {"details", std::move(details)}};
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::ranges::sort(values);
  const auto index = static_cast<std::size_t>(std::ceil(
      fraction * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

nlohmann::json distribution(const std::vector<double>& values) {
  if (values.empty()) {
    return {{"sample_count", 0}, {"p50", nullptr}, {"p95", nullptr},
            {"max", nullptr}};
  }
  return {{"sample_count", values.size()},
          {"p50", percentile(values, 0.50)},
          {"p95", percentile(values, 0.95)},
          {"max", percentile(values, 1.0)}};
}

nlohmann::json asr_timing(const std::vector<nlohmann::json>& events) {
  std::vector<double> latencies;
  std::vector<double> inference;
  std::vector<double> rtfs;
  std::size_t final_count{};
  for (const auto& event : events) {
    if (type_of(event) != "asr_final") continue;
    ++final_count;
    const auto& payload = payload_of(event);
    if (const auto value = payload.find("latency_ms");
        value != payload.end() && value->is_number()) {
      latencies.push_back(value->get<double>());
    }
    if (const auto value = payload.find("inference_ms");
        value != payload.end() && value->is_number()) {
      inference.push_back(value->get<double>());
    }
    if (const auto value = payload.find("rtf");
        value != payload.end() && value->is_number()) {
      rtfs.push_back(value->get<double>());
    }
  }
  return {{"final_count", final_count},
          {"latency_ms", distribution(latencies)},
          {"inference_ms", distribution(inference)},
          {"rtf", distribution(rtfs)}};
}

nlohmann::json command_plans(const std::vector<nlohmann::json>& events) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto& event : events) {
    if (type_of(event) != "command_plan") continue;
    const auto& payload = payload_of(event);
    nlohmann::json actions = nlohmann::json::array();
    if (const auto found = payload.find("actions");
        found != payload.end() && found->is_array()) {
      for (const auto& action : *found) {
        actions.push_back({{"sequence", action.value("sequence", 0)},
                           {"type", string_value(action, "type")},
                           {"volume_delta_percent",
                            nullable_value(action,
                                           "volume_delta_percent")}});
      }
    }
    result.push_back({{"utterance_id", string_value(payload, "utterance_id")},
                      {"normalized_text", string_value(payload, "normalized_text")},
                      {"execution_mode", string_value(payload, "execution_mode")},
                      {"actions", std::move(actions)}});
  }
  return result;
}

nlohmann::json compare_plans(const std::vector<nlohmann::json>& recorded,
                             const std::vector<nlohmann::json>& replayed) {
  const auto old_plans = command_plans(recorded);
  const auto new_plans = command_plans(replayed);
  const auto paired = std::min(old_plans.size(), new_plans.size());
  std::size_t matching{};
  for (std::size_t index = 0; index < paired; ++index) {
    if (old_plans[index]["actions"] == new_plans[index]["actions"]) ++matching;
  }
  const auto denominator = std::max(old_plans.size(), new_plans.size());
  return {{"recorded_count", old_plans.size()},
          {"replayed_count", new_plans.size()},
          {"has_comparable_plans", denominator != 0},
          {"matching_action_sequences", matching},
          {"action_sequence_accuracy",
           denominator == 0
               ? nlohmann::json(nullptr)
               : nlohmann::json(static_cast<double>(matching) /
                                static_cast<double>(denominator))},
          {"replayed", new_plans}};
}

nlohmann::json dry_run_actions(const std::vector<nlohmann::json>& events) {
  nlohmann::json actions = nlohmann::json::array();
  bool all_dry_run = true;
  std::size_t terminal_count{};
  for (const auto& event : events) {
    const auto type = type_of(event);
    if (!type.starts_with("action_") || type == "action_queued" ||
        type == "action_started" || type == "action_submit_failed") {
      continue;
    }
    const auto& payload = payload_of(event);
    const auto status = string_value(payload, "status", type.substr(7));
    ++terminal_count;
    all_dry_run = all_dry_run && status == "dry_run";
    actions.push_back({{"utterance_id", string_value(payload, "utterance_id")},
                       {"action_id", string_value(payload, "action_id")},
                       {"sequence", payload.value("sequence", 0)},
                       {"type", string_value(payload, "type")},
                       {"status", status},
                       {"requested_volume_delta_percent",
                        nullable_value(payload,
                                       "requested_volume_delta_percent")},
                       {"message", string_value(payload, "message")}});
  }
  return {{"terminal_count", terminal_count},
          {"has_terminal_actions", terminal_count != 0},
          {"all_terminal_actions_dry_run", all_dry_run},
          {"actions", std::move(actions)}};
}

}  // namespace

std::vector<nlohmann::json> read_benchmark_events(
    const std::filesystem::path& path, std::vector<std::string>* errors) {
  std::vector<nlohmann::json> result;
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (errors) errors->push_back("cannot open " + path.string());
    return result;
  }
  std::string line;
  std::size_t line_number{};
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;
    try {
      auto event = nlohmann::json::parse(line);
      if (event.is_object()) result.push_back(std::move(event));
    } catch (const std::exception& error) {
      if (errors) {
        errors->push_back("line " + std::to_string(line_number) + ": " +
                          error.what());
      }
    }
  }
  return result;
}

nlohmann::json build_benchmark_comparison(
    const std::vector<nlohmann::json>& recorded,
    const std::vector<nlohmann::json>& replayed,
    std::uint32_t frame_samples) {
  const auto old_sequence = comparable_sequence(recorded);
  const auto new_sequence = comparable_sequence(replayed);
  const auto lcs = lcs_length(old_sequence, new_sequence);
  const auto denominator = std::max(old_sequence.size(), new_sequence.size());
  return {{"recorded_event_count", recorded.size()},
          {"replayed_event_count", replayed.size()},
          {"event_sequence",
           {{"recorded", old_sequence},
            {"replayed", new_sequence},
            {"recorded_counts", counts(old_sequence)},
            {"replayed_counts", counts(new_sequence)},
            {"exact", old_sequence == new_sequence},
            {"lcs_length", lcs},
            {"similarity", denominator == 0
                               ? 1.0
                               : static_cast<double>(lcs) /
                                     static_cast<double>(denominator)}}},
          {"candidates", compare_candidates(recorded, replayed, frame_samples)},
          {"asr_timing", {{"recorded", asr_timing(recorded)},
                           {"replayed", asr_timing(replayed)}}},
          {"command_plans", compare_plans(recorded, replayed)},
          {"dry_run", dry_run_actions(replayed)}};
}

nlohmann::json summarize_replayed_utterances(
    const std::vector<nlohmann::json>& events, std::uint64_t origin_sample) {
  constexpr double kSampleRate = 16000.0;
  const auto seconds = [origin_sample](std::uint64_t sample) {
    return sample >= origin_sample
               ? static_cast<double>(sample - origin_sample) / kSampleRate
               : -static_cast<double>(origin_sample - sample) / kSampleRate;
  };

  auto hits = nlohmann::json::array();
  std::map<std::string, nlohmann::json> utterances;
  std::vector<std::string> order;
  std::set<std::string> microphone_activations;
  const auto utterance = [&](const std::string& id) -> nlohmann::json& {
    auto [it, inserted] = utterances.try_emplace(id);
    if (inserted) {
      order.push_back(id);
      it->second = {{"utterance_id", id},
                    {"origin", nullptr},
                    {"position", nullptr},
                    {"activation_id", nullptr},
                    {"audio_source", "aec"},
                    {"candidate", nullptr},
                    {"asr_text", nullptr},
                    {"asr_latency_ms", nullptr},
                    {"outcome", "pending"},
                    {"normalized_text", nullptr},
                    {"actions", nlohmann::json::array()},
                    {"rejection", nullptr}};
    }
    return it->second;
  };

  for (const auto& event : events) {
    const auto type = type_of(event);
    const auto& payload = payload_of(event);
    if (type == "kws_hit") {
      const auto span = payload.value("wake_span", nlohmann::json::object());
      hits.push_back({{"keyword", string_value(payload, "keyword")},
                      {"detector", string_value(payload, "detector", "aec")},
                      {"wake_start_s", seconds(unsigned_value(span, "start"))},
                      {"wake_end_s", seconds(unsigned_value(span, "end"))},
                      {"detected_s",
                       seconds(unsigned_value(payload, "detected_at_sample"))},
                      {"detected_sample", unsigned_value(payload, "detected_at_sample")},
                      {"suppressed", nullptr}});
      continue;
    }
    if (type == "kws_suppressed") {
      // Suppression is reported right after the hit it applies to.
      const auto at = unsigned_value(event, "timestamp_sample");
      for (auto it = hits.rbegin(); it != hits.rend(); ++it) {
        if ((*it)["detected_sample"] == at &&
            (*it)["detector"] == string_value(payload, "detector", "aec")) {
          (*it)["suppressed"] = string_value(payload, "reason", "suppressed");
          break;
        }
      }
      continue;
    }
    if (type == "recognition_audio_fallback") {
      microphone_activations.insert(string_value(payload, "activation_id"));
      continue;
    }
    const auto id = string_value(payload, "utterance_id");
    if (id.empty()) continue;
    if (type == "candidate") {
      auto& record = utterance(id);
      std::uint64_t start = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t end{};
      for (const auto& span : payload.value("source_spans", nlohmann::json::array())) {
        start = std::min(start, unsigned_value(span, "start"));
        end = std::max(end, unsigned_value(span, "end"));
      }
      if (start > end) start = end;
      record["origin"] = string_value(payload, "origin");
      record["position"] = nullable_value(payload, "position");
      record["activation_id"] = string_value(payload, "activation_id");
      record["candidate"] = {{"start_s", seconds(start)},
                             {"end_s", seconds(end)},
                             {"start_sample", start},
                             {"end_sample", end},
                             {"duration_s", static_cast<double>(end - start) / kSampleRate},
                             {"boundary_source", string_value(payload, "boundary_source")},
                             {"truncated", payload.value("truncated", false)},
                             {"timed_out", payload.value("timed_out", false)},
                             {"discontinuity", payload.value("discontinuity", false)}};
      if (record["outcome"] == "pending") record["outcome"] = "no_asr";
    } else if (type == "candidate_rejected") {
      auto& record = utterance(id);
      if (record["origin"].is_null()) record["origin"] = string_value(payload, "origin");
      record["outcome"] = "candidate_rejected";
      record["rejection"] = {{"stage", "candidate"},
                             {"reason", string_value(payload, "reason")}};
    } else if (type == "asr_final") {
      auto& record = utterance(id);
      record["asr_text"] = string_value(payload, "text");
      record["asr_latency_ms"] = nullable_value(payload, "latency_ms");
      if (record["outcome"] == "pending" || record["outcome"] == "no_asr") {
        record["outcome"] = "asr_only";
      }
    } else if (type == "asr_error" || type == "asr_cancelled") {
      auto& record = utterance(id);
      if (record["outcome"] == "pending" || record["outcome"] == "no_asr") {
        record["outcome"] = type == "asr_error" ? "asr_error" : "asr_cancelled";
        record["rejection"] = {{"stage", "asr"},
                               {"reason", string_value(payload, "detail")}};
      }
    } else if (type == "parse_debug") {
      const auto error = payload.find("error");
      if (error != payload.end() && error->is_object()) {
        auto& record = utterance(id);
        record["parse_error"] = {{"stage", string_value(payload, "stage")},
                                 {"code", string_value(*error, "code")},
                                 {"message", string_value(*error, "message")}};
      }
    } else if (type == "command_plan") {
      auto& record = utterance(id);
      record["outcome"] = "plan";
      record["normalized_text"] = string_value(payload, "normalized_text");
      auto actions = nlohmann::json::array();
      for (const auto& action : payload.value("actions", nlohmann::json::array())) {
        actions.push_back({{"type", string_value(action, "type")},
                           {"volume_delta_percent",
                            nullable_value(action, "volume_delta_percent")}});
      }
      record["actions"] = std::move(actions);
    } else if (type == "command_rejected") {
      auto& record = utterance(id);
      record["outcome"] = "rejected";
      record["rejection"] = {{"stage", "command"},
                             {"reason", string_value(payload, "reason")}};
      if (record["asr_text"].is_null()) record["asr_text"] = string_value(payload, "text");
    }
  }

  std::vector<nlohmann::json> records;
  records.reserve(order.size());
  for (const auto& id : order) {
    auto record = std::move(utterances[id]);
    if (record["activation_id"].is_string() &&
        microphone_activations.contains(record["activation_id"].get<std::string>())) {
      record["audio_source"] = "microphone";
    }
    records.push_back(std::move(record));
  }
  const auto start_of = [](const nlohmann::json& record) {
    return record["candidate"].is_object()
               ? record["candidate"]["start_s"].get<double>()
               : std::numeric_limits<double>::max();
  };
  std::ranges::stable_sort(records, std::less{}, start_of);
  for (auto& hit : hits) hit.erase("detected_sample");
  return {{"keyword_hits", std::move(hits)}, {"utterances", std::move(records)}};
}

}  // namespace dvo
