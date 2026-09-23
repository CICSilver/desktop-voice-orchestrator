#include "dvo/benchmark_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
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

std::string type_of(const nlohmann::json& event) {
  return event.is_object() ? event.value("type", "") : "";
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
      const auto key = payload.value("utterance_id", "") + ":" +
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

std::string string_value(const nlohmann::json& value,
                         std::string_view key) {
  const auto found = value.find(key);
  return found != value.end() && found->is_string()
             ? found->get<std::string>()
             : std::string{};
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
    const auto id = payload.value("utterance_id", "");
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
    const bool position_equal = old_values[index].value("position", "") ==
                                new_values[index].value("position", "");
    const bool is_exact = same_shape && position_equal && maximum_error == 0;
    const bool is_within = same_shape && position_equal &&
                           maximum_error <= frame_samples;
    exact += is_exact ? 1U : 0U;
    within_frame += is_within ? 1U : 0U;
    details.push_back({{"index", index},
                       {"recorded_utterance_id",
                        old_values[index].value("utterance_id", "")},
                       {"replayed_utterance_id",
                        new_values[index].value("utterance_id", "")},
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
                           {"type", action.value("type", "")},
                           {"volume_delta_percent",
                            nullable_value(action,
                                           "volume_delta_percent")}});
      }
    }
    result.push_back({{"utterance_id", payload.value("utterance_id", "")},
                      {"normalized_text", payload.value("normalized_text", "")},
                      {"execution_mode", payload.value("execution_mode", "")},
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
    const auto status = payload.value("status", type.substr(7));
    ++terminal_count;
    all_dry_run = all_dry_run && status == "dry_run";
    actions.push_back({{"utterance_id", payload.value("utterance_id", "")},
                       {"action_id", payload.value("action_id", "")},
                       {"sequence", payload.value("sequence", 0)},
                       {"type", payload.value("type", "")},
                       {"status", status},
                       {"requested_volume_delta_percent",
                        nullable_value(payload,
                                       "requested_volume_delta_percent")},
                       {"message", payload.value("message", "")}});
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

}  // namespace dvo
