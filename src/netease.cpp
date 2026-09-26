#include "dvo/netease.h"

#include <algorithm>

namespace dvo {
namespace {

// Ids appear both as strings and as numbers in NetEase JSON.
[[nodiscard]] std::optional<std::string> id_text(const nlohmann::json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_number_unsigned()) return std::to_string(value.get<std::uint64_t>());
  if (value.is_number_integer()) return std::to_string(value.get<std::int64_t>());
  return std::nullopt;
}

}  // namespace

std::optional<NeteaseTrack> parse_netease_history_track(std::string_view json) {
  const auto value = nlohmann::json::parse(json, nullptr, false);
  if (!value.is_object()) return std::nullopt;
  NeteaseTrack track;
  if (const auto it = value.find("id"); it != value.end()) {
    if (auto id = id_text(*it)) track.id = std::move(*id);
  }
  if (const auto it = value.find("name"); it != value.end() && it->is_string()) {
    track.name = it->get<std::string>();
  }
  if (const auto it = value.find("artists"); it != value.end() && it->is_array()) {
    for (const auto& artist : *it) {
      if (!artist.is_object()) continue;
      if (const auto name = artist.find("name"); name != artist.end() && name->is_string() &&
                                                 !name->get<std::string>().empty()) {
        track.artists.push_back(name->get<std::string>());
      }
    }
  }
  if (track.id.empty() || track.name.empty()) return std::nullopt;
  return track;
}

bool netease_track_matches_media(const NeteaseTrack& track, std::string_view title,
                                 std::string_view artist) {
  if (track.name.empty() || track.name != title || artist.empty()) return false;
  return std::ranges::any_of(track.artists, [&](const std::string& name) {
    return artist.find(name) != std::string_view::npos;
  });
}

NcmResponse parse_ncm_response(std::string_view output) {
  NcmResponse response;
  const auto begin = output.find('{');
  const auto end = output.rfind('}');
  if (begin == std::string_view::npos || end == std::string_view::npos || end < begin) {
    response.error = "ncm-cli printed no JSON object";
    return response;
  }
  response.body = nlohmann::json::parse(output.substr(begin, end - begin + 1), nullptr, false);
  if (!response.body.is_object()) {
    response.error = "ncm-cli printed invalid JSON";
    return response;
  }
  const auto code = response.body.find("code");
  if (code == response.body.end() || !code->is_number_integer() || code->get<int>() != 200) {
    const auto message = response.body.find("message");
    response.error = "ncm-cli returned code " +
                     (code == response.body.end() ? std::string{"(none)"} : code->dump());
    if (message != response.body.end() && message->is_string()) {
      response.error += ": " + message->get<std::string>();
    }
  }
  return response;
}

std::optional<NcmSearchHit> find_ncm_search_hit(const nlohmann::json& body,
                                              std::string_view original_id) {
  const auto data = body.find("data");
  if (data == body.end() || !data->is_object()) return std::nullopt;
  const auto records = data->find("records");
  if (records == data->end() || !records->is_array()) return std::nullopt;
  for (const auto& record : *records) {
    if (!record.is_object()) continue;
    const auto original = record.find("originalId");
    if (original == record.end() || id_text(*original) != original_id) continue;
    const auto encrypted = record.find("id");
    if (encrypted == record.end() || !encrypted->is_string() ||
        encrypted->get<std::string>().empty()) {
      return std::nullopt;
    }
    const auto liked = record.find("liked");
    return NcmSearchHit{encrypted->get<std::string>(),
                        liked != record.end() && liked->is_boolean() && liked->get<bool>()};
  }
  return std::nullopt;
}

}  // namespace dvo
