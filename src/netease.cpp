#include "dvo/netease.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>

namespace dvo {
namespace {

// Ids appear both as strings and as numbers in NetEase JSON.
[[nodiscard]] std::optional<std::string> id_text(const nlohmann::json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_number_unsigned()) return std::to_string(value.get<std::uint64_t>());
  if (value.is_number_integer()) {
    const auto integer = value.get<std::int64_t>();
    if (integer < 0) return std::nullopt;
    return std::to_string(integer);
  }
  return std::nullopt;
}

[[nodiscard]] bool is_positive_integer_id(std::string_view value) {
  if (value.empty() || value.front() < '1' || value.front() > '9') return false;
  return std::ranges::all_of(value, [](char ch) { return ch >= '0' && ch <= '9'; });
}

[[nodiscard]] std::string base64_encode(std::string_view input) {
  static constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((input.size() + 2) / 3) * 4);
  std::size_t i = 0;
  while (i + 3 <= input.size()) {
    const std::uint32_t triple =
        (static_cast<std::uint32_t>(static_cast<unsigned char>(input[i])) << 16U) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(input[i + 1])) << 8U) |
        static_cast<std::uint32_t>(static_cast<unsigned char>(input[i + 2]));
    out.push_back(kAlphabet[(triple >> 18U) & 0x3FU]);
    out.push_back(kAlphabet[(triple >> 12U) & 0x3FU]);
    out.push_back(kAlphabet[(triple >> 6U) & 0x3FU]);
    out.push_back(kAlphabet[triple & 0x3FU]);
    i += 3;
  }
  const std::size_t rem = input.size() - i;
  if (rem == 1) {
    const std::uint32_t triple =
        static_cast<std::uint32_t>(static_cast<unsigned char>(input[i])) << 16U;
    out.push_back(kAlphabet[(triple >> 18U) & 0x3FU]);
    out.push_back(kAlphabet[(triple >> 12U) & 0x3FU]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    const std::uint32_t triple =
        (static_cast<std::uint32_t>(static_cast<unsigned char>(input[i])) << 16U) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(input[i + 1])) << 8U);
    out.push_back(kAlphabet[(triple >> 18U) & 0x3FU]);
    out.push_back(kAlphabet[(triple >> 12U) & 0x3FU]);
    out.push_back(kAlphabet[(triple >> 6U) & 0x3FU]);
    out.push_back('=');
  }
  return out;
}

[[nodiscard]] std::string normalize_match_key(std::string_view input) {
  std::size_t begin = 0;
  while (begin < input.size() &&
         static_cast<unsigned char>(input[begin]) <= 0x20) {
    ++begin;
  }
  std::size_t end = input.size();
  while (end > begin &&
         static_cast<unsigned char>(input[end - 1]) <= 0x20) {
    --end;
  }
  std::string out;
  out.reserve(end - begin);
  for (std::size_t i = begin; i < end; ++i) {
    const auto ch = static_cast<unsigned char>(input[i]);
    if (ch >= 'A' && ch <= 'Z') {
      out.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else {
      out.push_back(static_cast<char>(ch));
    }
  }
  return out;
}

void append_unique_artist(std::vector<std::string>& artists, std::string value) {
  if (normalize_match_key(value).empty()) return;
  if (std::ranges::find(artists, value) == artists.end()) {
    artists.push_back(std::move(value));
  }
}

void collect_artist_array(const nlohmann::json& array,
                          std::vector<std::string>& artists) {
  if (!array.is_array()) return;
  for (const auto& entry : array) {
    if (entry.is_string()) {
      append_unique_artist(artists, entry.get<std::string>());
      continue;
    }
    if (!entry.is_object()) continue;
    if (const auto name = entry.find("name");
        name != entry.end() && name->is_string()) {
      append_unique_artist(artists, name->get<std::string>());
    }
    if (const auto trans = entry.find("trans");
        trans != entry.end() && trans->is_string()) {
      append_unique_artist(artists, trans->get<std::string>());
    }
    for (const char* alias_key : {"alias", "alia"}) {
      if (const auto aliases = entry.find(alias_key);
          aliases != entry.end() && aliases->is_array()) {
        for (const auto& alias : *aliases) {
          if (alias.is_string()) {
            append_unique_artist(artists, alias.get<std::string>());
          }
        }
      }
    }
  }
}

[[nodiscard]] std::optional<NeteaseSearchCandidate> parse_candidate_object(
    const nlohmann::json& item, MediaSearchType type) {
  if (!item.is_object()) return std::nullopt;
  std::string resolved_id;
  if (const auto orig = item.find("originalId"); orig != item.end()) {
    if (auto id = id_text(*orig); id && is_positive_integer_id(*id)) {
      resolved_id = std::move(*id);
    }
  }
  if (resolved_id.empty()) {
    if (const auto id_it = item.find("id"); id_it != item.end()) {
      if (auto id = id_text(*id_it); id && is_positive_integer_id(*id)) {
        resolved_id = std::move(*id);
      }
    }
  }
  if (resolved_id.empty()) return std::nullopt;

  const auto name_it = item.find("name");
  if (name_it == item.end() || !name_it->is_string() ||
      name_it->get<std::string>().empty()) {
    return std::nullopt;
  }

  NeteaseSearchCandidate candidate;
  candidate.id = std::move(resolved_id);
  candidate.name = name_it->get<std::string>();

  if (type == MediaSearchType::song) {
    if (const auto ar = item.find("ar"); ar != item.end()) {
      collect_artist_array(*ar, candidate.artists);
    }
    if (const auto artists = item.find("artists"); artists != item.end()) {
      collect_artist_array(*artists, candidate.artists);
    }
    if (const auto artist_name = item.find("artistName");
        artist_name != item.end() && artist_name->is_string()) {
      append_unique_artist(candidate.artists, artist_name->get<std::string>());
    }
  } else {
    if (const auto creator = item.find("creator");
        creator != item.end() && creator->is_object()) {
      if (const auto nick = creator->find("nickname");
          nick != creator->end() && nick->is_string()) {
        append_unique_artist(candidate.artists, nick->get<std::string>());
      }
    }
    if (const auto artists = item.find("artists"); artists != item.end()) {
      collect_artist_array(*artists, candidate.artists);
    }
  }
  return candidate;
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

std::optional<std::string> build_netease_orpheus_play_url(
    MediaSearchType type, std::string_view item_id) {
  const auto type_str = std::string_view(to_string(type));
  if (type_str == "unknown" || !is_positive_integer_id(item_id)) {
    return std::nullopt;
  }
  const std::string payload = std::string("{\"type\":\"") + std::string(type_str) +
                              "\",\"id\":\"" + std::string(item_id) +
                              "\",\"cmd\":\"play\"}";
  return "orpheus://" + base64_encode(payload);
}

std::vector<NeteaseSearchCandidate> parse_netease_search_candidates(
    const nlohmann::json& body, MediaSearchType type) {
  std::vector<NeteaseSearchCandidate> candidates;
  if (!body.is_object()) return candidates;

  const auto append_from_array = [&](const nlohmann::json& arr) {
    if (!arr.is_array()) return;
    for (const auto& item : arr) {
      if (auto candidate = parse_candidate_object(item, type)) {
        candidates.push_back(std::move(*candidate));
      }
    }
  };

  const char* list_key =
      type == MediaSearchType::playlist ? "playlists" : "songs";
  if (const auto result = body.find("result");
      result != body.end() && result->is_object()) {
    if (const auto list = result->find(list_key); list != result->end()) {
      append_from_array(*list);
    }
  }
  if (candidates.empty()) {
    if (const auto data = body.find("data");
        data != body.end() && data->is_object()) {
      if (const auto records = data->find("records"); records != data->end()) {
        append_from_array(*records);
      }
      if (candidates.empty()) {
        if (const auto list = data->find(list_key); list != data->end()) {
          append_from_array(*list);
        }
      }
    }
  }
  return candidates;
}

std::optional<NeteaseSearchCandidate> select_netease_search_candidate(
    const nlohmann::json& body, MediaSearchType type, std::string_view artist) {
  auto candidates = parse_netease_search_candidates(body, type);
  if (candidates.empty()) return std::nullopt;

  const auto target_artist = normalize_match_key(artist);
  if (target_artist.empty()) return std::move(candidates.front());

  for (auto& candidate : candidates) {
    for (const auto& cand_artist : candidate.artists) {
      if (normalize_match_key(cand_artist) == target_artist) {
        return std::move(candidate);
      }
    }
  }

  for (auto& candidate : candidates) {
    for (const auto& cand_artist : candidate.artists) {
      const auto norm = normalize_match_key(cand_artist);
      if (!norm.empty() &&
          (norm.find(target_artist) != std::string::npos ||
           target_artist.find(norm) != std::string::npos)) {
        return std::move(candidate);
      }
    }
    if (type == MediaSearchType::playlist) {
      const auto norm_name = normalize_match_key(candidate.name);
      if (norm_name.find(target_artist) != std::string::npos) {
        return std::move(candidate);
      }
    }
  }

  if (type == MediaSearchType::playlist) {
    return std::move(candidates.front());
  }
  return std::nullopt;
}

}  // namespace dvo
