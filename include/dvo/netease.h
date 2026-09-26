#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace dvo {

// A song as the NetEase Cloud Music client records it in its local play
// history (Library/webdb.dat, table historyTracks, column jsonStr). The client
// adds a row when a song starts, so the newest row is the current song.
struct NeteaseTrack {
  std::string id;  // numeric "original" id
  std::string name;
  std::vector<std::string> artists;
};

[[nodiscard]] std::optional<NeteaseTrack> parse_netease_history_track(std::string_view json);

// Whether the Windows media session's metadata describes this track: the title
// must be identical and at least one recorded artist must appear in the
// session's artist text (the client may join several artists into one string).
[[nodiscard]] bool netease_track_matches_media(const NeteaseTrack& track, std::string_view title,
                                               std::string_view artist);

// ncm-cli prints one JSON object per call; a successful call has "code": 200.
struct NcmResponse {
  nlohmann::json body;
  std::string error;
  [[nodiscard]] bool ok() const { return error.empty(); }
};
[[nodiscard]] NcmResponse parse_ncm_response(std::string_view output);

struct NcmSearchHit {
  std::string encrypted_id;
  bool liked{};
};

// Finds the song with this original id in `ncm-cli search song` output. Only
// an exact id counts: different versions of a song share its title and artist.
[[nodiscard]] std::optional<NcmSearchHit> find_ncm_search_hit(const nlohmann::json& body,
                                                            std::string_view original_id);

}  // namespace dvo
