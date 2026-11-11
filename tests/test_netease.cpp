#include <catch2/catch_test_macros.hpp>

#include "dvo/netease.h"
#include "dvo/netease_cdp.h"

TEST_CASE("NetEase play-history rows parse into id, name and artists") {
  const auto track = dvo::parse_netease_history_track(
      R"({"id":"26600299","name":"不要在我寂寞的时候说爱我","artists":[{"id":"1","name":"T.R.Y."}],)"
      R"("duration":251000})");
  REQUIRE(track);
  CHECK(track->id == "26600299");
  CHECK(track->name == "不要在我寂寞的时候说爱我");
  CHECK(track->artists == std::vector<std::string>{"T.R.Y."});

  const auto numeric = dvo::parse_netease_history_track(R"({"id":159568,"name":"花间梦事","artists":[]})");
  REQUIRE(numeric);
  CHECK(numeric->id == "159568");

  CHECK_FALSE(dvo::parse_netease_history_track("not json"));
  CHECK_FALSE(dvo::parse_netease_history_track(R"({"name":"no id"})"));
}

TEST_CASE("A history row matches the media session only by exact title and a shared artist") {
  const dvo::NeteaseTrack track{"1", "时光卷轴", {"封茗囧菌", "双笙（陈元汐）"}};
  CHECK(dvo::netease_track_matches_media(track, "时光卷轴", "封茗囧菌/双笙（陈元汐）"));
  CHECK(dvo::netease_track_matches_media(track, "时光卷轴", "封茗囧菌"));
  CHECK_FALSE(dvo::netease_track_matches_media(track, "时光卷轴 (Live)", "封茗囧菌"));
  CHECK_FALSE(dvo::netease_track_matches_media(track, "时光卷轴", "别的歌手"));
  CHECK_FALSE(dvo::netease_track_matches_media(track, "时光卷轴", ""));
  CHECK_FALSE(dvo::netease_track_matches_media({"1", "时光卷轴", {}}, "时光卷轴", "封茗囧菌"));
}

TEST_CASE("ncm-cli responses need code 200") {
  const auto ok = dvo::parse_ncm_response("{\n  \"code\": 200,\n  \"data\": null\n}\n");
  CHECK(ok.ok());
  const auto rejected = dvo::parse_ncm_response(R"({"code":301,"message":"需要登录"})");
  REQUIRE_FALSE(rejected.ok());
  CHECK(rejected.error.find("301") != std::string::npos);
  CHECK(rejected.error.find("需要登录") != std::string::npos);
  CHECK_FALSE(dvo::parse_ncm_response("").ok());
  CHECK_FALSE(dvo::parse_ncm_response("{broken").ok());
}

TEST_CASE("The DevTools channel targets the client's main page") {
  // As listed by client 3.1.41 while starting and once loaded.
  const auto starting = nlohmann::json::parse(R"([{"type":"page","url":"orpheus://native/start.html",
      "webSocketDebuggerUrl":"ws://127.0.0.1:9223/devtools/page/A1"}])");
  CHECK_FALSE(dvo::netease_cdp_page_path(starting));
  const auto loaded = nlohmann::json::parse(R"([
      {"type":"page","url":"orpheus://orpheus/pub/subApp.html","webSocketDebuggerUrl":"ws://127.0.0.1:9223/devtools/page/B2"},
      {"type":"page","url":"orpheus://orpheus/pub/app.html","webSocketDebuggerUrl":"ws://127.0.0.1:9223/devtools/page/EB2C"}])");
  CHECK(dvo::netease_cdp_page_path(loaded) == "/devtools/page/EB2C");
  CHECK_FALSE(dvo::netease_cdp_page_path(nlohmann::json::object()));
}

TEST_CASE("DevTools evaluate replies yield the value or the page's error") {
  const auto value = dvo::netease_cdp_result(nlohmann::json::parse(
      R"({"id":1,"result":{"result":{"type":"object","value":{"ok":true,"mode":"playRandom"}}}})"));
  CHECK(value["mode"] == "playRandom");
  try {
    (void)dvo::netease_cdp_result(nlohmann::json::parse(
        R"({"id":1,"result":{"result":{"type":"object"},"exceptionDetails":{"text":"Uncaught",
            "exception":{"description":"Error: NetEase store not found"}}}})"));
    FAIL("a page exception must throw");
  } catch (const dvo::NeteaseCdpError& error) {
    CHECK(error.kind() == dvo::NeteaseCdpError::Kind::script);
    CHECK(std::string(error.what()).find("store not found") != std::string::npos);
  }
  CHECK_THROWS_AS(dvo::netease_cdp_result(nlohmann::json::parse(R"({"id":1,"error":{"code":-32000}})")),
                  dvo::NeteaseCdpError);
}

TEST_CASE("Play mode scripts embed the mode as a JavaScript string") {
  const auto script = dvo::netease_play_mode_script("playOneCycle");
  CHECK(script.find(R"(const target = "playOneCycle";)") != std::string::npos);
  CHECK(script.find("playing/switchPlayingMode") != std::string::npos);
  CHECK(dvo::netease_play_daily_script().find("'dailyRecommend'") != std::string::npos);
}

TEST_CASE("Only the search record with the exact original id is used") {
  // Two versions of the same song share its title and artist.
  const auto body = nlohmann::json::parse(R"({"code":200,"data":{"recordCount":2,"records":[
      {"originalId":382717,"id":"4E5BD5E96C396727D8B341ED48D2DDDF","name":"不要在我寂寞的时候说爱我","liked":false},
      {"originalId":26600299,"id":"D65BB5674659A113FBC5595AB2A21562","name":"不要在我寂寞的时候说爱我","liked":true}]}})");
  const auto hit = dvo::find_ncm_search_hit(body, "26600299");
  REQUIRE(hit);
  CHECK(hit->encrypted_id == "D65BB5674659A113FBC5595AB2A21562");
  CHECK(hit->liked);
  const auto other = dvo::find_ncm_search_hit(body, "382717");
  REQUIRE(other);
  CHECK_FALSE(other->liked);
  CHECK_FALSE(dvo::find_ncm_search_hit(body, "1"));
  CHECK_FALSE(dvo::find_ncm_search_hit(nlohmann::json::parse(R"({"code":200,"data":null})"), "1"));
}

namespace {

std::string decode_base64_for_test(std::string_view input) {
  static constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  int val = 0;
  int valb = -8;
  for (const char ch : input) {
    if (ch == '=') break;
    const auto pos = kAlphabet.find(ch);
    REQUIRE(pos != std::string_view::npos);
    val = (val << 6) + static_cast<int>(pos);
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<char>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("build_netease_orpheus_play_url encodes valid song and playlist IDs and rejects invalid ones") {
  const auto song_url =
      dvo::build_netease_orpheus_play_url(dvo::MediaSearchType::song, "186016");
  REQUIRE(song_url.has_value());
  REQUIRE(song_url->starts_with("orpheus://"));
  const auto song_json = nlohmann::json::parse(
      decode_base64_for_test(std::string_view(*song_url).substr(10)));
  CHECK(song_json == nlohmann::json{{"type", "song"}, {"id", "186016"}, {"cmd", "play"}});

  const auto playlist_url = dvo::build_netease_orpheus_play_url(
      dvo::MediaSearchType::playlist, "3778678");
  REQUIRE(playlist_url.has_value());
  REQUIRE(playlist_url->starts_with("orpheus://"));
  const auto playlist_json = nlohmann::json::parse(
      decode_base64_for_test(std::string_view(*playlist_url).substr(10)));
  CHECK(playlist_json ==
        nlohmann::json{{"type", "playlist"}, {"id", "3778678"}, {"cmd", "play"}});

  for (const char* bad_id : {"", "0", "-186016", "0186016", "186a16", "186016;play"}) {
    CAPTURE(bad_id);
    CHECK_FALSE(
        dvo::build_netease_orpheus_play_url(dvo::MediaSearchType::song, bad_id)
            .has_value());
  }
}

TEST_CASE("NetEase search candidate selection filters covers when artist is specified") {
  const auto song_body = nlohmann::json::parse(R"json({
    "code": 200,
    "result": {
      "songs": [
        {
          "id": 990001,
          "name": "晴天 (温柔男声版)",
          "ar": [{"name": "路人翻唱"}]
        },
        {
          "id": 186016,
          "name": "晴天",
          "ar": [{"name": "周杰伦", "alias": ["Jay Chou"]}]
        }
      ]
    }
  })json");

  // Without artist: picks the first valid candidate.
  const auto first_hit =
      dvo::select_netease_search_candidate(song_body, dvo::MediaSearchType::song);
  REQUIRE(first_hit.has_value());
  CHECK(first_hit->id == "990001");

  // With artist: skips the cover and selects Jay Chou's original track.
  const auto jay_hit = dvo::select_netease_search_candidate(
      song_body, dvo::MediaSearchType::song, "周杰伦");
  REQUIRE(jay_hit.has_value());
  CHECK(jay_hit->id == "186016");
  CHECK(jay_hit->name == "晴天");

  // Alias match (case-insensitive) also resolves to Jay Chou's track.
  const auto alias_hit = dvo::select_netease_search_candidate(
      song_body, dvo::MediaSearchType::song, "jay chou");
  REQUIRE(alias_hit.has_value());
  CHECK(alias_hit->id == "186016");

  // Unmatched artist on song search returns nullopt instead of playing an unrelated cover.
  CHECK_FALSE(dvo::select_netease_search_candidate(
      song_body, dvo::MediaSearchType::song, "林俊杰").has_value());

  // Playlist search JSON (`result.playlists`) and ncm-cli JSON (`data.records`).
  const auto playlist_body = nlohmann::json::parse(R"json({
    "code": 200,
    "result": {
      "playlists": [
        {"id": 50001, "name": "深夜轻音乐纯享", "creator": {"nickname": "睡眠电台"}},
        {"id": 50002, "name": "周杰伦经典全集", "creator": {"nickname": "云音乐歌单"}}
      ]
    }
  })json");
  const auto default_playlist = dvo::select_netease_search_candidate(
      playlist_body, dvo::MediaSearchType::playlist);
  REQUIRE(default_playlist.has_value());
  CHECK(default_playlist->id == "50001");

  const auto artist_playlist = dvo::select_netease_search_candidate(
      playlist_body, dvo::MediaSearchType::playlist, "周杰伦");
  REQUIRE(artist_playlist.has_value());
  CHECK(artist_playlist->id == "50002");

  const auto ncm_body = nlohmann::json::parse(R"json({
    "code": 200,
    "data": {
      "records": [
        {"originalId": 202001, "id": "ENC_COVER", "name": "七里香 (Cover)", "artists": [{"name": "某歌手"}]},
        {"originalId": 186001, "id": "ENC_ORIG", "name": "七里香", "artists": [{"name": "周杰伦"}]}
      ]
    }
  })json");
  const auto ncm_selected = dvo::select_netease_search_candidate(
      ncm_body, dvo::MediaSearchType::song, "周杰伦");
  REQUIRE(ncm_selected.has_value());
  CHECK(ncm_selected->id == "186001");
}
