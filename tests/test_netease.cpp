#include <catch2/catch_test_macros.hpp>

#include "dvo/netease.h"

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
