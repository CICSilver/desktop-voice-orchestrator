#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

#include "dvo/preprocessor.h"

namespace {

dvo::AudioPacket stereo_packet(std::uint64_t qpc, float value) {
  dvo::AudioPacket packet;
  packet.stream = dvo::AudioStreamKind::microphone;
  packet.format = {48000, 2};
  packet.qpc_100ns = qpc;
  packet.samples.assign(480 * 2, value);
  return packet;
}

}  // namespace

TEST_CASE("bypass preprocessor downmixes resamples and assigns absolute QPC") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);

  const auto first = preprocessor.process(stereo_packet(1000000, 0.25F), std::nullopt);
  REQUIRE(first.frames.size() == 1);
  REQUIRE(first.frames[0].first_sample == 0);
  REQUIRE(first.frames[0].qpc_100ns == 1000000);
  REQUIRE(first.frames[0].samples[80] == Catch::Approx(0.25F));

  const auto second = preprocessor.process(stereo_packet(1100000, -0.5F), std::nullopt);
  REQUIRE(second.frames.size() == 1);
  REQUIRE(second.frames[0].first_sample == dvo::kFrameSamples);
  REQUIRE(second.frames[0].qpc_100ns == 1100000);
  REQUIRE(second.frames[0].samples[80] == Catch::Approx(-0.5F));
}

TEST_CASE("bypass preprocessor preserves the absolute sample clock across reset") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  (void)preprocessor.process(stereo_packet(2000000, 0.1F), std::nullopt);

  auto changed = stereo_packet(4000000, 0.2F);
  changed.format.sample_rate = 32000;
  changed.samples.assign(320 * 2, 0.2F);
  changed.discontinuity = true;
  const auto result = preprocessor.process(changed, std::nullopt);

  REQUIRE(result.reset_required);
  REQUIRE(result.frames.size() == 1);
  REQUIRE(result.frames[0].first_sample == dvo::kFrameSamples);
  REQUIRE(result.frames[0].qpc_100ns == 4000000);
  REQUIRE(result.frames[0].discontinuity);
}
