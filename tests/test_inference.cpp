#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <filesystem>

#include "dvo/inference.h"

#if DVO_HAS_SHERPA
#include <sherpa-onnx/c-api/c-api.h>
#endif

TEST_CASE("pinned sherpa model detects an official reference keyword") {
#if !DVO_HAS_SHERPA
  SKIP("core-only build does not include sherpa-onnx");
#else
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  const auto model = root / "models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20";
  const auto wave_path = model / "test_wavs/en_0.wav";
  if (!std::filesystem::is_regular_file(wave_path)) {
    SKIP("run the fetch_models target to enable the model integration test");
  }

  dvo::KwsConfig config;
  config.encoder = model / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
  config.decoder = model / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx";
  config.joiner = model / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
  config.tokens = model / "tokens.txt";
  config.keywords = model / "test_wavs/keywords.txt";
  auto kws = dvo::create_keyword_spotter(config);
  REQUIRE(kws->available());

  const auto* wave = SherpaOnnxReadWave(wave_path.string().c_str());
  REQUIRE(wave != nullptr);
  REQUIRE(wave->sample_rate == static_cast<std::int32_t>(dvo::kProcessingSampleRate));

  std::optional<dvo::KwsHit> detected;
  std::uint64_t offset{};
  const auto feed = [&](const float* samples, std::size_t count) {
    std::size_t cursor{};
    while (cursor < count && !detected) {
      dvo::NormalizedFrame frame;
      frame.first_sample = offset;
      const auto n = std::min<std::size_t>(frame.samples.size(), count - cursor);
      std::copy_n(samples + cursor, n, frame.samples.begin());
      detected = kws->accept(frame);
      cursor += n;
      offset += frame.samples.size();
    }
  };
  feed(wave->samples, static_cast<std::size_t>(wave->num_samples));
  std::array<float, 8000> tail{};
  feed(tail.data(), tail.size());
  SherpaOnnxFreeWave(wave);

  REQUIRE(detected.has_value());
  REQUIRE(detected->keyword == "LIGHT_UP");
  REQUIRE_FALSE(detected->tokens.empty());
  REQUIRE(detected->wake_span.start < detected->wake_span.end);
  REQUIRE(detected->wake_span.end <= detected->detected_at_sample + dvo::kProcessingSampleRate);
#endif
}
