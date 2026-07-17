#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

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

dvo::AudioPacket timeline_packet(dvo::AudioStreamKind stream, std::uint64_t qpc,
                                 std::uint64_t device_position,
                                 std::uint64_t sequence, float value) {
  auto packet = stereo_packet(qpc, value);
  packet.stream = stream;
  packet.arrival_qpc_100ns = qpc + 20'000;
  packet.device_position = device_position;
  packet.stream_epoch = 1;
  packet.sequence = sequence;
  return packet;
}

dvo::AudioPacket rate_timeline_packet(dvo::AudioStreamKind stream,
                                      std::uint64_t qpc,
                                      std::uint64_t device_position,
                                      std::uint64_t sequence,
                                      std::uint32_t sample_rate,
                                      std::size_t frame_count, float value) {
  dvo::AudioPacket packet;
  packet.stream = stream;
  packet.format = {sample_rate, 1};
  packet.qpc_100ns = qpc;
  packet.arrival_qpc_100ns = qpc + 20'000;
  packet.device_position = device_position;
  packet.stream_epoch = 1;
  packet.sequence = sequence;
  packet.samples.assign(frame_count, value);
  return packet;
}

dvo::AudioPacket mono_timeline_packet(dvo::AudioStreamKind stream,
                                      std::uint64_t frame_index,
                                      const std::vector<float>& samples) {
  dvo::AudioPacket packet;
  packet.stream = stream;
  packet.format = {16000, 1};
  packet.qpc_100ns = 80'000'000 + frame_index * 100'000;
  packet.arrival_qpc_100ns = packet.qpc_100ns + 5'000;
  packet.device_position = frame_index * dvo::kFrameSamples;
  packet.stream_epoch = 1;
  packet.sequence = frame_index + 1;
  const auto begin = static_cast<std::size_t>(frame_index) * dvo::kFrameSamples;
  packet.samples.assign(samples.begin() + static_cast<std::ptrdiff_t>(begin),
                        samples.begin() + static_cast<std::ptrdiff_t>(
                                              begin + dvo::kFrameSamples));
  return packet;
}

}  // namespace

TEST_CASE("AEC application config maps every preprocessor setting") {
  dvo::AecConfig config;
  config.alignment_wait_ms = 17;
  config.target_render_buffer_ms = 70;
  config.max_render_buffer_ms = 640;
  config.drift_window_ms = 7000;
  config.hard_resync_error_ms = 91;
  config.max_drift_ppm = 432;
  config.delay_offset_ms = -12;
  config.microphone_channel_index = -1;
  config.auto_delay_enabled = false;
  config.auto_delay_min_ms = 20;
  config.auto_delay_max_ms = 180;
  config.auto_delay_window_ms = 1500;
  config.auto_delay_update_ms = 300;
  config.auto_delay_min_correlation = 0.48F;
  config.high_pass_filter = false;
  config.noise_suppression = true;
  config.gain_control = true;
  config.stats_hz = 4;

  const auto timeline = dvo::make_preprocessor_timeline_config(config);
  REQUIRE(timeline.alignment_wait_ms == 17);
  REQUIRE(timeline.target_render_buffer_ms == 70);
  REQUIRE(timeline.max_buffer_ms == 640);
  REQUIRE(timeline.drift_window_ms == 7000);
  REQUIRE(timeline.hard_resync_error_ms == 91);
  REQUIRE(timeline.max_drift_ppm == Catch::Approx(432.0));
  REQUIRE(timeline.delay_offset_ms == -12);
  REQUIRE(timeline.microphone_channel_index == -1);
  REQUIRE_FALSE(timeline.auto_delay_enabled);
  REQUIRE(timeline.auto_delay_min_ms == 20);
  REQUIRE(timeline.auto_delay_max_ms == 180);
  REQUIRE(timeline.auto_delay_window_ms == 1500);
  REQUIRE(timeline.auto_delay_update_ms == 300);
  REQUIRE(timeline.auto_delay_min_correlation == Catch::Approx(0.48F));
  REQUIRE_FALSE(timeline.high_pass_filter);
  REQUIRE(timeline.noise_suppression);
  REQUIRE(timeline.gain_control);
  REQUIRE(timeline.stats_hz == 4);
}

TEST_CASE("microphone channel can be selected or explicitly averaged") {
  dvo::AudioConfig config;
  auto packet = stereo_packet(1'500'000, 0.0F);
  for (std::size_t frame = 0; frame < 480; ++frame) {
    packet.samples[frame * 2] = 0.25F;
    packet.samples[frame * 2 + 1] = -0.75F;
  }

  SECTION("select one physical channel") {
    dvo::PreprocessorTimelineConfig timeline;
    timeline.microphone_channel_index = 1;
    dvo::BypassPreprocessor preprocessor(config, timeline);
    REQUIRE(preprocessor.PushPacket(packet).accepted);
    dvo::NormalizedFrame frame;
    REQUIRE(preprocessor.TryPopFrame(frame));
    REQUIRE(frame.samples[80] == Catch::Approx(-0.75F));
  }

  SECTION("average all channels") {
    dvo::PreprocessorTimelineConfig timeline;
    timeline.microphone_channel_index = -1;
    dvo::BypassPreprocessor preprocessor(config, timeline);
    REQUIRE(preprocessor.PushPacket(packet).accepted);
    dvo::NormalizedFrame frame;
    REQUIRE(preprocessor.TryPopFrame(frame));
    REQUIRE(frame.samples[80] == Catch::Approx(-0.25F));
  }

  SECTION("reject an unavailable channel without reading outside the packet") {
    dvo::PreprocessorTimelineConfig timeline;
    timeline.microphone_channel_index = 2;
    dvo::BypassPreprocessor preprocessor(config, timeline);
    const auto result = preprocessor.PushPacket(packet);
    REQUIRE_FALSE(result.accepted);
    REQUIRE(result.reset_required);
    REQUIRE(result.reset_reason == dvo::PreprocessResetReason::invalid_packet);
  }
}

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

TEST_CASE("push and pop API aligns render and microphone on the same timeline") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);

  const auto render = timeline_packet(dvo::AudioStreamKind::loopback, 3'000'000, 0, 1,
                                      -0.75F);
  const auto microphone = timeline_packet(dvo::AudioStreamKind::microphone, 3'000'000,
                                          0, 1, 0.25F);
  REQUIRE(preprocessor.PushPacket(render).accepted);
  const auto pushed = preprocessor.PushPacket(microphone);
  REQUIRE(pushed.accepted);
  REQUIRE(pushed.frames_ready == 1);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.first_sample == 0);
  REQUIRE(frame.qpc_100ns == 3'000'000);
  REQUIRE(frame.samples[80] == Catch::Approx(0.25F));
  REQUIRE_FALSE(preprocessor.TryPopFrame(frame));

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.render_seen);
  REQUIRE(diagnostics.render_available);
  REQUIRE(diagnostics.render_packets == 1);
  REQUIRE(diagnostics.microphone_packets == 1);
}

TEST_CASE("WebRTC preprocessor has an explicit compilable fallback") {
  if (dvo::webrtc_aec3_compiled()) return;

  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.delay_offset_ms = 17;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);
  const auto microphone = timeline_packet(dvo::AudioStreamKind::microphone, 4'000'000,
                                          0, 1, 0.4F);
  REQUIRE(preprocessor.PushPacket(microphone).accepted);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.samples[40] == Catch::Approx(0.4F).margin(0.002F));
  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.aec_requested);
  REQUIRE_FALSE(diagnostics.aec_compiled);
  REQUIRE_FALSE(diagnostics.aec_active);
  REQUIRE(diagnostics.degraded);
  REQUIRE(diagnostics.state == dvo::AecRuntimeState::degraded_backend_unavailable);
  REQUIRE(std::string_view(dvo::to_string(diagnostics.state)) ==
          "degraded_backend_unavailable");
  REQUIRE(diagnostics.stream_delay_ms == 17);
  REQUIRE(diagnostics.fallback_frames == 1);
}

TEST_CASE("AEC delay hint is not distorted by packet delivery backlog") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  timeline.delay_offset_ms = 23;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);

  auto render = timeline_packet(dvo::AudioStreamKind::loopback, 4'400'000,
                                0, 1, 0.1F);
  auto microphone = timeline_packet(dvo::AudioStreamKind::microphone,
                                    4'400'000, 0, 1, 0.2F);
  render.arrival_qpc_100ns = render.qpc_100ns + 5'000;
  microphone.arrival_qpc_100ns = microphone.qpc_100ns + 905'000;
  REQUIRE(preprocessor.PushPacket(render).accepted);
  REQUIRE(preprocessor.PushPacket(microphone).accepted);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(preprocessor.Diagnostics().stream_delay_ms == 23);
}

TEST_CASE("AEC automatically estimates a repeatable render to microphone delay") {
  constexpr std::size_t frame_count = 260;
  constexpr std::size_t delay_samples = 80 * dvo::kProcessingSampleRate / 1000;
  std::vector<float> render(frame_count * dvo::kFrameSamples);
  std::vector<float> microphone(render.size(), 0.0F);
  std::uint32_t random_state = 0x12345678U;
  for (auto& sample : render) {
    random_state ^= random_state << 13U;
    random_state ^= random_state >> 17U;
    random_state ^= random_state << 5U;
    sample = (static_cast<float>(random_state & 0xffffU) / 32767.5F - 1.0F) *
             0.25F;
  }
  for (std::size_t i = delay_samples; i < microphone.size(); ++i) {
    microphone[i] = render[i - delay_samples] * 0.7F;
  }

  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  timeline.auto_delay_enabled = true;
  timeline.auto_delay_min_ms = 20;
  timeline.auto_delay_max_ms = 150;
  timeline.auto_delay_window_ms = 500;
  timeline.auto_delay_update_ms = 100;
  timeline.auto_delay_min_correlation = 0.35F;
  timeline.delay_offset_ms = 7;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);

  for (std::uint64_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    REQUIRE(preprocessor.PushPacket(mono_timeline_packet(
                dvo::AudioStreamKind::loopback, frame_index, render))
                .accepted);
    REQUIRE(preprocessor.PushPacket(mono_timeline_packet(
                dvo::AudioStreamKind::microphone, frame_index, microphone))
                .accepted);
    dvo::NormalizedFrame output;
    while (preprocessor.TryPopFrame(output)) {
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.auto_delay_enabled);
  REQUIRE(diagnostics.auto_delay_available);
  REQUIRE(diagnostics.auto_delay_updates > 1);
  REQUIRE(diagnostics.auto_delay_confidence > 0.9);
  REQUIRE(diagnostics.auto_delay_ms == Catch::Approx(80.0).margin(1.0));
  REQUIRE(diagnostics.stream_delay_ms == 87);
}

TEST_CASE("compiled WebRTC backend continuously processes aligned frames") {
  if (!dvo::webrtc_aec3_compiled()) return;

  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);

  std::size_t output_frames{};
  for (std::uint64_t i = 0; i < 30; ++i) {
    auto render = timeline_packet(dvo::AudioStreamKind::loopback,
                                  4'500'000 + i * 100'000, i * 480,
                                  i + 1, (i % 2 == 0) ? 0.15F : -0.15F);
    auto microphone = timeline_packet(dvo::AudioStreamKind::microphone,
                                      4'500'000 + i * 100'000, i * 480,
                                      i + 1, (i % 2 == 0) ? 0.2F : -0.2F);
    REQUIRE(preprocessor.PushPacket(render).accepted);
    REQUIRE(preprocessor.PushPacket(microphone).accepted);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
      ++output_frames;
      for (const auto sample : frame.samples) REQUIRE(std::isfinite(sample));
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(output_frames == 30);
  REQUIRE(diagnostics.aec_compiled);
  REQUIRE(diagnostics.aec_active);
  REQUIRE_FALSE(diagnostics.degraded);
  REQUIRE(diagnostics.state == dvo::AecRuntimeState::active);
  REQUIRE(diagnostics.aec_frames == output_frames);
  REQUIRE(diagnostics.fallback_frames == 0);
  REQUIRE(diagnostics.processing_measurements == output_frames);
  REQUIRE(diagnostics.max_processing_time_us > 0.0);
}

TEST_CASE("synthetic idle render transitions to real render without a timeline reset") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);

  auto idle = timeline_packet(dvo::AudioStreamKind::loopback, 4'800'000, 0, 1,
                              0.0F);
  idle.synthetic = true;
  idle.silent = true;
  REQUIRE(preprocessor.PushPacket(idle).accepted);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 4'800'000, 0, 1, 0.2F))
              .accepted);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  auto idle_diagnostics = preprocessor.Diagnostics();
  REQUIRE(idle_diagnostics.render_available);
  REQUIRE(idle_diagnostics.render_synthetic);
  if (dvo::webrtc_aec3_compiled()) {
    REQUIRE(idle_diagnostics.aec_active);
    REQUIRE_FALSE(idle_diagnostics.degraded);
  }

  const auto real = timeline_packet(dvo::AudioStreamKind::loopback, 4'900'000,
                                    480, 2, 0.15F);
  const auto real_push = preprocessor.PushPacket(real);
  REQUIRE(real_push.accepted);
  REQUIRE_FALSE(real_push.reset_required);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 4'900'000, 480, 2, 0.25F))
              .accepted);
  REQUIRE(preprocessor.TryPopFrame(frame));

  const auto real_diagnostics = preprocessor.Diagnostics();
  REQUIRE(real_diagnostics.render_available);
  REQUIRE_FALSE(real_diagnostics.render_synthetic);
  REQUIRE(real_diagnostics.render_packets == 2);
  REQUIRE(real_diagnostics.last_reset_reason == dvo::PreprocessResetReason::none);
  if (dvo::webrtc_aec3_compiled()) {
    REQUIRE(real_diagnostics.aec_active);
    REQUIRE_FALSE(real_diagnostics.degraded);
  }
}

TEST_CASE("late real render replaces an unconsumed synthetic idle interval") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);

  auto idle = timeline_packet(dvo::AudioStreamKind::loopback, 5'200'000, 0, 1,
                              0.0F);
  idle.synthetic = true;
  idle.silent = true;
  REQUIRE(preprocessor.PushPacket(idle).accepted);

  // Model a real WASAPI event that was delivered after the idle scheduler had
  // provisionally covered the same timestamp range.
  const auto real = timeline_packet(dvo::AudioStreamKind::loopback, 5'200'000,
                                    0, 2, 0.35F);
  const auto real_push = preprocessor.PushPacket(real);
  REQUIRE(real_push.accepted);
  REQUIRE_FALSE(real_push.reset_required);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 5'200'000, 0, 1, 0.2F))
              .accepted);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.render_available);
  REQUIRE_FALSE(diagnostics.render_synthetic);
  REQUIRE(diagnostics.render_synthetic_samples_replaced == dvo::kFrameSamples);
  REQUIRE(diagnostics.last_reset_reason == dvo::PreprocessResetReason::none);
  if (dvo::webrtc_aec3_compiled()) {
    REQUIRE(diagnostics.aec_active);
    REQUIRE_FALSE(diagnostics.degraded);
  }
}

TEST_CASE("late real render hard resync discards provisional replacement records") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::WebRtcAec3Preprocessor preprocessor(config, timeline);
  constexpr std::uint64_t base_qpc = 5'600'000;

  // Build more than the default 80 ms hard-resync window of provisional
  // loopback audio, then deliver the real packet for the first interval.
  for (std::uint64_t i = 0; i < 10; ++i) {
    auto idle = timeline_packet(dvo::AudioStreamKind::loopback,
                                base_qpc + i * 100'000, i * 480, i + 1,
                                0.0F);
    idle.synthetic = true;
    idle.silent = true;
    REQUIRE(preprocessor.PushPacket(idle).accepted);
  }

  const auto real = timeline_packet(dvo::AudioStreamKind::loopback, base_qpc,
                                    0, 11, 0.35F);
  const auto pushed = preprocessor.PushPacket(real);
  REQUIRE(pushed.accepted);
  REQUIRE(pushed.reset_required);
  CHECK(pushed.reset_reason == dvo::PreprocessResetReason::hard_resync);
  CHECK(preprocessor.Diagnostics().hard_resyncs == 1);
}

TEST_CASE("timestamp errors reset state and preserve the output sample clock") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 5'000'000, 0, 1, 0.1F))
              .accepted);
  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));

  auto bad = timeline_packet(dvo::AudioStreamKind::microphone, 5'100'000, 480, 2,
                             0.2F);
  bad.timestamp_error = true;
  const auto rejected = preprocessor.PushPacket(bad);
  REQUIRE_FALSE(rejected.accepted);
  REQUIRE(rejected.reset_required);
  REQUIRE(rejected.reset_reason == dvo::PreprocessResetReason::timestamp_error);

  const auto resumed = timeline_packet(dvo::AudioStreamKind::microphone, 7'000'000, 0,
                                       1, 0.3F);
  REQUIRE(preprocessor.PushPacket(resumed).accepted);
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.first_sample == dvo::kFrameSamples);
  REQUIRE(frame.discontinuity);
  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.timestamp_errors == 1);
  REQUIRE(diagnostics.last_reset_reason ==
          dvo::PreprocessResetReason::timestamp_error);
}

TEST_CASE("large timeline errors hard resynchronize without reusing sample indices") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 8'000'000, 0, 1, 0.1F))
              .accepted);
  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));

  const auto jumped = timeline_packet(dvo::AudioStreamKind::microphone, 13'000'000,
                                      480, 2, 0.2F);
  const auto result = preprocessor.PushPacket(jumped);
  REQUIRE(result.accepted);
  REQUIRE(result.reset_required);
  REQUIRE(result.reset_reason == dvo::PreprocessResetReason::hard_resync);
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.first_sample == dvo::kFrameSamples);
  REQUIRE(frame.qpc_100ns == 13'000'000);
  REQUIRE(frame.discontinuity);
  REQUIRE(preprocessor.Diagnostics().hard_resyncs == 1);
}

TEST_CASE("independent device clocks produce a bounded drift estimate") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.drift_window_ms = 1000;
  dvo::BypassPreprocessor preprocessor(config, timeline);
  constexpr std::uint64_t base_qpc = 20'000'000;
  constexpr std::uint64_t microphone_step = 100'000;
  constexpr std::uint64_t render_step = 99'950;  // approximately +500 ppm

  for (std::uint64_t i = 0; i < 125; ++i) {
    REQUIRE(preprocessor.PushPacket(timeline_packet(
                dvo::AudioStreamKind::loopback, base_qpc + i * render_step,
                i * 480, i + 1, -0.1F))
                .accepted);
    REQUIRE(preprocessor.PushPacket(timeline_packet(
                dvo::AudioStreamKind::microphone, base_qpc + i * microphone_step,
                i * 480, i + 1, 0.1F))
                .accepted);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.drift_estimate_valid);
  REQUIRE(diagnostics.microphone_rate_hz == Catch::Approx(48'000.0).margin(1.0));
  REQUIRE(diagnostics.render_rate_hz > 48'015.0);
  REQUIRE(diagnostics.render_rate_hz < 48'035.0);
  REQUIRE(diagnostics.relative_drift_ppm > 300.0);
  REQUIRE(diagnostics.relative_drift_ppm < 700.0);
  REQUIRE(diagnostics.speexdsp_compiled == dvo::speexdsp_compiled());
  if (dvo::speexdsp_compiled()) {
    REQUIRE(diagnostics.microphone_resampler_speex);
    REQUIRE(diagnostics.render_resampler_speex);
    REQUIRE(diagnostics.render_resampler_rate_updates >= 2);
    REQUIRE(diagnostics.resampler_failures == 0);
  } else {
    REQUIRE_FALSE(diagnostics.microphone_resampler_speex);
    REQUIRE_FALSE(diagnostics.render_resampler_speex);
    REQUIRE(diagnostics.render_resampler_rate_updates == 0);
    REQUIRE(diagnostics.drift_correction_samples > 0);
  }
}

TEST_CASE("different nominal device rates do not masquerade as clock drift") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.drift_window_ms = 1000;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::BypassPreprocessor preprocessor(config, timeline);
  constexpr std::uint64_t base_qpc = 22'000'000;

  for (std::uint64_t i = 0; i < 125; ++i) {
    REQUIRE(preprocessor.PushPacket(rate_timeline_packet(
                dvo::AudioStreamKind::loopback, base_qpc + i * 100'000,
                i * 480, i + 1, 48'000, 480, -0.1F))
                .accepted);
    REQUIRE(preprocessor.PushPacket(rate_timeline_packet(
                dvo::AudioStreamKind::microphone, base_qpc + i * 100'000,
                i * 160, i + 1, 16'000, 160, 0.1F))
                .accepted);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.drift_estimate_valid);
  REQUIRE(diagnostics.microphone_rate_hz == Catch::Approx(16'000.0).margin(1.0));
  REQUIRE(diagnostics.render_rate_hz == Catch::Approx(48'000.0).margin(1.0));
  REQUIRE(std::abs(diagnostics.relative_drift_ppm) < 10.0);
  REQUIRE_FALSE(diagnostics.drift_out_of_range);
}

TEST_CASE("Speex fractional rate updates stay bounded for live clock estimates") {
  if (!dvo::speexdsp_compiled()) return;

  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.drift_window_ms = 1000;
  timeline.alignment_wait_ms = 0;
  timeline.target_render_buffer_ms = 10;
  dvo::BypassPreprocessor preprocessor(config, timeline);
  constexpr std::uint64_t base_qpc = 25'000'000;

  for (std::uint64_t i = 0; i < 250; ++i) {
    const auto render_qpc = base_qpc + static_cast<std::uint64_t>(
        std::llround(static_cast<long double>(i) * 99'993.48L));
    REQUIRE(preprocessor.PushPacket(timeline_packet(
                dvo::AudioStreamKind::loopback, render_qpc, i * 480,
                i + 1, -0.1F))
                .accepted);
    REQUIRE(preprocessor.PushPacket(timeline_packet(
                dvo::AudioStreamKind::microphone, base_qpc + i * 100'000,
                i * 480, i + 1, 0.1F))
                .accepted);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.render_resampler_speex);
  REQUIRE(diagnostics.render_resampler_rate_updates >= 2);
  REQUIRE(diagnostics.resampler_failures == 0);
  REQUIRE(diagnostics.last_resampler_error_code == 0);
}

TEST_CASE("relative drift normalizes independent nominal sample rates") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.drift_window_ms = 1000;
  dvo::BypassPreprocessor preprocessor(config, timeline);
  constexpr std::uint64_t base_qpc = 25'000'000;

  for (std::uint64_t i = 0; i < 125; ++i) {
    auto render = timeline_packet(dvo::AudioStreamKind::loopback,
                                  base_qpc + i * 100'000, i * 441,
                                  i + 1, -0.1F);
    render.format.sample_rate = 44'100;
    render.samples.assign(441 * 2, -0.1F);
    REQUIRE(preprocessor.PushPacket(render).accepted);

    auto microphone = timeline_packet(dvo::AudioStreamKind::microphone,
                                      base_qpc + i * 100'000, i * 480,
                                      i + 1, 0.1F);
    REQUIRE(preprocessor.PushPacket(microphone).accepted);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
    }
  }

  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.drift_estimate_valid);
  CHECK(diagnostics.microphone_rate_hz == Catch::Approx(48'000.0).margin(1.0));
  CHECK(diagnostics.render_rate_hz == Catch::Approx(44'100.0).margin(1.0));
  CHECK(diagnostics.relative_drift_ppm == Catch::Approx(0.0).margin(1.0));
  CHECK_FALSE(diagnostics.drift_out_of_range);
}

TEST_CASE("continuous device positions reject packet QPC scheduling jitter") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  constexpr std::uint64_t base_qpc = 30'000'000;
  constexpr std::array<std::int64_t, 6> jitter{
      0, 7'500, -7'500, 6'000, -6'000, 0};

  std::size_t output_frames{};
  for (std::uint64_t i = 0; i < jitter.size(); ++i) {
    const auto qpc = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(base_qpc + i * 100'000) + jitter[i]);
    auto packet = timeline_packet(dvo::AudioStreamKind::microphone, qpc,
                                  10'000 + i * 480, i + 1, 0.1F);
    const auto pushed = preprocessor.PushPacket(packet);
    REQUIRE(pushed.accepted);
    REQUIRE_FALSE(pushed.reset_required);
    dvo::NormalizedFrame frame;
    while (preprocessor.TryPopFrame(frame)) {
      REQUIRE_FALSE(frame.discontinuity);
      ++output_frames;
    }
  }
  REQUIRE(output_frames == jitter.size());
}

TEST_CASE("output buffering is finite and reports dropped frames") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.max_buffer_ms = 100;
  dvo::BypassPreprocessor preprocessor(config, timeline);

  for (std::uint64_t i = 0; i < 30; ++i) {
    REQUIRE(preprocessor.PushPacket(timeline_packet(
                dvo::AudioStreamKind::microphone, 40'000'000 + i * 100'000,
                i * 480, i + 1, 0.1F))
                .accepted);
  }
  const auto diagnostics = preprocessor.Diagnostics();
  REQUIRE(diagnostics.output_frames_ready == 10);
  REQUIRE(diagnostics.output_frames_dropped == 20);
}

TEST_CASE("oversized microphone packets retain a bounded tail and signal reset") {
  dvo::AudioConfig config;
  dvo::PreprocessorTimelineConfig timeline;
  timeline.max_buffer_ms = 100;
  dvo::BypassPreprocessor preprocessor(config, timeline);

  auto oversized = timeline_packet(dvo::AudioStreamKind::microphone, 50'000'000,
                                   0, 1, 0.2F);
  oversized.samples.assign(480 * 20 * 2, 0.2F);
  const auto result = preprocessor.PushPacket(oversized);
  REQUIRE(result.accepted);
  REQUIRE(result.reset_required);
  REQUIRE(result.reset_reason == dvo::PreprocessResetReason::buffer_overflow);

  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.first_sample == 0);
  REQUIRE(frame.qpc_100ns == 51'000'000);
  REQUIRE(frame.discontinuity);
  REQUIRE(preprocessor.Diagnostics().buffer_overflows == 1);
}

TEST_CASE("packet sequence and stream epoch changes are explicit reset boundaries") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  REQUIRE(preprocessor.PushPacket(timeline_packet(
              dvo::AudioStreamKind::microphone, 60'000'000, 0, 1, 0.1F))
              .accepted);
  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));

  SECTION("sequence gap") {
    const auto result = preprocessor.PushPacket(timeline_packet(
        dvo::AudioStreamKind::microphone, 60'100'000, 480, 3, 0.2F));
    REQUIRE(result.accepted);
    REQUIRE(result.reset_required);
    REQUIRE(result.reset_reason == dvo::PreprocessResetReason::sequence_gap);
  }

  SECTION("stream epoch") {
    auto reopened = timeline_packet(dvo::AudioStreamKind::microphone, 60'100'000,
                                    480, 2, 0.2F);
    reopened.stream_epoch = 2;
    const auto result = preprocessor.PushPacket(reopened);
    REQUIRE(result.accepted);
    REQUIRE(result.reset_required);
    REQUIRE(result.reset_reason == dvo::PreprocessResetReason::stream_epoch_change);
  }

  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.first_sample == dvo::kFrameSamples);
  REQUIRE(frame.discontinuity);
}

TEST_CASE("silent packet flags override untrusted sample contents") {
  dvo::AudioConfig config;
  dvo::BypassPreprocessor preprocessor(config);
  auto packet = timeline_packet(dvo::AudioStreamKind::microphone, 70'000'000,
                                0, 1, 0.9F);
  packet.silent = true;
  REQUIRE(preprocessor.PushPacket(packet).accepted);
  dvo::NormalizedFrame frame;
  REQUIRE(preprocessor.TryPopFrame(frame));
  REQUIRE(frame.samples[80] == Catch::Approx(0.0F));
}
