#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/config.h"

#ifndef DVO_HAS_WEBRTC_APM
#define DVO_HAS_WEBRTC_APM 0
#endif

#ifndef DVO_HAS_SPEEXDSP
#define DVO_HAS_SPEEXDSP 0
#endif

namespace dvo {

enum class PreprocessResetReason {
  none,
  explicit_request,
  invalid_packet,
  discontinuity,
  timestamp_error,
  format_change,
  stream_epoch_change,
  sequence_gap,
  hard_resync,
  buffer_overflow,
  backend_error,
  resampler_error,
};

[[nodiscard]] const char* to_string(PreprocessResetReason reason) noexcept;

enum class AecRuntimeState {
  bypass,
  waiting_for_render,
  active,
  degraded_backend_unavailable,
  degraded_missing_render,
  degraded_drift,
  backend_error,
};

[[nodiscard]] const char* to_string(AecRuntimeState state) noexcept;

// Timeline/alignment parameters deliberately live next to the preprocessor
// interface for now. They can be moved into AppConfig without coupling the
// audio core to the configuration/UI migration.
struct PreprocessorTimelineConfig {
  std::uint32_t alignment_wait_ms{30};
  std::uint32_t target_render_buffer_ms{60};
  std::uint32_t max_buffer_ms{500};
  std::uint32_t drift_window_ms{5000};
  std::uint32_t hard_resync_error_ms{80};
  double max_drift_ppm{1000.0};
  std::int32_t delay_offset_ms{};
  bool high_pass_filter{true};
  bool noise_suppression{};
  bool gain_control{};
  std::uint32_t stats_hz{1};
};

[[nodiscard]] PreprocessorTimelineConfig make_preprocessor_timeline_config(
    const AecConfig& config);

struct PreprocessPushResult {
  bool accepted{};
  bool reset_required{};
  PreprocessResetReason reset_reason{PreprocessResetReason::none};
  std::size_t frames_ready{};
};

struct PreprocessDiagnostics {
  AecRuntimeState state{AecRuntimeState::bypass};
  bool aec_requested{};
  bool aec_compiled{};
  bool aec_active{};
  bool degraded{};
  bool render_seen{};
  bool render_available{};
  bool render_synthetic{};
  bool drift_out_of_range{};
  bool drift_estimate_valid{};
  bool speexdsp_compiled{};
  bool microphone_resampler_speex{};
  bool render_resampler_speex{};

  double microphone_rate_hz{};
  double render_rate_hz{};
  double relative_drift_ppm{};
  int stream_delay_ms{};

  std::size_t microphone_buffered_samples{};
  std::size_t render_buffered_samples{};
  std::size_t target_render_buffer_samples{};
  std::int64_t render_buffer_error_samples{};
  std::size_t output_frames_ready{};

  std::uint64_t microphone_packets{};
  std::uint64_t render_packets{};
  std::uint64_t output_frames{};
  std::uint64_t aec_frames{};
  std::uint64_t fallback_frames{};
  std::uint64_t resets{};
  std::uint64_t timestamp_errors{};
  std::uint64_t hard_resyncs{};
  std::uint64_t drift_correction_samples{};
  std::uint64_t buffer_overflows{};
  std::uint64_t output_frames_dropped{};
  std::uint64_t resampler_failures{};
  std::uint64_t render_resampler_rate_updates{};
  std::uint64_t render_synthetic_samples_replaced{};
  std::uint64_t processing_measurements{};
  double last_processing_time_us{};
  double average_processing_time_us{};
  double max_processing_time_us{};
  PreprocessResetReason last_reset_reason{PreprocessResetReason::none};

  std::optional<double> echo_return_loss_db;
  std::optional<double> echo_return_loss_enhancement_db;
  std::optional<double> residual_echo_likelihood;
  std::optional<double> divergent_filter_fraction;
  std::optional<int> estimated_delay_ms;
};

// Kept as a compatibility result while VoiceFrontendRuntime migrates from the
// old process(mic, latest_render) call. New code should use PushPacket and
// TryPopFrame so every render packet is delivered exactly once.
struct PreprocessResult {
  std::vector<NormalizedFrame> frames;
  bool reset_required{};
  bool render_available{};
  bool degraded{};
};

class IAudioPreprocessor {
 public:
  virtual ~IAudioPreprocessor() = default;

  // Both microphone and loopback packets are submitted in monotonic order per
  // stream. The implementation owns bounded per-stream FIFOs and emits capture
  // frames only when their 10 ms timeline interval is ready.
  virtual PreprocessPushResult PushPacket(const AudioPacket& packet) = 0;
  virtual bool TryPopFrame(NormalizedFrame& frame) = 0;
  virtual void Reset(
      PreprocessResetReason reason = PreprocessResetReason::explicit_request) = 0;
  [[nodiscard]] virtual PreprocessDiagnostics Diagnostics() const = 0;

  // Transitional wrappers. They preserve the phase-1 ABI at source level but
  // cannot provide correct AEC if callers repeatedly pass only the latest
  // render packet. Remove after runtime has migrated to PushPacket.
  PreprocessResult process(const AudioPacket& microphone,
                           const std::optional<AudioPacket>& render);
  void reset() { Reset(); }
};

class BypassPreprocessor final : public IAudioPreprocessor {
 public:
  explicit BypassPreprocessor(
      const AudioConfig& config,
      PreprocessorTimelineConfig timeline = PreprocessorTimelineConfig{});
  ~BypassPreprocessor() override;

  PreprocessPushResult PushPacket(const AudioPacket& packet) override;
  bool TryPopFrame(NormalizedFrame& frame) override;
  void Reset(PreprocessResetReason reason) override;
  [[nodiscard]] PreprocessDiagnostics Diagnostics() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// When DVO_HAS_WEBRTC_APM=1 this class owns an official WebRTC APM/AEC3
// instance. With the dependency disabled it remains constructible and emits a
// clearly diagnosed degraded bypass stream, allowing core-only builds and
// deterministic tests to use the same interface.
class WebRtcAec3Preprocessor final : public IAudioPreprocessor {
 public:
  explicit WebRtcAec3Preprocessor(
      const AudioConfig& config,
      PreprocessorTimelineConfig timeline = PreprocessorTimelineConfig{});
  ~WebRtcAec3Preprocessor() override;

  PreprocessPushResult PushPacket(const AudioPacket& packet) override;
  bool TryPopFrame(NormalizedFrame& frame) override;
  void Reset(PreprocessResetReason reason) override;
  [[nodiscard]] PreprocessDiagnostics Diagnostics() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] constexpr bool webrtc_aec3_compiled() noexcept {
  return DVO_HAS_WEBRTC_APM != 0;
}

[[nodiscard]] constexpr bool speexdsp_compiled() noexcept {
  return DVO_HAS_SPEEXDSP != 0;
}

}  // namespace dvo
