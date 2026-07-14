#include "dvo/preprocessor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if DVO_HAS_WEBRTC_APM
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/environment/environment_factory.h"
#include "api/scoped_refptr.h"
#endif

#if DVO_HAS_SPEEXDSP
#include <speex/speex_resampler.h>
#endif

namespace dvo {
namespace {

constexpr std::uint64_t kQpcUnitsPerSecond = 10'000'000ULL;
constexpr std::int64_t kQpcUnitsPerOutputSample =
    static_cast<std::int64_t>(kQpcUnitsPerSecond / kProcessingSampleRate);
static_assert(kQpcUnitsPerSecond % kProcessingSampleRate == 0);

enum SampleFlags : std::uint8_t {
  kRealSample = 1U << 0U,
  kSyntheticSample = 1U << 1U,
  kGapSample = 1U << 2U,
};

struct BufferAppendResult {
  std::size_t correction_samples{};
  std::size_t overflow_samples{};
  std::size_t replaced_synthetic_samples{};
  bool contains_gap{};
};

struct BufferExtractResult {
  bool complete{};
  bool contains_gap{};
  bool contains_real{};
  bool contains_synthetic{};
};

class TimedSampleBuffer {
 public:
  BufferAppendResult append(std::int64_t start, const std::vector<float>& values,
                            std::uint8_t flags, std::size_t maximum_samples) {
    BufferAppendResult result;
    if (values.empty()) return result;

    std::size_t skip{};
    const auto values_end = start + static_cast<std::int64_t>(values.size());
    if (values_end <= floor_) return result;
    if (start < floor_) {
      skip = static_cast<std::size_t>(floor_ - start);
      start = floor_;
      result.correction_samples += skip;
    }

    if (samples_.empty()) {
      start_ = start;
    } else {
      if (start < start_) {
        const auto older = std::min(
            values.size() - std::min(skip, values.size()),
            static_cast<std::size_t>(start_ - start));
        skip += older;
        start += static_cast<std::int64_t>(older);
        result.correction_samples += older;
        if (skip >= values.size()) return result;
      }
      const auto current_end = end();
      if (start > current_end) {
        const auto gap = static_cast<std::size_t>(start - current_end);
        result.correction_samples += gap;
        if (gap <= 4) {
          // One-sample insertions are the normal manifestation of correcting
          // two independent hardware clocks. Hold the last sample instead of
          // creating an audible zero spike.
          const float held = samples_.empty() ? 0.0F : samples_.back();
          const auto held_flags = sample_flags_.empty()
                                      ? static_cast<std::uint8_t>(kGapSample)
                                      : sample_flags_.back();
          samples_.insert(samples_.end(), gap, held);
          sample_flags_.insert(sample_flags_.end(), gap, held_flags);
        } else {
          samples_.insert(samples_.end(), gap, 0.0F);
          sample_flags_.insert(sample_flags_.end(), gap, kGapSample);
          result.contains_gap = true;
        }
      } else if (start < current_end) {
        const auto remaining = values.size() - std::min(skip, values.size());
        const auto overlap = std::min(
            static_cast<std::size_t>(current_end - start), remaining);
        // Idle loopback is deliberately emitted behind real time, but a
        // delayed WASAPI event can still overlap a synthetic interval that has
        // not yet been consumed. Real render owns that interval: replace the
        // provisional zeros in place. Synthetic packets arriving later never
        // overwrite real render.
        if ((flags & kRealSample) != 0) {
          const auto buffer_offset = static_cast<std::size_t>(start - start_);
          for (std::size_t i = 0; i < overlap; ++i) {
            const auto existing_flags = sample_flags_[buffer_offset + i];
            if ((existing_flags & kRealSample) == 0 &&
                (existing_flags & (kSyntheticSample | kGapSample)) != 0) {
              samples_[buffer_offset + i] = values[skip + i];
              sample_flags_[buffer_offset + i] = flags;
              ++result.replaced_synthetic_samples;
            }
          }
        }
        const auto extra_skip = overlap;
        skip += extra_skip;
        start += static_cast<std::int64_t>(extra_skip);
        result.correction_samples += extra_skip;
      }
    }

    if (skip < values.size()) {
      samples_.insert(samples_.end(), values.begin() + static_cast<std::ptrdiff_t>(skip),
                      values.end());
      sample_flags_.insert(sample_flags_.end(), values.size() - skip, flags);
    }

    if (samples_.size() > maximum_samples) {
      const auto remove = samples_.size() - maximum_samples;
      pop_front(remove);
      result.overflow_samples = remove;
    }
    return result;
  }

  [[nodiscard]] bool has_range(std::int64_t start, std::size_t count) const {
    if (samples_.empty() || start < start_) return false;
    return start + static_cast<std::int64_t>(count) <= end();
  }

  BufferExtractResult extract(std::int64_t start, std::span<float> output) const {
    BufferExtractResult result;
    if (!has_range(start, output.size())) return result;
    result.complete = true;
    const auto offset = static_cast<std::size_t>(start - start_);
    for (std::size_t i = 0; i < output.size(); ++i) {
      output[i] = samples_[offset + i];
      const auto flags = sample_flags_[offset + i];
      result.contains_gap |= (flags & kGapSample) != 0;
      result.contains_real |= (flags & kRealSample) != 0;
      result.contains_synthetic |= (flags & kSyntheticSample) != 0;
    }
    return result;
  }

  void discard_before(std::int64_t index) {
    floor_ = std::max(floor_, index);
    if (samples_.empty() || index <= start_) return;
    const auto remove = std::min(
        samples_.size(), static_cast<std::size_t>(std::max<std::int64_t>(0, index - start_)));
    pop_front(remove);
    if (samples_.empty()) start_ = index;
  }

  void clear() {
    samples_.clear();
    sample_flags_.clear();
    start_ = 0;
    floor_ = std::numeric_limits<std::int64_t>::min() / 4;
  }

  [[nodiscard]] bool empty() const { return samples_.empty(); }
  [[nodiscard]] std::size_t size() const { return samples_.size(); }
  [[nodiscard]] std::int64_t start() const { return start_; }
  [[nodiscard]] std::int64_t end() const {
    return start_ + static_cast<std::int64_t>(samples_.size());
  }

 private:
  void pop_front(std::size_t count) {
    count = std::min(count, samples_.size());
    for (std::size_t i = 0; i < count; ++i) {
      samples_.pop_front();
      sample_flags_.pop_front();
    }
    start_ += static_cast<std::int64_t>(count);
  }

  std::deque<float> samples_;
  std::deque<std::uint8_t> sample_flags_;
  std::int64_t start_{};
  std::int64_t floor_{std::numeric_limits<std::int64_t>::min() / 4};
};

class ClockRateEstimator {
 public:
  enum class Update { accepted, ignored, backwards };

  Update update(const AudioPacket& packet, std::uint64_t window_100ns) {
    nominal_rate_hz_ = static_cast<double>(packet.format.sample_rate);
    if (packet.synthetic || packet.timestamp_error) return Update::ignored;
    if (!anchor_valid_) {
      set_anchor(packet);
      return Update::accepted;
    }
    if (packet.qpc_100ns <= anchor_qpc_ || packet.device_position < anchor_position_) {
      set_anchor(packet);
      rate_valid_ = false;
      return Update::backwards;
    }

    const auto qpc_delta = packet.qpc_100ns - anchor_qpc_;
    if (qpc_delta < window_100ns) return Update::accepted;
    const auto position_delta = packet.device_position - anchor_position_;
    const double measured = static_cast<double>(position_delta) *
                            static_cast<double>(kQpcUnitsPerSecond) /
                            static_cast<double>(qpc_delta);
    set_anchor(packet);
    if (measured < nominal_rate_hz_ * 0.95 || measured > nominal_rate_hz_ * 1.05) {
      rate_valid_ = false;
      return Update::ignored;
    }
    estimated_rate_hz_ = rate_valid_ ? estimated_rate_hz_ * 0.8 + measured * 0.2 : measured;
    rate_valid_ = true;
    return Update::accepted;
  }

  void reset() {
    anchor_valid_ = false;
    rate_valid_ = false;
    nominal_rate_hz_ = 0.0;
    estimated_rate_hz_ = 0.0;
  }

  [[nodiscard]] bool valid() const { return rate_valid_; }
  [[nodiscard]] double rate_hz() const {
    return rate_valid_ ? estimated_rate_hz_ : nominal_rate_hz_;
  }

 private:
  void set_anchor(const AudioPacket& packet) {
    anchor_qpc_ = packet.qpc_100ns;
    anchor_position_ = packet.device_position;
    anchor_valid_ = true;
  }

  std::uint64_t anchor_qpc_{};
  std::uint64_t anchor_position_{};
  bool anchor_valid_{};
  bool rate_valid_{};
  double nominal_rate_hz_{};
  double estimated_rate_hz_{};
};

struct BackendMetrics {
  std::optional<double> echo_return_loss_db;
  std::optional<double> echo_return_loss_enhancement_db;
  std::optional<double> residual_echo_likelihood;
  std::optional<double> divergent_filter_fraction;
  std::optional<int> estimated_delay_ms;
};

class AecBackend {
 public:
  virtual ~AecBackend() = default;
  virtual bool process(const std::array<float, kFrameSamples>& render,
                       const std::array<float, kFrameSamples>& capture,
                       int stream_delay_ms,
                       std::array<float, kFrameSamples>& output) = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual BackendMetrics metrics() const = 0;
};

#if DVO_HAS_WEBRTC_APM
class WebRtcAecBackend final : public AecBackend {
 public:
  explicit WebRtcAecBackend(const PreprocessorTimelineConfig& settings) {
    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = true;
    config.high_pass_filter.enabled = settings.high_pass_filter;
    config.noise_suppression.enabled = settings.noise_suppression;
    // GainController2 is digital-only and therefore does not require a
    // platform microphone-level feedback loop.
    config.gain_controller2.enabled = settings.gain_control;
    auto builder = webrtc::BuiltinAudioProcessingBuilder(config);
    // With echo_canceller.enabled, AudioProcessingImpl creates its built-in
    // EchoCanceller3 when no custom factory is injected. Keeping the default
    // path also matches the symbols exported by vcpkg's monolithic WebRTC
    // library; api/audio:aec3_factory is a separate public GN target.
    apm_ = builder.Build(webrtc::CreateEnvironment());
    if (!apm_) throw std::runtime_error("WebRTC AudioProcessing creation failed");
    stats_interval_frames_ =
        std::max<std::uint32_t>(1, 100 / std::max<std::uint32_t>(1, settings.stats_hz));
  }

  bool process(const std::array<float, kFrameSamples>& render,
               const std::array<float, kFrameSamples>& capture,
               int stream_delay_ms,
               std::array<float, kFrameSamples>& output) override {
    std::array<float, kFrameSamples> render_output{};
    const float* render_source[] = {render.data()};
    float* render_destination[] = {render_output.data()};
    if (apm_->ProcessReverseStream(render_source, stream_config_, stream_config_,
                                   render_destination) != webrtc::AudioProcessing::kNoError) {
      return false;
    }
    const auto delay_result = apm_->set_stream_delay_ms(stream_delay_ms);
    if (delay_result != webrtc::AudioProcessing::kNoError &&
        delay_result != webrtc::AudioProcessing::kBadStreamParameterWarning) {
      return false;
    }
    const float* capture_source[] = {capture.data()};
    float* capture_destination[] = {output.data()};
    if (apm_->ProcessStream(capture_source, stream_config_, stream_config_,
                            capture_destination) != webrtc::AudioProcessing::kNoError) {
      return false;
    }
    if (++frames_since_stats_ >= stats_interval_frames_) {
      frames_since_stats_ = 0;
      const auto stats = apm_->GetStatistics();
      metrics_.echo_return_loss_db = copy_optional(stats.echo_return_loss);
      metrics_.echo_return_loss_enhancement_db =
          copy_optional(stats.echo_return_loss_enhancement);
      metrics_.residual_echo_likelihood = copy_optional(stats.residual_echo_likelihood);
      metrics_.divergent_filter_fraction = copy_optional(stats.divergent_filter_fraction);
      if (stats.delay_ms) metrics_.estimated_delay_ms = static_cast<int>(*stats.delay_ms);
    }
    return true;
  }

  void reset() override {
    if (apm_) static_cast<void>(apm_->Initialize());
    frames_since_stats_ = 0;
    metrics_ = {};
  }

  [[nodiscard]] BackendMetrics metrics() const override { return metrics_; }

 private:
  template <typename Optional>
  static std::optional<double> copy_optional(const Optional& value) {
    return value ? std::optional<double>(static_cast<double>(*value)) : std::nullopt;
  }

  webrtc::scoped_refptr<webrtc::AudioProcessing> apm_;
  webrtc::StreamConfig stream_config_{static_cast<int>(kProcessingSampleRate), 1};
  std::uint32_t frames_since_stats_{};
  std::uint32_t stats_interval_frames_{100};
  BackendMetrics metrics_;
};
#endif

std::unique_ptr<AecBackend> create_webrtc_backend(
    const PreprocessorTimelineConfig& settings) {
#if DVO_HAS_WEBRTC_APM
  try {
    return std::make_unique<WebRtcAecBackend>(settings);
  } catch (const std::exception&) {
    // A binary can be built with APM support yet still fail to initialize the
    // backend (for example because of an ABI mismatch).  Keep the real-time
    // path alive and expose the fallback through diagnostics.
    return nullptr;
  }
#else
  static_cast<void>(settings);
  return nullptr;
#endif
}

std::vector<float> downmix_packet(const AudioPacket& packet) {
  const auto channels = static_cast<std::size_t>(packet.format.channels);
  const auto input_frames = packet.samples.size() / channels;
  std::vector<float> mono(input_frames);
  if (!packet.silent) {
    for (std::size_t frame = 0; frame < input_frames; ++frame) {
      float sum{};
      for (std::size_t channel = 0; channel < channels; ++channel) {
        sum += packet.samples[frame * channels + channel];
      }
      mono[frame] = std::clamp(sum / static_cast<float>(channels), -1.0F, 1.0F);
    }
  }
  return mono;
}

struct ResampleOutcome {
  std::vector<float> samples;
  bool used_speex{};
  bool failed{};
  bool rate_updated{};
};

class StreamingResampler {
 public:
  StreamingResampler() = default;
  StreamingResampler(const StreamingResampler&) = delete;
  StreamingResampler& operator=(const StreamingResampler&) = delete;
  ~StreamingResampler() { destroy_speex(); }

  ResampleOutcome process(const AudioPacket& packet, double effective_rate_hz) {
    auto mono = downmix_packet(packet);
    if (mono.empty()) return {};
    if (!std::isfinite(effective_rate_hz) || effective_rate_hz <= 0.0) {
      effective_rate_hz = static_cast<double>(packet.format.sample_rate);
    }

#if DVO_HAS_SPEEXDSP
    if (!speex_disabled_) {
      auto outcome = process_speex(mono, effective_rate_hz);
      if (outcome.used_speex) return outcome;
      if (outcome.failed) {
        outcome.samples = process_linear(mono, effective_rate_hz);
        return outcome;
      }
    }
#endif

    ResampleOutcome outcome;
    outcome.samples = process_linear(mono, effective_rate_hz);
    return outcome;
  }

  // A synthetic idle packet has already advanced the streaming resampler for
  // this hardware interval. If the delayed real WASAPI packet later replaces
  // that still-buffered interval, resample it to the exact provisional output
  // length without advancing phase a second time.
  ResampleOutcome replace_provisional(const AudioPacket& packet,
                                      std::size_t output_count) const {
    ResampleOutcome outcome;
    const auto mono = downmix_packet(packet);
    if (mono.empty() || output_count == 0) return outcome;
    outcome.samples.resize(output_count);
    const long double step =
        static_cast<long double>(mono.size()) / output_count;
    for (std::size_t i = 0; i < output_count; ++i) {
      const auto position = static_cast<long double>(i) * step;
      const auto lower = std::min(
          static_cast<std::size_t>(std::floor(position)), mono.size() - 1);
      const auto upper = std::min(lower + 1, mono.size() - 1);
      const auto fraction = static_cast<float>(position - std::floor(position));
      outcome.samples[i] = mono[lower] + (mono[upper] - mono[lower]) * fraction;
    }
#if DVO_HAS_SPEEXDSP
    outcome.used_speex = speex_ != nullptr;
#endif
    return outcome;
  }

  void reset() {
    destroy_speex();
    reset_linear();
#if DVO_HAS_SPEEXDSP
    speex_disabled_ = false;
    ratio_numerator_ = 0;
    ratio_denominator_ = 0;
#endif
  }

 private:
  // The lightweight build still needs a genuinely streaming fallback.  Keep
  // the next output position in the continuous input-sample domain and retain
  // the previous packet's final sample, otherwise a non-integer ratio (or a
  // changing drift ratio) restarts interpolation at every packet boundary and
  // creates periodic skipped/repeated samples.
  std::vector<float> process_linear(const std::vector<float>& mono,
                                    double effective_rate_hz) {
    if (mono.empty()) return {};
    const auto packet_start_index = linear_input_samples_;
    const auto packet_end_index = packet_start_index + mono.size();
    const auto packet_start = static_cast<long double>(packet_start_index);
    const auto packet_end = static_cast<long double>(packet_end_index);
    if (!linear_initialized_) {
      linear_next_input_position_ = packet_start;
      linear_initialized_ = true;
    }

    const long double input_step =
        static_cast<long double>(effective_rate_hz) / kProcessingSampleRate;
    std::vector<float> output;
    const auto estimate = static_cast<std::size_t>(
        std::ceil(static_cast<long double>(mono.size()) / input_step)) + 1;
    output.reserve(estimate);

    while (linear_next_input_position_ < packet_end) {
      const auto lower_position = std::floor(linear_next_input_position_);
      const auto lower = static_cast<std::uint64_t>(lower_position);
      auto fraction = linear_next_input_position_ - lower_position;
      if (fraction < 1e-12L) fraction = 0.0L;
      const auto upper = lower + (fraction == 0.0L ? 0 : 1);
      if (upper >= packet_end_index) break;

      const auto sample_at = [&](std::uint64_t position) {
        if (position < packet_start_index) {
          return linear_has_previous_sample_ ? linear_previous_sample_ : mono.front();
        }
        return mono[static_cast<std::size_t>(position - packet_start_index)];
      };
      const auto a = sample_at(lower);
      const auto b = sample_at(upper);
      output.push_back(a + (b - a) * static_cast<float>(fraction));
      linear_next_input_position_ += input_step;
    }

    linear_previous_sample_ = mono.back();
    linear_input_samples_ += mono.size();
    linear_has_previous_sample_ = true;
    return output;
  }

  void reset_linear() {
    linear_next_input_position_ = 0.0L;
    linear_input_samples_ = 0;
    linear_previous_sample_ = 0.0F;
    linear_has_previous_sample_ = false;
    linear_initialized_ = false;
  }

  void destroy_speex() {
#if DVO_HAS_SPEEXDSP
    if (speex_) speex_resampler_destroy(speex_);
    speex_ = nullptr;
#endif
  }

#if DVO_HAS_SPEEXDSP
  ResampleOutcome process_speex(const std::vector<float>& mono,
                                double effective_rate_hz) {
    ResampleOutcome outcome;
    constexpr std::uint32_t kRateScale = 1000;
    constexpr std::uint32_t kRatioDenominator =
        kProcessingSampleRate * kRateScale;
    const auto maximum_u32 = std::numeric_limits<std::uint32_t>::max();
    if (effective_rate_hz > static_cast<double>(maximum_u32) / kRateScale ||
        mono.size() > maximum_u32) {
      disable_speex();
      outcome.failed = true;
      return outcome;
    }

    const auto ratio_numerator = static_cast<std::uint32_t>(
        std::max<long long>(1, std::llround(effective_rate_hz * kRateScale)));
    const auto display_rate = static_cast<std::uint32_t>(
        std::max<long long>(1, std::llround(effective_rate_hz)));
    int error = RESAMPLER_ERR_SUCCESS;
    if (!speex_) {
      speex_ = speex_resampler_init_frac(
          1, ratio_numerator, kRatioDenominator, display_rate,
          kProcessingSampleRate, SPEEX_RESAMPLER_QUALITY_VOIP, &error);
      if (!speex_ || error != RESAMPLER_ERR_SUCCESS) {
        disable_speex();
        outcome.failed = true;
        return outcome;
      }
      ratio_numerator_ = ratio_numerator;
      ratio_denominator_ = kRatioDenominator;
      outcome.rate_updated = true;
    } else if (ratio_numerator_ != ratio_numerator ||
               ratio_denominator_ != kRatioDenominator) {
      error = speex_resampler_set_rate_frac(
          speex_, ratio_numerator, kRatioDenominator, display_rate,
          kProcessingSampleRate);
      if (error != RESAMPLER_ERR_SUCCESS) {
        disable_speex();
        outcome.failed = true;
        return outcome;
      }
      ratio_numerator_ = ratio_numerator;
      ratio_denominator_ = kRatioDenominator;
      outcome.rate_updated = true;
    }

    const long double expected =
        static_cast<long double>(mono.size()) * kProcessingSampleRate /
        static_cast<long double>(effective_rate_hz);
    const auto capacity_long_double = std::ceil(expected) + 512.0L;
    if (capacity_long_double > maximum_u32) {
      disable_speex();
      outcome.failed = true;
      return outcome;
    }
    const auto capacity = static_cast<std::uint32_t>(capacity_long_double);
    outcome.samples.resize(capacity);
    auto input_length = static_cast<std::uint32_t>(mono.size());
    auto output_length = capacity;
    error = speex_resampler_process_float(
        speex_, 0, mono.data(), &input_length, outcome.samples.data(),
        &output_length);
    if (error != RESAMPLER_ERR_SUCCESS || input_length != mono.size()) {
      disable_speex();
      outcome.samples.clear();
      outcome.failed = true;
      return outcome;
    }
    outcome.samples.resize(output_length);
    outcome.used_speex = true;
    return outcome;
  }

  void disable_speex() {
    destroy_speex();
    speex_disabled_ = true;
    ratio_numerator_ = 0;
    ratio_denominator_ = 0;
    reset_linear();
  }

  SpeexResamplerState* speex_{};
  bool speex_disabled_{};
  std::uint32_t ratio_numerator_{};
  std::uint32_t ratio_denominator_{};
#endif

  long double linear_next_input_position_{};
  std::uint64_t linear_input_samples_{};
  float linear_previous_sample_{};
  bool linear_has_previous_sample_{};
  bool linear_initialized_{};
};

struct SyntheticResampleRecord {
  std::int64_t start_grid{};
  std::size_t input_samples{};
  std::size_t output_samples{};
};

struct StreamState {
  TimedSampleBuffer buffer;
  ClockRateEstimator clock;
  StreamingResampler resampler;
  AudioFormat format{};
  bool format_seen{};
  std::uint64_t epoch{};
  bool epoch_seen{};
  std::uint64_t last_sequence{};
  bool sequence_seen{};
  std::int64_t expected_next_grid{};
  bool expected_next_grid_valid{};
  std::uint64_t latest_latency_100ns{};
  bool latest_latency_valid{};
  std::uint64_t grid_anchor_position{};
  std::int64_t grid_anchor_start{};
  bool grid_anchor_valid{};
  std::deque<SyntheticResampleRecord> synthetic_resamples;

  void reset() {
    buffer.clear();
    clock.reset();
    resampler.reset();
    format = {};
    format_seen = false;
    epoch = 0;
    epoch_seen = false;
    last_sequence = 0;
    sequence_seen = false;
    expected_next_grid = 0;
    expected_next_grid_valid = false;
    latest_latency_100ns = 0;
    latest_latency_valid = false;
    grid_anchor_position = 0;
    grid_anchor_start = 0;
    grid_anchor_valid = false;
    synthetic_resamples.clear();
  }
};

struct QueuedFrame {
  NormalizedFrame frame;
  AecRuntimeState state{AecRuntimeState::bypass};
  bool render_available{};
  bool render_synthetic{};
  bool aec_applied{};
  bool degraded{};
  int stream_delay_ms{};
};

class TimelineEngine {
 public:
  TimelineEngine(const AudioConfig& audio, PreprocessorTimelineConfig timeline,
                 bool aec_requested, std::unique_ptr<AecBackend> backend)
      : audio_(audio), timeline_(timeline), aec_requested_(aec_requested),
        backend_(std::move(backend)) {
    if (audio_.target_sample_rate != kProcessingSampleRate || audio_.frame_ms != 10) {
      throw std::invalid_argument("timeline preprocessor requires 16 kHz / 10 ms output");
    }
    if (timeline_.max_buffer_ms < 20 ||
        timeline_.alignment_wait_ms > timeline_.max_buffer_ms ||
        timeline_.target_render_buffer_ms < 10 ||
        timeline_.target_render_buffer_ms > timeline_.max_buffer_ms ||
        timeline_.drift_window_ms < 1000 || timeline_.drift_window_ms > 60000 ||
        timeline_.hard_resync_error_ms < 10 || timeline_.max_drift_ppm < 0.0 ||
        timeline_.stats_hz == 0 || timeline_.stats_hz > 20) {
      throw std::invalid_argument("invalid preprocessor timeline configuration");
    }
    maximum_buffer_samples_ = static_cast<std::size_t>(timeline_.max_buffer_ms) *
                              kProcessingSampleRate / 1000;
    alignment_wait_samples_ = static_cast<std::size_t>(timeline_.alignment_wait_ms) *
                              kProcessingSampleRate / 1000;
    target_render_buffer_samples_ =
        static_cast<std::size_t>(timeline_.target_render_buffer_ms) *
        kProcessingSampleRate / 1000;
    drift_window_100ns_ = static_cast<std::uint64_t>(timeline_.drift_window_ms) * 10'000ULL;
    hard_resync_samples_ = static_cast<std::int64_t>(timeline_.hard_resync_error_ms) *
                           kProcessingSampleRate / 1000;
    maximum_output_frames_ = std::max<std::size_t>(4, timeline_.max_buffer_ms / 10);
    diagnostics_.aec_requested = aec_requested_;
    diagnostics_.aec_compiled = webrtc_aec3_compiled();
    diagnostics_.speexdsp_compiled = speexdsp_compiled();
    diagnostics_.degraded = aec_requested_ && !backend_;
    diagnostics_.state = !aec_requested_ ? AecRuntimeState::bypass
                         : backend_       ? AecRuntimeState::waiting_for_render
                                          : AecRuntimeState::degraded_backend_unavailable;
    diagnostics_.target_render_buffer_samples = target_render_buffer_samples_;
  }

  PreprocessPushResult push(const AudioPacket& packet) {
    PreprocessPushResult result;
    const auto output_drops_before = diagnostics_.output_frames_dropped;
    if (packet.stream == AudioStreamKind::processed || packet.format.sample_rate == 0 ||
        packet.format.channels == 0 || packet.samples.empty() ||
        packet.samples.size() % packet.format.channels != 0) {
      Reset(PreprocessResetReason::invalid_packet);
      result.reset_required = true;
      result.reset_reason = PreprocessResetReason::invalid_packet;
      result.frames_ready = output_.size();
      return result;
    }
    if (packet.timestamp_error) {
      ++diagnostics_.timestamp_errors;
      Reset(PreprocessResetReason::timestamp_error);
      result.reset_required = true;
      result.reset_reason = PreprocessResetReason::timestamp_error;
      result.frames_ready = output_.size();
      return result;
    }

    auto& state = stream(packet.stream);
    auto reset_reason = detect_reset_reason(state, packet);
    if (reset_reason != PreprocessResetReason::none) {
      Reset(reset_reason);
      result.reset_required = true;
      result.reset_reason = reset_reason;
    }

    if (!origin_valid_) {
      origin_qpc_100ns_ = packet.qpc_100ns;
      origin_valid_ = true;
    }
    auto start_grid = packet_start_grid(state, packet);
    update_metadata(state, packet);

    const auto clock_update = state.clock.update(packet, drift_window_100ns_);
    if (clock_update == ClockRateEstimator::Update::backwards) {
      Reset(PreprocessResetReason::hard_resync);
      ++diagnostics_.hard_resyncs;
      result.reset_required = true;
      result.reset_reason = PreprocessResetReason::hard_resync;
      origin_qpc_100ns_ = packet.qpc_100ns;
      origin_valid_ = true;
      start_grid = 0;
      auto& restarted = stream(packet.stream);
      set_grid_anchor(restarted, packet, start_grid);
      update_metadata(restarted, packet);
      static_cast<void>(restarted.clock.update(packet, drift_window_100ns_));
    }

    auto& active = stream(packet.stream);
    const auto packet_input_samples =
        packet.samples.size() / static_cast<std::size_t>(packet.format.channels);
    std::size_t provisional_record_begin{};
    std::size_t provisional_record_count{};
    std::size_t provisional_input_samples{};
    std::size_t provisional_output_samples{};
    if (packet.stream == AudioStreamKind::loopback && !packet.synthetic) {
      auto expected_start = start_grid;
      for (std::size_t record_index = 0;
           record_index < active.synthetic_resamples.size(); ++record_index) {
        const auto& record = active.synthetic_resamples[record_index];
        if (record.start_grid < expected_start) continue;
        if (record.start_grid != expected_start) break;
        if (provisional_record_count == 0) provisional_record_begin = record_index;
        provisional_input_samples += record.input_samples;
        provisional_output_samples += record.output_samples;
        ++provisional_record_count;
        expected_start += static_cast<std::int64_t>(record.output_samples);
        if (provisional_input_samples >= packet_input_samples) break;
      }
      if (provisional_input_samples != packet_input_samples) {
        provisional_record_count = 0;
        provisional_output_samples = 0;
      }
    }
    auto resampled = provisional_record_count != 0
                         ? active.resampler.replace_provisional(
                               packet, provisional_output_samples)
                         : active.resampler.process(packet, active.clock.rate_hz());
    if (active.expected_next_grid_valid) {
      const auto error = start_grid - active.expected_next_grid;
      if (std::abs(error) > hard_resync_samples_) {
        Reset(PreprocessResetReason::hard_resync);
        ++diagnostics_.hard_resyncs;
        result.reset_required = true;
        result.reset_reason = PreprocessResetReason::hard_resync;
        origin_qpc_100ns_ = packet.qpc_100ns;
        origin_valid_ = true;
        start_grid = 0;
        auto& restarted = stream(packet.stream);
        set_grid_anchor(restarted, packet, start_grid);
        update_metadata(restarted, packet);
        static_cast<void>(restarted.clock.update(packet, drift_window_100ns_));
        resampled = restarted.resampler.process(packet, restarted.clock.rate_hz());
      }
    }

    auto& destination = stream(packet.stream);
    if (packet.stream == AudioStreamKind::microphone) {
      diagnostics_.microphone_resampler_speex = resampled.used_speex;
    } else {
      diagnostics_.render_resampler_speex = resampled.used_speex;
      if (resampled.rate_updated) ++diagnostics_.render_resampler_rate_updates;
    }
    if (resampled.failed) {
      ++diagnostics_.resampler_failures;
      pending_discontinuity_ = true;
      result.reset_required = true;
      result.reset_reason = PreprocessResetReason::resampler_error;
      diagnostics_.last_reset_reason = PreprocessResetReason::resampler_error;
      if (backend_) backend_->reset();
      backend_had_render_ = false;
    }
    auto values = std::move(resampled.samples);
    const auto flags = packet.synthetic ? kSyntheticSample : kRealSample;
    const auto append = destination.buffer.append(start_grid, values, flags,
                                                  maximum_buffer_samples_);
    if (packet.stream == AudioStreamKind::loopback) {
      if (provisional_record_count != 0) {
        const auto first = destination.synthetic_resamples.begin() +
                           static_cast<std::ptrdiff_t>(provisional_record_begin);
        destination.synthetic_resamples.erase(
            first, first + static_cast<std::ptrdiff_t>(provisional_record_count));
      } else if (packet.synthetic) {
        destination.synthetic_resamples.push_back(
            {start_grid, packet_input_samples, values.size()});
        const auto maximum_records = maximum_output_frames_ + 10;
        while (destination.synthetic_resamples.size() > maximum_records) {
          destination.synthetic_resamples.pop_front();
        }
      } else {
        // A non-matching real packet means the device resumed on a different
        // hardware boundary. Old provisional records can no longer be
        // replaced safely.
        destination.synthetic_resamples.clear();
      }
    }
    diagnostics_.drift_correction_samples += append.correction_samples;
    if (packet.stream == AudioStreamKind::loopback) {
      diagnostics_.render_synthetic_samples_replaced +=
          append.replaced_synthetic_samples;
    }
    if (append.contains_gap && packet.stream == AudioStreamKind::microphone) {
      pending_discontinuity_ = true;
    }
    if (append.overflow_samples != 0) {
      ++diagnostics_.buffer_overflows;
      if (packet.stream == AudioStreamKind::microphone) {
        pending_discontinuity_ = true;
        result.reset_required = true;
        result.reset_reason = PreprocessResetReason::buffer_overflow;
        diagnostics_.last_reset_reason = PreprocessResetReason::buffer_overflow;
        if (next_frame_grid_valid_ && next_frame_grid_ < destination.buffer.start()) {
          next_frame_grid_ = destination.buffer.start();
        }
        if (backend_) backend_->reset();
        backend_had_render_ = false;
      }
    }
    destination.expected_next_grid =
        start_grid + static_cast<std::int64_t>(values.size());
    destination.expected_next_grid_valid = true;
    update_packet_diagnostics(packet, destination);

    if (packet.stream == AudioStreamKind::microphone && !next_frame_grid_valid_) {
      next_frame_grid_ = destination.buffer.start();
      next_frame_grid_valid_ = true;
    }
    produce_frames();
    if (diagnostics_.output_frames_dropped != output_drops_before) {
      result.reset_required = true;
      result.reset_reason = PreprocessResetReason::buffer_overflow;
      diagnostics_.last_reset_reason = PreprocessResetReason::buffer_overflow;
    }
    update_buffer_diagnostics();
    result.accepted = true;
    result.frames_ready = output_.size();
    return result;
  }

  bool pop(NormalizedFrame& frame) {
    if (output_.empty()) return false;
    auto queued = std::move(output_.front());
    output_.pop_front();
    frame = std::move(queued.frame);
    diagnostics_.state = queued.state;
    diagnostics_.render_available = queued.render_available;
    diagnostics_.render_synthetic = queued.render_synthetic;
    diagnostics_.aec_active = queued.aec_applied;
    diagnostics_.degraded = queued.degraded;
    diagnostics_.stream_delay_ms = queued.stream_delay_ms;
    diagnostics_.output_frames_ready = output_.size();
    return true;
  }

  void Reset(PreprocessResetReason reason) {
    microphone_.reset();
    render_.reset();
    output_.clear();
    origin_valid_ = false;
    next_frame_grid_valid_ = false;
    pending_discontinuity_ = true;
    previous_degraded_valid_ = false;
    backend_had_render_ = false;
    if (backend_) backend_->reset();
    ++diagnostics_.resets;
    diagnostics_.last_reset_reason = reason;
    diagnostics_.aec_active = false;
    diagnostics_.degraded = aec_requested_ && !backend_;
    diagnostics_.state = !aec_requested_ ? AecRuntimeState::bypass
                         : backend_       ? AecRuntimeState::waiting_for_render
                                          : AecRuntimeState::degraded_backend_unavailable;
    diagnostics_.render_seen = false;
    diagnostics_.render_available = false;
    diagnostics_.render_synthetic = false;
    diagnostics_.drift_out_of_range = false;
    diagnostics_.drift_estimate_valid = false;
    diagnostics_.microphone_resampler_speex = false;
    diagnostics_.render_resampler_speex = false;
    diagnostics_.microphone_rate_hz = 0.0;
    diagnostics_.render_rate_hz = 0.0;
    diagnostics_.relative_drift_ppm = 0.0;
    diagnostics_.stream_delay_ms = 0;
    diagnostics_.microphone_buffered_samples = 0;
    diagnostics_.render_buffered_samples = 0;
    diagnostics_.render_buffer_error_samples =
        -static_cast<std::int64_t>(target_render_buffer_samples_);
    diagnostics_.output_frames_ready = 0;
  }

  [[nodiscard]] PreprocessDiagnostics diagnostics() const {
    auto value = diagnostics_;
    value.output_frames_ready = output_.size();
    value.microphone_buffered_samples = microphone_.buffer.size();
    value.render_buffered_samples = render_.buffer.size();
    value.target_render_buffer_samples = target_render_buffer_samples_;
    value.render_buffer_error_samples =
        static_cast<std::int64_t>(render_.buffer.size()) -
        static_cast<std::int64_t>(target_render_buffer_samples_);
    return value;
  }

 private:
  static void update_metadata(StreamState& state, const AudioPacket& packet) {
    state.format = packet.format;
    state.format_seen = true;
    state.epoch = packet.stream_epoch;
    state.epoch_seen = true;
    if (packet.sequence != 0) {
      state.last_sequence = packet.sequence;
      state.sequence_seen = true;
    }
    if (packet.arrival_qpc_100ns >= packet.qpc_100ns && packet.arrival_qpc_100ns != 0) {
      state.latest_latency_100ns = packet.arrival_qpc_100ns - packet.qpc_100ns;
      state.latest_latency_valid = true;
    }
  }

  [[nodiscard]] PreprocessResetReason detect_reset_reason(
      const StreamState& state, const AudioPacket& packet) const {
    if (packet.discontinuity) return PreprocessResetReason::discontinuity;
    if (state.format_seen && (state.format.sample_rate != packet.format.sample_rate ||
                              state.format.channels != packet.format.channels)) {
      return PreprocessResetReason::format_change;
    }
    if (state.epoch_seen && state.epoch != packet.stream_epoch) {
      return PreprocessResetReason::stream_epoch_change;
    }
    if (state.sequence_seen && packet.sequence != 0 &&
        packet.sequence != state.last_sequence + 1) {
      return PreprocessResetReason::sequence_gap;
    }
    return PreprocessResetReason::none;
  }

  [[nodiscard]] StreamState& stream(AudioStreamKind kind) {
    return kind == AudioStreamKind::microphone ? microphone_ : render_;
  }

  [[nodiscard]] std::int64_t qpc_to_grid(std::uint64_t qpc) const {
    if (qpc >= origin_qpc_100ns_) {
      const auto delta = qpc - origin_qpc_100ns_;
      return static_cast<std::int64_t>((delta + kQpcUnitsPerOutputSample / 2) /
                                       kQpcUnitsPerOutputSample);
    }
    const auto delta = origin_qpc_100ns_ - qpc;
    return -static_cast<std::int64_t>((delta + kQpcUnitsPerOutputSample / 2) /
                                      kQpcUnitsPerOutputSample);
  }

  [[nodiscard]] std::int64_t packet_start_grid(StreamState& state,
                                                const AudioPacket& packet) const {
    const auto qpc_grid = qpc_to_grid(packet.qpc_100ns);
    // device_position is the continuous hardware sample clock and is not
    // subject to the sub-millisecond scheduling jitter present in the packet
    // QPC timestamp. QPC anchors the two independent streams; subsequent
    // packet starts follow device position. Legacy manifests that do not carry
    // device positions leave it at zero and retain QPC-only compatibility.
    if (packet.device_position == 0) return qpc_grid;
    if (!state.grid_anchor_valid ||
        packet.device_position < state.grid_anchor_position) {
      set_grid_anchor(state, packet, qpc_grid);
      return qpc_grid;
    }
    const auto device_delta = packet.device_position - state.grid_anchor_position;
    const auto output_delta = static_cast<std::int64_t>(std::llround(
        static_cast<long double>(device_delta) * kProcessingSampleRate /
        static_cast<long double>(packet.format.sample_rate)));
    return state.grid_anchor_start + output_delta;
  }

  static void set_grid_anchor(StreamState& state, const AudioPacket& packet,
                              std::int64_t start_grid) {
    if (packet.device_position == 0) {
      state.grid_anchor_valid = false;
      return;
    }
    state.grid_anchor_position = packet.device_position;
    state.grid_anchor_start = start_grid;
    state.grid_anchor_valid = true;
  }

  [[nodiscard]] std::uint64_t grid_to_qpc(std::int64_t grid) const {
    const long double value = static_cast<long double>(origin_qpc_100ns_) +
                              static_cast<long double>(grid) *
                                  static_cast<long double>(kQpcUnitsPerOutputSample);
    if (value <= 0.0L) return 0;
    if (value >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::llround(value));
  }

  [[nodiscard]] int stream_delay_ms() const {
    // Loopback packets are post-render observations and capture/render frames
    // are extracted at the same QPC grid, immediately followed by the two APM
    // calls.  Therefore (t_render - t_analyze) +
    // (t_process - t_capture) is approximately zero. Packet arrival latency
    // is deliberately excluded: FIFO backlog and asymmetric delivery change
    // arrival time without changing the hardware relation and would feed AEC3
    // a false delay hint. A measured installation-specific correction remains
    // available through delay_offset_ms.
    return static_cast<int>(
        std::clamp<std::int64_t>(timeline_.delay_offset_ms, 0, 500));
  }

  void produce_frames() {
    while (next_frame_grid_valid_ &&
           microphone_.buffer.has_range(next_frame_grid_, kFrameSamples)) {
      std::array<float, kFrameSamples> capture{};
      std::array<float, kFrameSamples> render{};
      const auto capture_info = microphone_.buffer.extract(next_frame_grid_, capture);
      auto render_info = render_.buffer.extract(next_frame_grid_, render);
      const bool render_usable = render_info.complete && !render_info.contains_gap;
      const bool needs_render = aec_requested_ && backend_ != nullptr;
      const auto microphone_depth = static_cast<std::size_t>(std::max<std::int64_t>(
          0, microphone_.buffer.end() - next_frame_grid_));
      const auto render_depth = static_cast<std::size_t>(std::max<std::int64_t>(
          0, render_.buffer.end() - next_frame_grid_));
      if (needs_render) {
        const bool waiting_for_missing_render =
            !render_usable && microphone_depth < kFrameSamples + alignment_wait_samples_;
        const bool filling_target_buffer =
            render_usable && render_depth < target_render_buffer_samples_ &&
            microphone_depth < target_render_buffer_samples_ + alignment_wait_samples_;
        if (waiting_for_missing_render || filling_target_buffer) {
          diagnostics_.state = AecRuntimeState::waiting_for_render;
          diagnostics_.render_available = render_usable;
          diagnostics_.render_buffered_samples = render_.buffer.size();
          diagnostics_.render_buffer_error_samples =
              static_cast<std::int64_t>(render_.buffer.size()) -
              static_cast<std::int64_t>(target_render_buffer_samples_);
          break;
        }
      }

      update_relative_drift();
      diagnostics_.drift_out_of_range =
          diagnostics_.drift_estimate_valid &&
          std::abs(diagnostics_.relative_drift_ppm) > timeline_.max_drift_ppm;
      const int delay_ms = stream_delay_ms();
      std::array<float, kFrameSamples> processed = capture;
      bool aec_applied{};
      bool degraded{};
      auto state = AecRuntimeState::bypass;
      if (aec_requested_) {
        if (!backend_) {
          degraded = true;
          state = AecRuntimeState::degraded_backend_unavailable;
        } else if (diagnostics_.drift_out_of_range) {
          degraded = true;
          state = AecRuntimeState::degraded_drift;
          if (backend_had_render_) backend_->reset();
          backend_had_render_ = false;
        } else if (!render_usable) {
          degraded = true;
          state = AecRuntimeState::degraded_missing_render;
          if (backend_had_render_) backend_->reset();
          backend_had_render_ = false;
        } else {
          if (!backend_had_render_) backend_->reset();
          backend_had_render_ = true;
          const auto started = std::chrono::steady_clock::now();
          const bool processed_ok = backend_->process(render, capture, delay_ms, processed);
          const auto processing_time_us = std::chrono::duration<double, std::micro>(
                                              std::chrono::steady_clock::now() - started)
                                              .count();
          observe_processing_time(processing_time_us);
          if (processed_ok) {
            aec_applied = true;
            state = AecRuntimeState::active;
          } else {
            processed = capture;
            degraded = true;
            state = AecRuntimeState::backend_error;
            backend_->reset();
            backend_had_render_ = false;
            diagnostics_.last_reset_reason = PreprocessResetReason::backend_error;
          }
        }
      }

      const bool degraded_transition =
          previous_degraded_valid_ && previous_degraded_ != degraded;
      previous_degraded_ = degraded;
      previous_degraded_valid_ = true;

      QueuedFrame queued;
      queued.frame.first_sample = next_output_sample_;
      queued.frame.qpc_100ns = grid_to_qpc(next_frame_grid_);
      queued.frame.samples = processed;
      queued.frame.discontinuity = pending_discontinuity_ || capture_info.contains_gap ||
                                   degraded_transition;
      queued.state = state;
      queued.render_available = render_usable;
      queued.render_synthetic =
          render_info.contains_synthetic && !render_info.contains_real;
      queued.aec_applied = aec_applied;
      queued.degraded = degraded;
      queued.stream_delay_ms = delay_ms;
      pending_discontinuity_ = false;

      if (output_.size() >= maximum_output_frames_) {
        output_.pop_front();
        ++diagnostics_.output_frames_dropped;
        queued.frame.discontinuity = true;
      }
      output_.push_back(std::move(queued));
      ++diagnostics_.output_frames;
      if (aec_applied) {
        ++diagnostics_.aec_frames;
      } else if (aec_requested_) {
        ++diagnostics_.fallback_frames;
      }
      diagnostics_.aec_active = aec_applied;
      diagnostics_.degraded = degraded;
      diagnostics_.state = state;
      diagnostics_.render_available = render_usable;
      diagnostics_.render_synthetic =
          render_info.contains_synthetic && !render_info.contains_real;
      diagnostics_.stream_delay_ms = delay_ms;
      if (backend_) copy_backend_metrics(backend_->metrics());

      next_output_sample_ += kFrameSamples;
      next_frame_grid_ += static_cast<std::int64_t>(kFrameSamples);
      microphone_.buffer.discard_before(next_frame_grid_);
      render_.buffer.discard_before(next_frame_grid_);
    }
  }

  void update_packet_diagnostics(const AudioPacket& packet, const StreamState& state) {
    if (packet.stream == AudioStreamKind::microphone) {
      ++diagnostics_.microphone_packets;
      diagnostics_.microphone_rate_hz = state.clock.rate_hz();
    } else {
      ++diagnostics_.render_packets;
      diagnostics_.render_seen = true;
      diagnostics_.render_rate_hz = state.clock.rate_hz();
    }
    update_relative_drift();
  }

  void update_relative_drift() {
    diagnostics_.microphone_rate_hz = microphone_.clock.rate_hz();
    diagnostics_.render_rate_hz = render_.clock.rate_hz();
    diagnostics_.drift_estimate_valid =
        microphone_.clock.valid() && render_.clock.valid() &&
        diagnostics_.microphone_rate_hz > 0.0;
    if (diagnostics_.drift_estimate_valid) {
      diagnostics_.relative_drift_ppm =
          (diagnostics_.render_rate_hz / diagnostics_.microphone_rate_hz - 1.0) *
          1'000'000.0;
    } else {
      // Do not reuse a previous window after the estimator rejects an anchor.
      // Nominal-rate processing continues while a fresh window is collected.
      diagnostics_.relative_drift_ppm = 0.0;
    }
  }

  void update_buffer_diagnostics() {
    diagnostics_.microphone_buffered_samples = microphone_.buffer.size();
    diagnostics_.render_buffered_samples = render_.buffer.size();
    diagnostics_.target_render_buffer_samples = target_render_buffer_samples_;
    diagnostics_.render_buffer_error_samples =
        static_cast<std::int64_t>(render_.buffer.size()) -
        static_cast<std::int64_t>(target_render_buffer_samples_);
    diagnostics_.output_frames_ready = output_.size();
  }

  void observe_processing_time(double microseconds) {
    diagnostics_.last_processing_time_us = microseconds;
    diagnostics_.max_processing_time_us =
        std::max(diagnostics_.max_processing_time_us, microseconds);
    ++diagnostics_.processing_measurements;
    diagnostics_.average_processing_time_us +=
        (microseconds - diagnostics_.average_processing_time_us) /
        static_cast<double>(diagnostics_.processing_measurements);
  }

  void copy_backend_metrics(const BackendMetrics& metrics) {
    diagnostics_.echo_return_loss_db = metrics.echo_return_loss_db;
    diagnostics_.echo_return_loss_enhancement_db =
        metrics.echo_return_loss_enhancement_db;
    diagnostics_.residual_echo_likelihood = metrics.residual_echo_likelihood;
    diagnostics_.divergent_filter_fraction = metrics.divergent_filter_fraction;
    diagnostics_.estimated_delay_ms = metrics.estimated_delay_ms;
  }

  AudioConfig audio_;
  PreprocessorTimelineConfig timeline_;
  bool aec_requested_{};
  std::unique_ptr<AecBackend> backend_;
  StreamState microphone_;
  StreamState render_;
  std::deque<QueuedFrame> output_;
  PreprocessDiagnostics diagnostics_;
  std::uint64_t origin_qpc_100ns_{};
  bool origin_valid_{};
  std::int64_t next_frame_grid_{};
  bool next_frame_grid_valid_{};
  std::uint64_t next_output_sample_{};
  bool pending_discontinuity_{};
  bool backend_had_render_{};
  bool previous_degraded_{};
  bool previous_degraded_valid_{};
  std::size_t maximum_buffer_samples_{};
  std::size_t alignment_wait_samples_{};
  std::size_t target_render_buffer_samples_{};
  std::uint64_t drift_window_100ns_{};
  std::int64_t hard_resync_samples_{};
  std::size_t maximum_output_frames_{};
};

}  // namespace

const char* to_string(PreprocessResetReason reason) noexcept {
  switch (reason) {
    case PreprocessResetReason::none: return "none";
    case PreprocessResetReason::explicit_request: return "explicit_request";
    case PreprocessResetReason::invalid_packet: return "invalid_packet";
    case PreprocessResetReason::discontinuity: return "discontinuity";
    case PreprocessResetReason::timestamp_error: return "timestamp_error";
    case PreprocessResetReason::format_change: return "format_change";
    case PreprocessResetReason::stream_epoch_change: return "stream_epoch_change";
    case PreprocessResetReason::sequence_gap: return "sequence_gap";
    case PreprocessResetReason::hard_resync: return "hard_resync";
    case PreprocessResetReason::buffer_overflow: return "buffer_overflow";
    case PreprocessResetReason::backend_error: return "backend_error";
    case PreprocessResetReason::resampler_error: return "resampler_error";
  }
  return "unknown";
}

const char* to_string(AecRuntimeState state) noexcept {
  switch (state) {
    case AecRuntimeState::bypass: return "bypass";
    case AecRuntimeState::waiting_for_render: return "waiting_for_render";
    case AecRuntimeState::active: return "active";
    case AecRuntimeState::degraded_backend_unavailable:
      return "degraded_backend_unavailable";
    case AecRuntimeState::degraded_missing_render: return "degraded_missing_render";
    case AecRuntimeState::degraded_drift: return "degraded_drift";
    case AecRuntimeState::backend_error: return "backend_error";
  }
  return "unknown";
}

PreprocessorTimelineConfig make_preprocessor_timeline_config(
    const AecConfig& config) {
  PreprocessorTimelineConfig timeline;
  timeline.alignment_wait_ms = config.alignment_wait_ms;
  timeline.target_render_buffer_ms = config.target_render_buffer_ms;
  timeline.max_buffer_ms = config.max_render_buffer_ms;
  timeline.drift_window_ms = config.drift_window_ms;
  timeline.hard_resync_error_ms = config.hard_resync_error_ms;
  timeline.max_drift_ppm = static_cast<double>(config.max_drift_ppm);
  timeline.delay_offset_ms = config.delay_offset_ms;
  timeline.high_pass_filter = config.high_pass_filter;
  timeline.noise_suppression = config.noise_suppression;
  timeline.gain_control = config.gain_control;
  timeline.stats_hz = config.stats_hz;
  return timeline;
}

PreprocessResult IAudioPreprocessor::process(
    const AudioPacket& microphone, const std::optional<AudioPacket>& render) {
  PreprocessResult result;
  if (render) {
    const auto pushed = PushPacket(*render);
    result.reset_required |= pushed.reset_required;
  }
  const auto pushed = PushPacket(microphone);
  result.reset_required |= pushed.reset_required;
  NormalizedFrame frame;
  while (TryPopFrame(frame)) result.frames.push_back(frame);
  const auto diagnostics = Diagnostics();
  result.render_available = diagnostics.render_available;
  result.degraded = diagnostics.degraded;
  return result;
}

struct BypassPreprocessor::Impl {
  Impl(const AudioConfig& config, PreprocessorTimelineConfig timeline)
      : engine(config, timeline, false, nullptr) {}
  TimelineEngine engine;
};

BypassPreprocessor::BypassPreprocessor(const AudioConfig& config,
                                       PreprocessorTimelineConfig timeline)
    : impl_(std::make_unique<Impl>(config, timeline)) {}

BypassPreprocessor::~BypassPreprocessor() = default;

PreprocessPushResult BypassPreprocessor::PushPacket(const AudioPacket& packet) {
  return impl_->engine.push(packet);
}

bool BypassPreprocessor::TryPopFrame(NormalizedFrame& frame) {
  return impl_->engine.pop(frame);
}

void BypassPreprocessor::Reset(PreprocessResetReason reason) {
  impl_->engine.Reset(reason);
}

PreprocessDiagnostics BypassPreprocessor::Diagnostics() const {
  return impl_->engine.diagnostics();
}

struct WebRtcAec3Preprocessor::Impl {
  Impl(const AudioConfig& config, PreprocessorTimelineConfig timeline)
      : engine(config, timeline, true, create_webrtc_backend(timeline)) {}
  TimelineEngine engine;
};

WebRtcAec3Preprocessor::WebRtcAec3Preprocessor(
    const AudioConfig& config, PreprocessorTimelineConfig timeline)
    : impl_(std::make_unique<Impl>(config, timeline)) {}

WebRtcAec3Preprocessor::~WebRtcAec3Preprocessor() = default;

PreprocessPushResult WebRtcAec3Preprocessor::PushPacket(const AudioPacket& packet) {
  return impl_->engine.push(packet);
}

bool WebRtcAec3Preprocessor::TryPopFrame(NormalizedFrame& frame) {
  return impl_->engine.pop(frame);
}

void WebRtcAec3Preprocessor::Reset(PreprocessResetReason reason) {
  impl_->engine.Reset(reason);
}

PreprocessDiagnostics WebRtcAec3Preprocessor::Diagnostics() const {
  return impl_->engine.diagnostics();
}

}  // namespace dvo
