#include "dvo/inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#if DVO_HAS_SHERPA
#include <sherpa-onnx/c-api/c-api.h>
#endif

namespace dvo {
namespace {

class DisabledVad final : public IVadDetector {
 public:
  explicit DisabledVad(std::string reason) : reason_(std::move(reason)) {}
  VadUpdate accept(const NormalizedFrame&) override { return {}; }
  void reset() override {}
  [[nodiscard]] bool available() const override { return false; }
  [[nodiscard]] std::string status() const override { return reason_; }
 private:
  std::string reason_;
};

class DisabledKws final : public IKeywordSpotter {
 public:
  explicit DisabledKws(std::string reason) : reason_(std::move(reason)) {}
  std::optional<KwsHit> accept(const NormalizedFrame&) override { return std::nullopt; }
  void reset(std::uint64_t) override {}
  [[nodiscard]] bool available() const override { return false; }
  [[nodiscard]] std::string status() const override { return reason_; }
 private:
  std::string reason_;
};

#if DVO_HAS_SHERPA

class SherpaVad final : public IVadDetector {
 public:
  explicit SherpaVad(const VadConfig& config) : config_(config) {
    SherpaOnnxVadModelConfig c{};
    model_path_ = config.model.string();
    provider_ = config.provider;
    c.silero_vad.model = model_path_.c_str();
    c.silero_vad.threshold = config.threshold;
    c.silero_vad.min_silence_duration = static_cast<float>(config.min_silence_ms) / 1000.0F;
    c.silero_vad.min_speech_duration = static_cast<float>(config.min_speech_ms) / 1000.0F;
    c.silero_vad.max_speech_duration = static_cast<float>(config.max_speech_ms) / 1000.0F;
    c.silero_vad.window_size = config.window_size;
    c.sample_rate = kProcessingSampleRate;
    c.num_threads = config.num_threads;
    c.provider = provider_.c_str();
    detector_ = SherpaOnnxCreateVoiceActivityDetector(&c, 30.0F);
    if (!detector_) throw std::runtime_error("SherpaOnnxCreateVoiceActivityDetector failed");
    pending_.reserve(static_cast<std::size_t>(config.window_size) * 2);
  }

  ~SherpaVad() override {
    if (detector_) SherpaOnnxDestroyVoiceActivityDetector(detector_);
  }

  VadUpdate accept(const NormalizedFrame& frame) override {
    if (!origin_) origin_ = frame.first_sample;
    pending_.insert(pending_.end(), frame.samples.begin(), frame.samples.end());
    while (pending_.size() >= static_cast<std::size_t>(config_.window_size)) {
      SherpaOnnxVoiceActivityDetectorAcceptWaveform(detector_, pending_.data(), config_.window_size);
      pending_.erase(pending_.begin(), pending_.begin() + config_.window_size);
      fed_samples_ += static_cast<std::uint64_t>(config_.window_size);
    }

    VadUpdate update;
    update.speech = SherpaOnnxVoiceActivityDetectorDetected(detector_) != 0;
    while (!SherpaOnnxVoiceActivityDetectorEmpty(detector_)) {
      const auto* segment = SherpaOnnxVoiceActivityDetectorFront(detector_);
      if (segment) {
        update.completed.push_back({{*origin_ + static_cast<std::uint64_t>(segment->start),
                                     *origin_ + static_cast<std::uint64_t>(segment->start + segment->n)}});
        SherpaOnnxDestroySpeechSegment(segment);
      }
      SherpaOnnxVoiceActivityDetectorPop(detector_);
    }
    return update;
  }

  void reset() override {
    SherpaOnnxVoiceActivityDetectorReset(detector_);
    pending_.clear();
    origin_.reset();
    fed_samples_ = 0;
  }

  [[nodiscard]] bool available() const override { return true; }
  [[nodiscard]] std::string status() const override { return "sherpa-onnx silero VAD ready"; }

 private:
  VadConfig config_;
  std::string model_path_;
  std::string provider_;
  const SherpaOnnxVoiceActivityDetector* detector_{};
  std::vector<float> pending_;
  std::optional<std::uint64_t> origin_;
  std::uint64_t fed_samples_{};
};

class SherpaKws final : public IKeywordSpotter {
 public:
  explicit SherpaKws(const KwsConfig& config) : config_(config) {
    encoder_ = config.encoder.string();
    decoder_ = config.decoder.string();
    joiner_ = config.joiner.string();
    tokens_ = config.tokens.string();
    keywords_ = config.keywords.string();
    provider_ = config.provider;

    SherpaOnnxKeywordSpotterConfig c{};
    c.feat_config.sample_rate = kProcessingSampleRate;
    c.feat_config.feature_dim = 80;
    c.model_config.transducer.encoder = encoder_.c_str();
    c.model_config.transducer.decoder = decoder_.c_str();
    c.model_config.transducer.joiner = joiner_.c_str();
    c.model_config.tokens = tokens_.c_str();
    c.model_config.provider = provider_.c_str();
    c.model_config.num_threads = config.num_threads;
    c.model_config.modeling_unit = "phone+ppinyin";
    c.max_active_paths = config.max_active_paths;
    c.num_trailing_blanks = config.num_trailing_blanks;
    c.keywords_score = config.boosting_score;
    c.keywords_threshold = config.threshold;
    c.keywords_file = keywords_.c_str();
    spotter_ = SherpaOnnxCreateKeywordSpotter(&c);
    if (!spotter_) throw std::runtime_error("SherpaOnnxCreateKeywordSpotter failed");
    stream_ = create_stream();
  }

  ~SherpaKws() override {
    if (stream_) SherpaOnnxDestroyOnlineStream(stream_);
    if (spotter_) SherpaOnnxDestroyKeywordSpotter(spotter_);
  }

  std::optional<KwsHit> accept(const NormalizedFrame& frame) override {
    SherpaOnnxOnlineStreamAcceptWaveform(stream_, kProcessingSampleRate,
                                         frame.samples.data(), static_cast<std::int32_t>(frame.samples.size()));
    while (SherpaOnnxIsKeywordStreamReady(spotter_, stream_)) {
      SherpaOnnxDecodeKeywordStream(spotter_, stream_);
    }
    const auto* result = SherpaOnnxGetKeywordResult(spotter_, stream_);
    if (!result) return std::nullopt;

    std::optional<KwsHit> hit;
    if (result->keyword && result->keyword[0] != '\0' && result->count > 0) {
      KwsHit value;
      value.keyword = result->keyword;
      value.detected_at_sample = frame.first_sample + frame.samples.size();
      value.tokens.reserve(static_cast<std::size_t>(result->count));
      value.token_samples.reserve(static_cast<std::size_t>(result->count));
      std::vector<std::uint64_t> segment_token_samples;
      segment_token_samples.reserve(static_cast<std::size_t>(result->count));
      for (std::int32_t i = 0; i < result->count; ++i) {
        value.tokens.emplace_back(result->tokens_arr[i] ? result->tokens_arr[i] : "");
        const auto seconds = result->start_time + result->timestamps[i];
        segment_token_samples.push_back(static_cast<std::uint64_t>(
            std::max(0.0F, seconds) * static_cast<float>(kProcessingSampleRate)));
      }

      // KeywordResult timestamps are relative to sherpa's current decoded
      // segment, not necessarily to the lifetime of OnlineStream. Preserve
      // their relative spacing and anchor the end of the keyword immediately
      // before the trailing blank that caused this trigger. This maps every
      // hit onto our absolute sample clock even after long idle periods.
      constexpr std::uint64_t output_stride = 640;
      const auto trailing_blank_samples =
          static_cast<std::uint64_t>(std::max(0, config_.num_trailing_blanks)) *
          output_stride;
      const auto anchored_end =
          value.detected_at_sample > trailing_blank_samples
              ? value.detected_at_sample - trailing_blank_samples
              : value.detected_at_sample;
      const auto segment_start = segment_token_samples.front();
      const auto segment_end = segment_token_samples.back() + output_stride;
      const auto keyword_duration = segment_end - segment_start;
      const auto anchored_start = anchored_end > keyword_duration
                                      ? anchored_end - keyword_duration
                                      : std::uint64_t{};
      for (const auto sample : segment_token_samples) {
        value.token_samples.push_back(anchored_start + sample - segment_start);
      }
      value.wake_span.start = anchored_start;
      value.wake_span.end = anchored_end;
      hit = std::move(value);
    }
    SherpaOnnxDestroyKeywordResult(result);
    if (hit) {
      // A fresh stream also discards any decoder state from the completed hit.
      replace_stream();
    }
    return hit;
  }

  void reset(std::uint64_t) override {
    replace_stream();
  }

  [[nodiscard]] bool available() const override { return true; }
  [[nodiscard]] std::string status() const override { return "sherpa-onnx KWS ready"; }

 private:
  [[nodiscard]] const SherpaOnnxOnlineStream* create_stream() const {
    const auto* stream = SherpaOnnxCreateKeywordStream(spotter_);
    if (!stream) throw std::runtime_error("SherpaOnnxCreateKeywordStream failed");
    return stream;
  }

  void replace_stream() {
    const auto* replacement = create_stream();
    const auto* previous = stream_;
    stream_ = replacement;
    if (previous) SherpaOnnxDestroyOnlineStream(previous);
  }

  KwsConfig config_;
  std::string encoder_, decoder_, joiner_, tokens_, keywords_, provider_;
  const SherpaOnnxKeywordSpotter* spotter_{};
  const SherpaOnnxOnlineStream* stream_{};
};

#endif

bool all_exist(std::initializer_list<std::filesystem::path> files) {
  return std::all_of(files.begin(), files.end(), [](const auto& path) {
    return !path.empty() && std::filesystem::is_regular_file(path);
  });
}

}  // namespace

std::unique_ptr<IVadDetector> create_vad(const VadConfig& config) {
  if (!config.enabled) return std::make_unique<DisabledVad>("VAD disabled by configuration");
  if (!std::filesystem::is_regular_file(config.model)) {
    return std::make_unique<DisabledVad>("VAD model missing: " + config.model.string());
  }
#if DVO_HAS_SHERPA
  return std::make_unique<SherpaVad>(config);
#else
  return std::make_unique<DisabledVad>("built without sherpa-onnx");
#endif
}

std::unique_ptr<IKeywordSpotter> create_keyword_spotter(const KwsConfig& config) {
  if (!config.enabled) return std::make_unique<DisabledKws>("KWS disabled by configuration");
  if (!all_exist({config.encoder, config.decoder, config.joiner, config.tokens, config.keywords})) {
    return std::make_unique<DisabledKws>("one or more KWS model/config files are missing");
  }
#if DVO_HAS_SHERPA
  return std::make_unique<SherpaKws>(config);
#else
  return std::make_unique<DisabledKws>("built without sherpa-onnx");
#endif
}

}  // namespace dvo
