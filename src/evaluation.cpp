#include "dvo/evaluation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "dvo/audio_types.h"
#include "dvo/inference.h"
#include "dvo/preprocessor.h"
#include "dvo/streaming_recognizer.h"
#include "dvo/wav_file.h"

namespace dvo {
namespace {

constexpr double kRate = 16000.0;
// A take window is [prompt shown, "next" pressed]. Wake detection and
// candidate end-pointing may land shortly after the user moves on.
constexpr double kSlackBefore = 0.2;
constexpr double kSlackAfter = 0.8;
// A wake hit's estimated end can trail the take: the VAD segment may include
// the key press that ended it, and a probe hit sits at the segment end. Takes
// are at least 2.5 s apart and nobody speaks within 0.1 s of a new prompt.
constexpr double kHitSlackAfter = 2.4;
// Reference decodes see a little context around the window.
constexpr double kReferencePadBefore = 0.3;
constexpr double kReferencePadAfter = 0.8;
constexpr double kEdgeProbeSeconds = 0.3;
// How much real audio a candidate re-decode adds before or after the span to
// test whether missing characters were cut off by segmentation.
constexpr double kExtensionSeconds = 0.5;
constexpr double kMusicLoopbackDbfs = -45.0;
constexpr double kSilenceDb = -120.0;

// ---------------------------------------------------------------- labels --

std::string string_field(const nlohmann::json& value, const char* key) {
  const auto found = value.find(key);
  return found != value.end() && found->is_string() ? found->get<std::string>()
                                                    : std::string{};
}

double number_field(const nlohmann::json& value, const char* key, double fallback) {
  const auto found = value.find(key);
  return found != value.end() && found->is_number() ? found->get<double>() : fallback;
}

EvaluationTake parse_take(const nlohmann::json& value, std::size_t index) {
  const auto where = "take " + std::to_string(index + 1);
  if (!value.is_object()) throw std::invalid_argument(where + " is not an object");
  EvaluationTake take;
  take.id = string_field(value, "id");
  if (take.id.empty()) throw std::invalid_argument(where + " has no id");
  take.kind = string_field(value, "kind");
  if (take.kind != "command" && take.kind != "negative" && take.kind != "silence") {
    throw std::invalid_argument(where + " has an unknown kind");
  }
  take.status = string_field(value, "status");
  if (take.status.empty()) take.status = "ok";
  if (take.status != "ok" && take.status != "discarded" && take.status != "skipped") {
    throw std::invalid_argument(where + " has an unknown status");
  }
  take.prompt = string_field(value, "prompt");
  take.text = string_field(value, "text");
  take.wake_position = string_field(value, "wake_position");
  if (take.wake_position.empty()) take.wake_position = "none";
  if (take.kind == "command" && take.wake_position != "prefix" &&
      take.wake_position != "suffix") {
    throw std::invalid_argument(where + " command needs a prefix or suffix wake word");
  }
  if (const auto actions = value.find("expected_actions");
      actions != value.end() && actions->is_array()) {
    for (const auto& action : *actions) {
      EvaluationExpectedAction expected;
      expected.type = string_field(action, "type");
      if (expected.type.empty()) throw std::invalid_argument(where + " has an action without type");
      if (const auto delta = action.find("volume_delta_percent");
          delta != action.end() && delta->is_number_integer()) {
        expected.volume_delta_percent = delta->get<int>();
      }
      take.expected_actions.push_back(std::move(expected));
    }
  }
  if (take.kind == "command" && take.expected_actions.empty()) {
    throw std::invalid_argument(where + " command has no expected actions");
  }
  const auto start_ms = number_field(value, "start_ms", -1.0);
  const auto end_ms = number_field(value, "end_ms", -1.0);
  if (take.status == "ok" && (start_ms < 0.0 || end_ms < start_ms)) {
    throw std::invalid_argument(where + " has an invalid time window");
  }
  take.start_s = std::max(0.0, start_ms) / 1000.0;
  take.end_s = std::max(0.0, end_ms) / 1000.0;
  take.attempt = static_cast<int>(number_field(value, "attempt", 1.0));
  return take;
}

// ------------------------------------------------------------------ text --

std::vector<char32_t> decode_utf8(std::string_view text) {
  std::vector<char32_t> result;
  for (std::size_t index = 0; index < text.size();) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2
                                         : (lead >> 4) == 0xE ? 3
                                         : (lead >> 3) == 0x1E ? 4 : 1;
    if (index + length > text.size()) length = 1;
    char32_t code = length == 1 ? lead
                    : length == 2 ? lead & 0x1F
                    : length == 3 ? lead & 0x0F : lead & 0x07;
    for (std::size_t next = 1; next < length; ++next) {
      code = (code << 6) | (static_cast<unsigned char>(text[index + next]) & 0x3F);
    }
    result.push_back(code);
    index += length;
  }
  return result;
}

void append_utf8(std::string& output, char32_t code) {
  if (code < 0x80) {
    output.push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    output.push_back(static_cast<char>(0xC0 | (code >> 6)));
    output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    output.push_back(static_cast<char>(0xE0 | (code >> 12)));
    output.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    output.push_back(static_cast<char>(0xF0 | (code >> 18)));
    output.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

// Letters, digits and CJK ideographs carry the words; punctuation and spacing
// differ freely between prompts and ASR output.
std::vector<char32_t> word_codepoints(std::string_view text) {
  std::vector<char32_t> result;
  for (auto code : decode_utf8(text)) {
    if (code >= U'A' && code <= U'Z') code = code - U'A' + U'a';
    if ((code >= U'a' && code <= U'z') || (code >= U'0' && code <= U'9') ||
        (code >= 0x3400 && code <= 0x4DBF) || (code >= 0x4E00 && code <= 0x9FFF)) {
      result.push_back(code);
    }
  }
  return result;
}

std::string encode(std::span<const char32_t> codes) {
  std::string result;
  for (const auto code : codes) append_utf8(result, code);
  return result;
}

// ----------------------------------------------------------------- audio --

double level_db(std::span<const float> samples) {
  if (samples.empty()) return kSilenceDb;
  double sum{};
  for (const auto sample : samples) sum += static_cast<double>(sample) * sample;
  const auto rms = std::sqrt(sum / static_cast<double>(samples.size()));
  return rms > 1e-6 ? 20.0 * std::log10(rms) : kSilenceDb;
}

std::size_t sample_at(double seconds, std::size_t size) {
  if (seconds <= 0.0) return 0;
  return std::min(size, static_cast<std::size_t>(seconds * kRate));
}

std::span<const float> slice(const std::vector<float>& signal, double from, double to) {
  const auto begin = sample_at(from, signal.size());
  const auto end = std::max(begin, sample_at(to, signal.size()));
  return std::span<const float>(signal).subspan(begin, end - begin);
}

// Frame levels over a span, in dB, using frame_seconds hops.
std::vector<double> frame_levels(std::span<const float> samples, double frame_seconds) {
  const auto frame = std::max<std::size_t>(1, static_cast<std::size_t>(frame_seconds * kRate));
  std::vector<double> levels;
  for (std::size_t offset = 0; offset + frame <= samples.size(); offset += frame) {
    levels.push_back(level_db(samples.subspan(offset, frame)));
  }
  return levels;
}

std::optional<double> percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return std::nullopt;
  std::ranges::sort(values);
  const auto index = static_cast<std::size_t>(
      std::floor(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

std::vector<float> read_mono(const std::filesystem::path& path, std::uint32_t* rate) {
  FloatWavReader reader(path);
  const auto format = reader.format();
  if (rate) *rate = format.sample_rate;
  const auto channels = std::max<std::uint16_t>(1, format.channels);
  std::vector<float> result;
  result.reserve(static_cast<std::size_t>(reader.frame_count()));
  constexpr std::uint64_t kChunk = 48000;
  for (std::uint64_t offset = 0; offset < reader.frame_count(); offset += kChunk) {
    const auto frames = reader.read_frames(offset, kChunk);
    for (std::size_t index = 0; index < frames.size(); index += channels) {
      result.push_back(frames[index]);
    }
  }
  return result;
}

// The microphone as the runtime's keyword path sees it: channel selection and
// resampling to 16 kHz through the same bypass preprocessor.
std::vector<float> microphone_16k(const std::filesystem::path& path,
                                  const AppConfig& config) {
  FloatWavReader reader(path);
  const auto format = reader.format();
  if (format.sample_rate == 0) return {};
  BypassPreprocessor preprocessor(config.audio, make_preprocessor_timeline_config(config.aec));
  const auto packet_frames = std::max<std::uint64_t>(1, format.sample_rate / 100);
  constexpr std::uint64_t kBaseQpc = 10'000'000;
  std::vector<float> result;
  std::optional<std::uint64_t> first_sample;
  NormalizedFrame frame;
  std::uint64_t sequence{};
  for (std::uint64_t offset = 0; offset < reader.frame_count(); offset += packet_frames) {
    AudioPacket packet;
    packet.stream = AudioStreamKind::microphone;
    packet.format = format;
    packet.samples = reader.read_frames(offset, packet_frames);
    packet.qpc_100ns = kBaseQpc + offset * 10'000'000ULL / format.sample_rate;
    packet.arrival_qpc_100ns = packet.qpc_100ns;
    packet.device_position = offset;
    packet.stream_epoch = 1;
    packet.sequence = ++sequence;
    (void)preprocessor.PushPacket(packet);
    while (preprocessor.TryPopFrame(frame)) {
      if (!first_sample) first_sample = frame.first_sample;
      if (frame.first_sample < *first_sample) continue;
      const auto index = static_cast<std::size_t>(frame.first_sample - *first_sample);
      if (result.size() < index + frame.samples.size()) {
        result.resize(index + frame.samples.size());
      }
      std::ranges::copy(frame.samples, result.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }
  return result;
}

bool spot_keyword(IKeywordSpotter& spotter, std::span<const float> samples) {
  spotter.reset(0);
  NormalizedFrame frame;
  for (std::size_t offset = 0; offset + kFrameSamples <= samples.size();
       offset += kFrameSamples) {
    frame.first_sample = offset;
    std::copy_n(samples.begin() + static_cast<std::ptrdiff_t>(offset), kFrameSamples,
                frame.samples.begin());
    if (spotter.accept(frame)) return true;
  }
  return false;
}

// Reference decodes use the same recognizer as the pipeline's authoritative
// exact-final pass, so "would more audio recover this character" is answered
// by the model that dropped it.
class ReferenceDecoder {
 public:
  explicit ReferenceDecoder(const AppConfig& config) {
    try {
      if (config.asr.final_decoder != "streaming") {
        offline_ = create_offline_asr_engine(
            {config.asr.final_decoder, config.asr.final_model, config.asr.final_tokens},
            config.asr.provider, config.asr.num_threads);
      } else {
        StreamingRecognizerConfig asr;
        asr.enabled = config.asr.enabled;
        asr.encoder = config.asr.encoder;
        asr.decoder = config.asr.decoder;
        asr.tokens = config.asr.tokens;
        asr.provider = config.asr.provider;
        asr.num_threads = config.asr.num_threads;
        asr.sample_rate = static_cast<std::uint32_t>(kRate);
        online_ = create_online_asr_engine(asr);
      }
    } catch (const std::exception&) {
      offline_.reset();
      online_.reset();
    }
  }

  [[nodiscard]] bool available() const { return offline_ || online_; }

  std::string decode(std::span<const float> samples) {
    if (samples.empty()) return {};
    if (offline_) return offline_->decode(samples, static_cast<std::uint32_t>(kRate)).text;
    if (!online_) return {};
    auto session = online_->create_session();
    if (!session) return {};
    (void)session->accept(samples, static_cast<std::uint32_t>(kRate));
    return session->finish().text;
  }

 private:
  std::unique_ptr<IOfflineAsrEngine> offline_;
  std::unique_ptr<IOnlineAsrEngine> online_;
};

// -------------------------------------------------------------- scoring --

nlohmann::json coverage_json(const TextCoverage& coverage) {
  return {{"expected_chars", coverage.expected_chars},
          {"matched_chars", coverage.matched_chars},
          {"head_missing", coverage.head_missing},
          {"tail_missing", coverage.tail_missing},
          {"missing_head", coverage.missing_head},
          {"missing_tail", coverage.missing_tail}};
}

nlohmann::json optional_json(const std::optional<double>& value) {
  return value ? nlohmann::json(std::round(*value * 10.0) / 10.0) : nlohmann::json(nullptr);
}

bool actions_match(const nlohmann::json& actual,
                   const std::vector<EvaluationExpectedAction>& expected) {
  if (actual.size() != expected.size()) return false;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& action = actual[index];
    if (action.value("type", "") != expected[index].type) return false;
    if (expected[index].volume_delta_percent) {
      const auto delta = action.find("volume_delta_percent");
      if (delta == action.end() || !delta->is_number_integer() ||
          delta->get<int>() != *expected[index].volume_delta_percent) {
        return false;
      }
    }
  }
  return true;
}

std::string wake_position_label(std::string_view position) {
  if (position == "prefix") return "前置";
  if (position == "suffix") return "后置";
  return "无唤醒";
}

std::string condition_label(std::string_view condition) {
  if (condition == "quiet") return "安静";
  if (condition == "music") return "播放音乐";
  return std::string(condition.empty() ? "未知环境" : condition);
}

std::string format_fixed(double value, int decimals) {
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(decimals);
  stream << value;
  return stream.str();
}

std::string ratio(std::uint64_t part, std::uint64_t total) {
  if (total == 0) return "—";
  return std::to_string(part) + "/" + std::to_string(total) + " (" +
         format_fixed(100.0 * static_cast<double>(part) / static_cast<double>(total), 0) +
         "%)";
}

std::uint64_t counter(const nlohmann::json& value, const char* key) {
  const auto found = value.find(key);
  return found != value.end() && found->is_number_unsigned() ? found->get<std::uint64_t>() : 0;
}

void add_counters(nlohmann::json& total, const nlohmann::json& part) {
  for (const auto& [key, value] : part.items()) {
    if (value.is_number_unsigned()) {
      total[key] = counter(total, key.c_str()) + value.get<std::uint64_t>();
    } else if (value.is_number_float()) {
      total[key] = total.value(key, 0.0) + value.get<double>();
    } else if (value.is_object()) {
      if (!total.contains(key)) total[key] = nlohmann::json::object();
      add_counters(total[key], value);
    }
  }
}

}  // namespace

EvaluationLabels parse_evaluation_labels(const nlohmann::json& value) {
  if (!value.is_object() || string_field(value, "schema") != "dvo-eval-labels") {
    throw std::invalid_argument("not a dvo-eval-labels document");
  }
  if (number_field(value, "schema_version", 0.0) != 1.0) {
    throw std::invalid_argument("unsupported labels schema_version");
  }
  EvaluationLabels labels;
  labels.collection_id = string_field(value, "collection_id");
  if (const auto group = value.find("group"); group != value.end() && group->is_object()) {
    labels.group_id = string_field(*group, "id");
    labels.group_title = string_field(*group, "title");
    labels.condition = string_field(*group, "condition");
    labels.distance_m = number_field(*group, "distance_m", 0.0);
  }
  labels.speaker = string_field(value, "speaker");
  labels.wake_word = string_field(value, "wake_word");
  labels.complete = value.value("complete", false);
  const auto takes = value.find("takes");
  if (takes == value.end() || !takes->is_array()) {
    throw std::invalid_argument("labels have no takes array");
  }
  for (std::size_t index = 0; index < takes->size(); ++index) {
    labels.takes.push_back(parse_take((*takes)[index], index));
  }
  std::ranges::stable_sort(labels.takes, std::less{}, &EvaluationTake::start_s);
  return labels;
}

EvaluationLabels load_evaluation_labels(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open " + path.string());
  return parse_evaluation_labels(nlohmann::json::parse(input));
}

TextCoverage text_coverage(std::string_view expected, std::string_view recognized) {
  const auto want = word_codepoints(expected);
  const auto got = word_codepoints(recognized);
  TextCoverage coverage;
  coverage.expected_chars = want.size();
  if (want.empty()) return coverage;

  const auto columns = got.size() + 1;
  std::vector<std::size_t> table((want.size() + 1) * columns);
  const auto at = [&](std::size_t row, std::size_t column) -> std::size_t& {
    return table[row * columns + column];
  };
  for (std::size_t row = 1; row <= want.size(); ++row) {
    for (std::size_t column = 1; column <= got.size(); ++column) {
      at(row, column) = want[row - 1] == got[column - 1]
                            ? at(row - 1, column - 1) + 1
                            : std::max(at(row - 1, column), at(row, column - 1));
    }
  }
  std::vector<bool> matched(want.size());
  for (std::size_t row = want.size(), column = got.size(); row > 0 && column > 0;) {
    if (want[row - 1] == got[column - 1]) {
      matched[row - 1] = true;
      --row;
      --column;
    } else if (at(row - 1, column) >= at(row, column - 1)) {
      --row;
    } else {
      --column;
    }
  }
  coverage.matched_chars =
      static_cast<std::size_t>(std::count(matched.begin(), matched.end(), true));
  if (coverage.matched_chars == 0) {
    coverage.head_missing = want.size();
    coverage.tail_missing = want.size();
    coverage.missing_head = encode(want);
    coverage.missing_tail = encode(want);
    return coverage;
  }
  const auto first =
      static_cast<std::size_t>(std::find(matched.begin(), matched.end(), true) - matched.begin());
  const auto last = want.size() - 1 -
                    static_cast<std::size_t>(std::find(matched.rbegin(), matched.rend(), true) -
                                             matched.rbegin());
  coverage.head_missing = first;
  coverage.tail_missing = want.size() - 1 - last;
  coverage.missing_head = encode(std::span(want).first(first));
  coverage.missing_tail = encode(std::span(want).subspan(last + 1));
  return coverage;
}

SessionReference compute_session_reference(const std::filesystem::path& session,
                                           const EvaluationLabels& labels,
                                           const AppConfig& config,
                                           const nlohmann::json& utterance_summary) {
  SessionReference reference;
  reference.takes.resize(labels.takes.size());

  const auto microphone = microphone_16k(session / "mic.wav", config);
  std::vector<float> processed;
  if (std::filesystem::is_regular_file(session / "processed.wav")) {
    std::uint32_t rate{};
    processed = read_mono(session / "processed.wav", &rate);
    if (rate != static_cast<std::uint32_t>(kRate)) processed.clear();
  }
  reference.processed_audio = !processed.empty();
  std::vector<float> loopback;
  std::uint32_t loopback_rate{};
  if (std::filesystem::is_regular_file(session / "loopback.wav")) {
    loopback = read_mono(session / "loopback.wav", &loopback_rate);
  }

  const auto spotter = create_keyword_spotter(config.kws);
  const bool kws_ready = spotter && spotter->available();
  ReferenceDecoder decoder(config);

  for (std::size_t index = 0; index < labels.takes.size(); ++index) {
    const auto& take = labels.takes[index];
    auto& result = reference.takes[index];
    if (take.status != "ok") continue;
    const auto from = take.start_s - kReferencePadBefore;
    const auto to = take.end_s + kReferencePadAfter;
    result.available = true;
    if (kws_ready) {
      result.kws_microphone = spot_keyword(*spotter, slice(microphone, from, to));
      if (!processed.empty()) {
        result.kws_processed = spot_keyword(*spotter, slice(processed, from, to));
      }
    }
    if (decoder.available()) {
      result.asr_microphone = decoder.decode(slice(microphone, from, to));
      if (!processed.empty()) {
        result.asr_processed = decoder.decode(slice(processed, from, to));
      }
    }
    result.speech_dbfs =
        percentile(frame_levels(slice(microphone, take.start_s, take.end_s), 0.1), 0.95);
    if (!loopback.empty() && loopback_rate != 0) {
      const auto begin = std::min(loopback.size(),
                                  static_cast<std::size_t>(take.start_s * loopback_rate));
      const auto end = std::min(loopback.size(),
                                std::max(begin, static_cast<std::size_t>(take.end_s * loopback_rate)));
      result.loopback_dbfs =
          level_db(std::span<const float>(loopback).subspan(begin, end - begin));
    }
  }

  for (const auto& utterance :
       utterance_summary.value("utterances", nlohmann::json::array())) {
    const auto candidate = utterance.find("candidate");
    if (candidate == utterance.end() || !candidate->is_object()) continue;
    // Re-decode from the stream the pipeline actually recognized.
    const bool microphone_source =
        processed.empty() || string_field(utterance, "audio_source") == "microphone";
    const auto& signal = microphone_source ? microphone : processed;
    const auto start = number_field(*candidate, "start_s", 0.0);
    const auto end = number_field(*candidate, "end_s", 0.0);
    CandidateReference result;
    if (const auto speech = percentile(frame_levels(slice(signal, start, end), 0.05), 0.95)) {
      const auto tail = frame_levels(slice(signal, end, end + kEdgeProbeSeconds), 0.05);
      if (!tail.empty()) result.tail_db = *std::ranges::max_element(tail) - *speech;
      if (start >= kEdgeProbeSeconds) {
        const auto head = frame_levels(slice(signal, start - kEdgeProbeSeconds, start), 0.05);
        if (!head.empty()) result.head_db = *std::ranges::max_element(head) - *speech;
      }
    }
    if (decoder.available()) {
      result.text = decoder.decode(slice(signal, start, end));
      result.text_head_extended =
          decoder.decode(slice(signal, start - kExtensionSeconds, end));
      result.text_tail_extended =
          decoder.decode(slice(signal, start, end + kExtensionSeconds));
    }
    reference.candidates[string_field(utterance, "utterance_id")] = std::move(result);
  }
  return reference;
}

nlohmann::json evaluate_labeled_session(const EvaluationLabels& labels,
                                        const nlohmann::json& utterance_summary,
                                        const SessionReference& reference) {
  const auto hits = utterance_summary.value("keyword_hits", nlohmann::json::array());
  const auto utterances = utterance_summary.value("utterances", nlohmann::json::array());
  const auto& takes = labels.takes;

  // Assign each keyword hit and candidate to one take. Discarded takes still
  // own their window so a retake's events are not blamed on its neighbours.
  const auto take_for_time = [&](double time) -> std::optional<std::size_t> {
    for (std::size_t index = 0; index < takes.size(); ++index) {
      if (time >= takes[index].start_s && time <= takes[index].end_s) return index;
    }
    std::optional<std::size_t> best;
    double best_distance = std::numeric_limits<double>::max();
    for (std::size_t index = 0; index < takes.size(); ++index) {
      const auto& take = takes[index];
      if (time < take.start_s - kSlackBefore || time > take.end_s + kHitSlackAfter) continue;
      const auto distance = time < take.start_s ? take.start_s - time : time - take.end_s;
      if (distance < best_distance) {
        best_distance = distance;
        best = index;
      }
    }
    return best;
  };
  std::vector<std::vector<nlohmann::json>> take_hits(takes.size());
  std::uint64_t unassigned_hits{};
  for (const auto& hit : hits) {
    if (const auto index = take_for_time(number_field(hit, "wake_end_s", -1.0))) {
      take_hits[*index].push_back(hit);
    } else if (hit["suppressed"].is_null()) {
      ++unassigned_hits;
    }
  }
  std::vector<std::vector<nlohmann::json>> take_utterances(takes.size());
  for (const auto& utterance : utterances) {
    const auto candidate = utterance.find("candidate");
    if (candidate == utterance.end() || !candidate->is_object()) continue;
    const auto start = number_field(*candidate, "start_s", 0.0);
    const auto end = number_field(*candidate, "end_s", 0.0);
    std::optional<std::size_t> best;
    double best_overlap{};
    for (std::size_t index = 0; index < takes.size(); ++index) {
      const auto overlap = std::min(end, takes[index].end_s + kSlackAfter) -
                           std::max(start, takes[index].start_s - kSlackBefore);
      if (overlap > best_overlap) {
        best_overlap = overlap;
        best = index;
      }
    }
    if (best) take_utterances[*best].push_back(utterance);
  }

  nlohmann::json commands = {{"total", 0U}, {"wake_detected", 0U}, {"wake_suppressed", 0U},
                             {"wake_missed_pipeline", 0U}, {"wake_missed_model", 0U},
                             {"reference_kws_microphone", 0U}, {"reference_kws_processed", 0U},
                             {"correct", 0U}, {"wrong_actions", 0U}, {"rejected", 0U},
                             {"no_asr", 0U}, {"candidate_rejected", 0U}, {"no_candidate", 0U},
                             {"no_wake", 0U}, {"tail_cut", 0U}, {"head_cut", 0U},
                             {"followup_covered", 0U},
                             {"asr_drop", 0U},
                             {"music_takes", 0U}};
  nlohmann::json by_position = nlohmann::json::object();
  nlohmann::json negatives = {{"total", 0U}, {"false_wakes", 0U}, {"false_plans", 0U},
                              {"followup_plans", 0U},
                              {"duration_s", 0.0}};
  auto results = nlohmann::json::array();

  for (std::size_t index = 0; index < takes.size(); ++index) {
    const auto& take = takes[index];
    const auto& ref = index < reference.takes.size() ? reference.takes[index] : TakeReference{};
    auto expected = nlohmann::json::array();
    for (const auto& action : take.expected_actions) {
      expected.push_back({{"type", action.type},
                          {"volume_delta_percent",
                           action.volume_delta_percent
                               ? nlohmann::json(*action.volume_delta_percent)
                               : nlohmann::json(nullptr)}});
    }
    nlohmann::json result = {{"id", take.id}, {"kind", take.kind}, {"status", take.status},
                             {"attempt", take.attempt}, {"prompt", take.prompt},
                             {"text", take.text}, {"wake_position", take.wake_position},
                             {"start_s", take.start_s}, {"end_s", take.end_s},
                             {"expected_actions", std::move(expected)}};
    if (take.status != "ok") {
      results.push_back(std::move(result));
      continue;
    }

    bool detected{};
    bool suppressed{};
    auto detectors = nlohmann::json::array();
    for (const auto& hit : take_hits[index]) {
      if (hit["suppressed"].is_null()) {
        detected = true;
        detectors.push_back(hit.value("detector", "aec"));
      } else {
        suppressed = true;
      }
    }
    auto actions = nlohmann::json::array();
    auto utterance_results = nlohmann::json::array();
    std::string recognized;
    bool any_rejected{}, any_no_asr{}, any_candidate_rejected{};
    for (auto utterance : take_utterances[index]) {
      for (const auto& action : utterance["actions"]) actions.push_back(action);
      const auto outcome = utterance.value("outcome", "");
      any_rejected = any_rejected || outcome == "rejected";
      any_no_asr = any_no_asr || outcome == "no_asr" || outcome == "asr_error" ||
                   outcome == "asr_cancelled";
      any_candidate_rejected = any_candidate_rejected || outcome == "candidate_rejected";
      if (utterance["asr_text"].is_string()) recognized += utterance["asr_text"].get<std::string>();
      if (const auto found = reference.candidates.find(string_field(utterance, "utterance_id"));
          found != reference.candidates.end()) {
        const auto optional_text = [](const std::optional<std::string>& text) {
          return text ? nlohmann::json(*text) : nlohmann::json(nullptr);
        };
        utterance["reference"] = {
            {"head_db", optional_json(found->second.head_db)},
            {"tail_db", optional_json(found->second.tail_db)},
            {"text", optional_text(found->second.text)},
            {"text_head_extended", optional_text(found->second.text_head_extended)},
            {"text_tail_extended", optional_text(found->second.text_tail_extended)}};
      }
      utterance_results.push_back(std::move(utterance));
    }
    const bool music = ref.loopback_dbfs && *ref.loopback_dbfs > kMusicLoopbackDbfs;
    result["reference"] = {{"available", ref.available},
                           {"kws_microphone", ref.kws_microphone},
                           {"kws_processed", ref.kws_processed},
                           {"asr_microphone", ref.asr_microphone},
                           {"asr_processed", ref.asr_processed},
                           {"speech_dbfs", optional_json(ref.speech_dbfs)},
                           {"loopback_dbfs", optional_json(ref.loopback_dbfs)},
                           {"music_detected", music}};
    result["utterances"] = utterance_results;
    result["actions"] = actions;

    if (take.kind != "command") {
      negatives["total"] = counter(negatives, "total") + 1;
      negatives["duration_s"] = negatives.value("duration_s", 0.0) + (take.end_s - take.start_s);
      // Speech inside a still-open activation window is a follow-up turn by
      // design; only a plan that needed a wake word here is a false execution.
      bool keyword_plan{};
      bool followup_plan{};
      for (const auto& utterance : utterance_results) {
        if (utterance.value("outcome", "") != "plan") continue;
        (utterance.value("origin", "") == "followup" ? followup_plan : keyword_plan) = true;
      }
      if (detected) negatives["false_wakes"] = counter(negatives, "false_wakes") + 1;
      if (keyword_plan) negatives["false_plans"] = counter(negatives, "false_plans") + 1;
      if (followup_plan && !keyword_plan) {
        negatives["followup_plans"] = counter(negatives, "followup_plans") + 1;
      }
      result["wake"] = {{"detected", detected}, {"detectors", detectors},
                        {"verdict", detected ? "false_wake" : "none"}};
      result["verdict"] = keyword_plan    ? "false_plan"
                          : detected      ? "false_wake"
                          : followup_plan ? "followup_in_window"
                                          : "ok";
      results.push_back(std::move(result));
      continue;
    }

    auto& position = by_position[take.wake_position];
    if (position.is_null()) {
      position = {{"total", 0U}, {"wake_detected", 0U}, {"correct", 0U}};
    }
    position["total"] = counter(position, "total") + 1;
    commands["total"] = counter(commands, "total") + 1;
    if (music) commands["music_takes"] = counter(commands, "music_takes") + 1;
    if (ref.kws_microphone) {
      commands["reference_kws_microphone"] = counter(commands, "reference_kws_microphone") + 1;
    }
    if (ref.kws_processed) {
      commands["reference_kws_processed"] = counter(commands, "reference_kws_processed") + 1;
    }
    // A command said while the previous command's activation window was still
    // open needs no wake word; it is not a wake miss.
    const bool followup_covered =
        !detected && !utterance_results.empty() &&
        std::ranges::all_of(utterance_results, [](const nlohmann::json& utterance) {
          return utterance.value("origin", "") == "followup";
        });
    std::string wake_verdict;
    if (detected) {
      wake_verdict = "detected";
      commands["wake_detected"] = counter(commands, "wake_detected") + 1;
      position["wake_detected"] = counter(position, "wake_detected") + 1;
    } else if (followup_covered) {
      wake_verdict = "followup";
      commands["followup_covered"] = counter(commands, "followup_covered") + 1;
      position["followup_covered"] = counter(position, "followup_covered") + 1;
    } else if (suppressed) {
      wake_verdict = "suppressed";
      commands["wake_suppressed"] = counter(commands, "wake_suppressed") + 1;
    } else if (ref.kws_microphone || ref.kws_processed) {
      wake_verdict = "missed_pipeline";
      commands["wake_missed_pipeline"] = counter(commands, "wake_missed_pipeline") + 1;
    } else {
      wake_verdict = "missed_model";
      commands["wake_missed_model"] = counter(commands, "wake_missed_model") + 1;
    }
    result["wake"] = {{"detected", detected}, {"suppressed", suppressed},
                      {"detectors", detectors}, {"verdict", wake_verdict}};

    std::string verdict;
    if (actions_match(actions, take.expected_actions)) {
      verdict = "correct";
    } else if (!actions.empty()) {
      verdict = "wrong_actions";
    } else if (utterance_results.empty()) {
      verdict = detected ? "no_candidate" : "no_wake";
    } else if (any_rejected) {
      verdict = "rejected";
    } else if (any_no_asr) {
      verdict = "no_asr";
    } else if (any_candidate_rejected) {
      verdict = "candidate_rejected";
    } else {
      verdict = "no_candidate";
    }
    commands[verdict] = counter(commands, verdict.c_str()) + 1;
    if (verdict == "correct") position["correct"] = counter(position, "correct") + 1;
    result["verdict"] = verdict;

    // Missing characters at an edge count as a segmentation cut only when
    // re-decoding that candidate with more real audio on that side recovers
    // them. Otherwise the recognizer dropped them (the streaming model is
    // sensitive to where decoding starts, so a whole-window decode hearing a
    // character is not evidence of a cut).
    nlohmann::json truncation = {{"verdict", "unknown"}};
    if (!recognized.empty()) {
      const auto pipeline = text_coverage(take.text, recognized);
      std::vector<std::string> texts;
      for (const auto& utterance : utterance_results) {
        texts.push_back(utterance["asr_text"].is_string()
                            ? utterance["asr_text"].get<std::string>()
                            : std::string{});
      }
      const auto joined = [&](std::size_t replaced, const std::string& text) {
        std::string value;
        for (std::size_t index = 0; index < texts.size(); ++index) {
          value += index == replaced ? text : texts[index];
        }
        return value;
      };
      const auto reference_of = [&](const nlohmann::json& utterance) -> const CandidateReference* {
        const auto found = reference.candidates.find(string_field(utterance, "utterance_id"));
        return found == reference.candidates.end() ? nullptr : &found->second;
      };
      const auto* first = texts.empty() ? nullptr : reference_of(utterance_results.front());
      const auto* last = texts.empty() ? nullptr : reference_of(utterance_results.back());
      const bool head_cut =
          pipeline.head_missing > 0 && first && first->text && first->text_head_extended &&
          text_coverage(take.text, joined(0, *first->text_head_extended)).head_missing <
              text_coverage(take.text, joined(0, *first->text)).head_missing;
      const bool tail_cut =
          pipeline.tail_missing > 0 && last && last->text && last->text_tail_extended &&
          text_coverage(take.text, joined(texts.size() - 1, *last->text_tail_extended)).tail_missing <
              text_coverage(take.text, joined(texts.size() - 1, *last->text)).tail_missing;
      const bool dropped = !head_cut && !tail_cut &&
                           (pipeline.head_missing > 0 || pipeline.tail_missing > 0);
      truncation = {{"verdict", tail_cut && head_cut ? "both"
                                : tail_cut           ? "tail_cut"
                                : head_cut           ? "head_cut"
                                : dropped            ? "asr_drop"
                                                     : "none"},
                    {"pipeline", coverage_json(pipeline)},
                    {"head_db", first ? optional_json(first->head_db) : nullptr},
                    {"tail_db", last ? optional_json(last->tail_db) : nullptr}};
      if (tail_cut) commands["tail_cut"] = counter(commands, "tail_cut") + 1;
      if (head_cut) commands["head_cut"] = counter(commands, "head_cut") + 1;
      if (dropped) commands["asr_drop"] = counter(commands, "asr_drop") + 1;
    }
    result["truncation"] = std::move(truncation);
    results.push_back(std::move(result));
  }

  return {{"session",
           {{"collection_id", labels.collection_id},
            {"group_id", labels.group_id},
            {"group_title", labels.group_title},
            {"condition", labels.condition},
            {"distance_m", labels.distance_m},
            {"speaker", labels.speaker},
            {"wake_word", labels.wake_word},
            {"labels_complete", labels.complete},
            {"reference_processed_audio", reference.processed_audio}}},
          {"summary",
           {{"commands", std::move(commands)},
            {"by_position", std::move(by_position)},
            {"negatives", std::move(negatives)},
            {"unassigned_keyword_hits", unassigned_hits}}},
          {"takes", std::move(results)}};
}

nlohmann::json aggregate_evaluations(const std::vector<nlohmann::json>& session_reports) {
  nlohmann::json overall = nlohmann::json::object();
  std::map<std::tuple<std::string, double>, nlohmann::json> groups;
  for (const auto& report : session_reports) {
    const auto& session = report["session"];
    const auto& summary = report["summary"];
    add_counters(overall, summary);
    overall["sessions"] = counter(overall, "sessions") + 1;
    auto& group = groups[{session.value("condition", ""), session.value("distance_m", 0.0)}];
    if (group.is_null()) {
      group = {{"condition", session.value("condition", "")},
               {"distance_m", session.value("distance_m", 0.0)}};
    }
    add_counters(group, summary);
    group["sessions"] = counter(group, "sessions") + 1;
  }
  auto group_list = nlohmann::json::array();
  for (auto& [key, group] : groups) group_list.push_back(std::move(group));
  return {{"overall", std::move(overall)}, {"groups", std::move(group_list)}};
}

std::string format_evaluation_summary(const std::vector<nlohmann::json>& session_reports,
                                      const nlohmann::json& aggregate) {
  std::ostringstream out;
  const auto describe = [&](const nlohmann::json& block) {
    const auto& commands = block.value("commands", nlohmann::json::object());
    const auto& positions = block.value("by_position", nlohmann::json::object());
    const auto& negatives = block.value("negatives", nlohmann::json::object());
    const auto total = counter(commands, "total");
    // Recall counts only takes that needed a wake word.
    out << "    唤醒召回    "
        << ratio(counter(commands, "wake_detected"),
                 total - counter(commands, "followup_covered"));
    for (const char* position : {"prefix", "suffix"}) {
      if (!positions.contains(position)) continue;
      const auto& part = positions[position];
      out << "   " << wake_position_label(position) << " "
          << ratio(counter(part, "wake_detected"),
                   counter(part, "total") - counter(part, "followup_covered"));
    }
    out << "   （另有 " << counter(commands, "followup_covered")
        << " 条落在上一条的激活窗口内，无需唤醒）";
    out << "\n    离线参考    麦克风 " << ratio(counter(commands, "reference_kws_microphone"), total)
        << "   AEC后 " << ratio(counter(commands, "reference_kws_processed"), total) << "\n";
    out << "    漏唤醒原因  流水线 " << counter(commands, "wake_missed_pipeline")
        << "   模型 " << counter(commands, "wake_missed_model")
        << "   被抑制 " << counter(commands, "wake_suppressed") << "\n";
    out << "    命令正确    " << ratio(counter(commands, "correct"), total);
    for (const char* position : {"prefix", "suffix"}) {
      if (!positions.contains(position)) continue;
      const auto& part = positions[position];
      out << "   " << wake_position_label(position) << " "
          << ratio(counter(part, "correct"), counter(part, "total"));
    }
    out << "\n    失败分布    解析拒绝 " << counter(commands, "rejected")
        << "   动作不符 " << counter(commands, "wrong_actions")
        << "   无识别结果 " << counter(commands, "no_asr")
        << "   无候选 " << counter(commands, "no_candidate") + counter(commands, "candidate_rejected")
        << "   未唤醒 " << counter(commands, "no_wake") << "\n";
    out << "    缺字        切分截断（尾/头） " << counter(commands, "tail_cut") << "/"
        << counter(commands, "head_cut") << "   模型丢字（延长也补不回） "
        << counter(commands, "asr_drop") << "\n";
    out << "    误唤醒      " << counter(negatives, "false_wakes") << " 次 / "
        << counter(negatives, "total") << " 条负样本（"
        << format_fixed(negatives.value("duration_s", 0.0), 0) << " 秒）   误执行 "
        << counter(negatives, "false_plans") << "   （激活窗口内按后续话语执行 "
        << counter(negatives, "followup_plans") << "）\n";
  };

  out << "================ 评估汇总 ================\n";
  for (const auto& group : aggregate.value("groups", nlohmann::json::array())) {
    out << "[" << condition_label(group.value("condition", "")) << " · "
        << format_fixed(group.value("distance_m", 0.0), 1) << " 米]  会话 "
        << counter(group, "sessions") << "\n";
    describe(group);
  }
  if (aggregate.value("groups", nlohmann::json::array()).size() > 1) {
    out << "[全部]\n";
    describe(aggregate["overall"]);
  }

  out << "\n================ 失败明细 ================\n";
  static const std::map<std::string, std::string> verdicts{
      {"wrong_actions", "动作不符"}, {"rejected", "解析拒绝"}, {"no_asr", "无识别结果"},
      {"candidate_rejected", "候选被丢弃"}, {"no_candidate", "唤醒了但没有候选"},
      {"no_wake", "未唤醒"}, {"false_wake", "误唤醒"}, {"false_plan", "误执行"},
      {"followup_in_window", "落在上一条的连续激活窗口内，按后续话语执行"}};
  static const std::map<std::string, std::string> wakes{
      {"missed_pipeline", "漏唤醒（离线参考能检出 → 流水线问题）"},
      {"missed_model", "漏唤醒（离线参考也检不出 → 模型/声学问题）"},
      {"suppressed", "唤醒被抑制"}};
  for (const auto& report : session_reports) {
    const auto& session = report["session"];
    bool header{};
    for (const auto& take : report["takes"]) {
      if (take.value("status", "") != "ok") continue;
      const auto verdict = take.value("verdict", "");
      const auto truncation = take.contains("truncation")
                                  ? take["truncation"].value("verdict", "none")
                                  : std::string("none");
      const bool cut = truncation == "tail_cut" || truncation == "head_cut" || truncation == "both";
      const bool dropped = truncation == "asr_drop";
      if ((verdict == "correct" || verdict == "ok") && !cut && !dropped) continue;
      if (!header) {
        out << session.value("group_title", session.value("group_id", "")) << "  ("
            << session.value("speaker", "") << ")\n";
        header = true;
      }
      out << "  ✗ " << take.value("id", "") << " [" << wake_position_label(take.value("wake_position", ""))
          << "] 「" << take.value("prompt", "") << "」\n     ";
      const auto wake = take.value("wake", nlohmann::json::object()).value("verdict", "");
      if (const auto found = wakes.find(wake); found != wakes.end()) out << found->second << "；";
      if (const auto found = verdicts.find(verdict); found != verdicts.end()) {
        out << found->second;
      } else if (verdict == "correct") {
        out << "命令正确";
      }
      std::string recognized;
      std::string rejection;
      for (const auto& utterance : take.value("utterances", nlohmann::json::array())) {
        if (utterance["asr_text"].is_string()) recognized += utterance["asr_text"].get<std::string>();
        if (utterance["rejection"].is_object() && rejection.empty()) {
          rejection = utterance["rejection"].value("reason", "");
        }
      }
      if (!recognized.empty()) out << "；识别「" << recognized << "」";
      if (!rejection.empty()) out << "；原因：" << rejection;
      if (cut || dropped) {
        const auto& coverage = take["truncation"]["pipeline"];
        const auto missing = coverage.value("missing_tail", "").empty()
                                 ? coverage.value("missing_head", "")
                                 : coverage.value("missing_tail", "");
        if (dropped) {
          out << "；模型丢字（缺「" << missing << "」，延长候选也补不回）";
        } else {
          if (truncation != "head_cut") out << "；尾部截断（缺「" << coverage.value("missing_tail", "") << "」";
          else out << "；头部截断（缺「" << coverage.value("missing_head", "") << "」";
          const auto& db = take["truncation"][truncation == "head_cut" ? "head_db" : "tail_db"];
          if (db.is_number()) out << "，边界外电平 " << format_fixed(db.get<double>(), 0) << " dB";
          out << "，延长后可补回）";
        }
      }
      const auto& ref = take.value("reference", nlohmann::json::object());
      if (!ref.value("asr_processed", "").empty() || !ref.value("asr_microphone", "").empty()) {
        out << "\n     参考识别：AEC后「" << ref.value("asr_processed", "") << "」 麦克风「"
            << ref.value("asr_microphone", "") << "」";
      }
      if (ref.contains("speech_dbfs") && ref["speech_dbfs"].is_number()) {
        out << "  语音电平 " << format_fixed(ref["speech_dbfs"].get<double>(), 0) << " dBFS";
      }
      out << "\n";
    }
  }
  return out.str();
}

}  // namespace dvo
