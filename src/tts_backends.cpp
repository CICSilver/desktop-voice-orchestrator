#include "dvo/announcement.h"

#ifndef _WIN32
#error tts_backends.cpp must only be built on Windows
#endif

#include <Windows.h>
#include <bcrypt.h>
#include <mmsystem.h>
#include <wincrypt.h>
#include <winhttp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if DVO_HAS_SHERPA
#include "sherpa-onnx/c-api/c-api.h"
#endif

namespace dvo {
namespace {

[[nodiscard]] std::wstring utf8_to_wide(std::string_view text) {
  if (text.empty()) return {};
  const int required = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  if (required <= 0) return {};
  std::wstring out(static_cast<std::size_t>(required), L'\0');
  MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), required);
  return out;
}

[[nodiscard]] std::string wide_to_utf8(std::wstring_view text) {
  if (text.empty()) return {};
  const int required = WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
      nullptr, 0, nullptr, nullptr);
  if (required <= 0) return {};
  std::string out(static_cast<std::size_t>(required), '\0');
  WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
      out.data(), required, nullptr, nullptr);
  return out;
}

[[nodiscard]] std::string trim_ascii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return std::string(value);
}

[[nodiscard]] std::string read_env_var_trimmed(std::string_view env_name) {
  if (env_name.empty()) return {};
  const std::wstring wide_name = utf8_to_wide(env_name);
  if (wide_name.empty()) return {};
  const DWORD required = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
  if (required == 0) return {};
  std::wstring value(static_cast<std::size_t>(required), L'\0');
  const DWORD written =
      GetEnvironmentVariableW(wide_name.c_str(), value.data(), required);
  if (written == 0) return {};
  value.resize(written);
  return trim_ascii(wide_to_utf8(value));
}

[[nodiscard]] std::string compute_sha256_hex(std::string_view text) {
  BCRYPT_ALG_HANDLE alg = nullptr;
  NTSTATUS status = BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
  if (status != 0) throw std::runtime_error("BCryptOpenAlgorithmProvider failed");
  struct AlgCloser {
    BCRYPT_ALG_HANDLE h;
    ~AlgCloser() { if (h) BCryptCloseAlgorithmProvider(h, 0); }
  } alg_closer{alg};

  DWORD object_len = 0;
  DWORD hash_len = 0;
  DWORD returned = 0;
  BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH,
                    reinterpret_cast<PUCHAR>(&object_len), sizeof(object_len),
                    &returned, 0);
  BCryptGetProperty(alg, BCRYPT_HASH_LENGTH,
                    reinterpret_cast<PUCHAR>(&hash_len), sizeof(hash_len),
                    &returned, 0);

  std::vector<UCHAR> hash_obj(object_len);
  std::vector<UCHAR> hash_buf(hash_len);
  BCRYPT_HASH_HANDLE hash_handle = nullptr;
  status = BCryptCreateHash(alg, &hash_handle, hash_obj.data(), object_len, nullptr, 0, 0);
  if (status != 0) throw std::runtime_error("BCryptCreateHash failed");
  struct HashCloser {
    BCRYPT_HASH_HANDLE h;
    ~HashCloser() { if (h) BCryptDestroyHash(h); }
  } hash_closer{hash_handle};

  if (!text.empty()) {
    status = BCryptHashData(hash_handle,
                            reinterpret_cast<PUCHAR>(const_cast<char*>(text.data())),
                            static_cast<ULONG>(text.size()), 0);
    if (status != 0) throw std::runtime_error("BCryptHashData failed");
  }
  status = BCryptFinishHash(hash_handle, hash_buf.data(), hash_len, 0);
  if (status != 0) throw std::runtime_error("BCryptFinishHash failed");

  constexpr char hex_chars[] = "0123456789abcdef";
  std::string hex_str;
  hex_str.reserve(hash_len * 2);
  for (unsigned char b : hash_buf) {
    hex_str.push_back(hex_chars[b >> 4]);
    hex_str.push_back(hex_chars[b & 0x0f]);
  }
  return hex_str;
}

[[nodiscard]] std::vector<std::uint8_t> base64_decode(std::string_view b64) {
  if (b64.empty()) return {};
  DWORD binary_len = 0;
  if (!CryptStringToBinaryA(b64.data(), static_cast<DWORD>(b64.size()),
                            CRYPT_STRING_BASE64, nullptr, &binary_len,
                            nullptr, nullptr)) {
    throw std::runtime_error("failed to parse base64 audio data length");
  }
  std::vector<std::uint8_t> result(binary_len);
  if (!CryptStringToBinaryA(b64.data(), static_cast<DWORD>(b64.size()),
                            CRYPT_STRING_BASE64,
                            reinterpret_cast<BYTE*>(result.data()), &binary_len,
                            nullptr, nullptr)) {
    throw std::runtime_error("failed to decode base64 audio data");
  }
  result.resize(binary_len);
  return result;
}

[[nodiscard]] std::uint32_t parse_sample_rate_from_mime(
    std::string_view mime, std::uint32_t fallback = 24000) {
  const std::size_t pos = mime.find("rate=");
  if (pos == std::string_view::npos) return fallback;
  const std::string_view sub = mime.substr(pos + 5);
  std::uint32_t rate = 0;
  const auto res = std::from_chars(sub.data(), sub.data() + sub.size(), rate);
  if (res.ec != std::errc{} || rate == 0) return fallback;
  return rate;
}

[[nodiscard]] std::vector<std::uint8_t> wrap_pcm_to_wav(
    std::span<const std::uint8_t> pcm_data,
    std::uint32_t sample_rate,
    std::uint16_t channels = 1,
    std::uint16_t bits_per_sample = 16) {
  if (pcm_data.size() >= 12 &&
      std::memcmp(pcm_data.data(), "RIFF", 4) == 0 &&
      std::memcmp(pcm_data.data() + 8, "WAVE", 4) == 0) {
    return std::vector<std::uint8_t>(pcm_data.begin(), pcm_data.end());
  }

  const std::uint32_t data_size = static_cast<std::uint32_t>(pcm_data.size());
  const std::uint32_t riff_size = 36 + data_size;
  const std::uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
  const std::uint16_t block_align = channels * (bits_per_sample / 8);

  std::vector<std::uint8_t> wav(44 + data_size);
  std::memcpy(&wav[0], "RIFF", 4);
  std::memcpy(&wav[4], &riff_size, 4);
  std::memcpy(&wav[8], "WAVE", 4);
  std::memcpy(&wav[12], "fmt ", 4);
  const std::uint32_t fmt_size = 16;
  std::memcpy(&wav[16], &fmt_size, 4);
  const std::uint16_t audio_format = 1;  // PCM
  std::memcpy(&wav[20], &audio_format, 2);
  std::memcpy(&wav[22], &channels, 2);
  std::memcpy(&wav[24], &sample_rate, 4);
  std::memcpy(&wav[28], &byte_rate, 4);
  std::memcpy(&wav[32], &block_align, 2);
  std::memcpy(&wav[34], &bits_per_sample, 2);
  std::memcpy(&wav[36], "data", 4);
  std::memcpy(&wav[40], &data_size, 4);
  if (data_size > 0) {
    std::memcpy(&wav[44], pcm_data.data(), data_size);
  }
  return wav;
}

[[nodiscard]] std::optional<std::vector<std::uint8_t>> try_read_wav_cache(
    const std::filesystem::path& cache_file) {
  std::error_code ec;
  if (!std::filesystem::exists(cache_file, ec)) return std::nullopt;

  std::ifstream is(cache_file, std::ios::binary | std::ios::ate);
  if (!is) return std::nullopt;
  const auto file_size = is.tellg();
  if (file_size < 44) return std::nullopt;

  std::vector<std::uint8_t> buffer(static_cast<std::size_t>(file_size));
  is.seekg(0, std::ios::beg);
  if (!is.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return std::nullopt;
  }

  if (std::memcmp(buffer.data(), "RIFF", 4) != 0 ||
      std::memcmp(buffer.data() + 8, "WAVE", 4) != 0) {
    return std::nullopt;
  }
  return buffer;
}

void write_wav_cache(const std::filesystem::path& cache_dir,
                     const std::string& hash,
                     std::span<const std::uint8_t> data) {
  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);

  const auto final_path = cache_dir / (hash + ".wav");
  const auto temp_path = cache_dir / (hash + ".tmp." + std::to_string(GetCurrentProcessId()));

  {
    std::ofstream os(temp_path, std::ios::binary | std::ios::trunc);
    if (!os) return;
    os.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
    os.flush();
  }

  MoveFileExW(temp_path.c_str(), final_path.c_str(),
              MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

class ScopedWinHttpHandle {
 public:
  ScopedWinHttpHandle() = default;
  explicit ScopedWinHttpHandle(HINTERNET handle) : handle_(handle) {}
  ~ScopedWinHttpHandle() {
    if (handle_) WinHttpCloseHandle(handle_);
  }
  ScopedWinHttpHandle(const ScopedWinHttpHandle&) = delete;
  ScopedWinHttpHandle& operator=(const ScopedWinHttpHandle&) = delete;
  ScopedWinHttpHandle(ScopedWinHttpHandle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  ScopedWinHttpHandle& operator=(ScopedWinHttpHandle&& other) noexcept {
    if (this != &other) {
      if (handle_) WinHttpCloseHandle(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  [[nodiscard]] HINTERNET get() const noexcept { return handle_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return handle_ != nullptr;
  }

 private:
  HINTERNET handle_{};
};

class CancelableWinHttpRequest {
 public:
  CancelableWinHttpRequest(HINTERNET handle,
                           std::chrono::steady_clock::time_point deadline,
                           std::stop_token stop = {})
      : handle_(handle), deadline_(deadline) {
    if (handle_ != nullptr) {
      if (stop.stop_possible()) {
        stop_cb_.emplace(stop, [this] { close(); });
      }
      watchdog_ = std::jthread([this] {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_until(lock, deadline_, [this] { return done_; })) {
          timed_out_ = true;
          if (HINTERNET to_close = std::exchange(handle_, nullptr)) {
            lock.unlock();
            WinHttpCloseHandle(to_close);
          }
        }
      });
    }
  }

  ~CancelableWinHttpRequest() { close(); }

  CancelableWinHttpRequest(const CancelableWinHttpRequest&) = delete;
  CancelableWinHttpRequest& operator=(const CancelableWinHttpRequest&) = delete;

  void close() {
    HINTERNET to_close = nullptr;
    {
      std::scoped_lock lock(mutex_);
      done_ = true;
      to_close = std::exchange(handle_, nullptr);
    }
    cv_.notify_all();
    if (watchdog_.joinable() && watchdog_.get_id() != std::this_thread::get_id()) {
      watchdog_.join();
    }
    if (to_close != nullptr) {
      WinHttpCloseHandle(to_close);
    }
  }

  [[nodiscard]] HINTERNET get() const noexcept {
    std::scoped_lock lock(mutex_);
    return handle_;
  }

  [[nodiscard]] bool timed_out() const noexcept {
    std::scoped_lock lock(mutex_);
    return timed_out_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  HINTERNET handle_{};
  std::chrono::steady_clock::time_point deadline_{};
  bool done_{false};
  bool timed_out_{false};
  std::jthread watchdog_;
  std::optional<std::stop_callback<std::function<void()>>> stop_cb_;
};

struct ParsedBaseUrl {
  bool secure{false};
  std::string host;
  INTERNET_PORT port{0};
  std::string base_path;
};

[[nodiscard]] std::optional<ParsedBaseUrl> parse_base_url(std::string_view raw_url) {
  const std::string trimmed = trim_ascii(raw_url);
  std::string_view rest = trimmed;
  ParsedBaseUrl out;
  if (rest.starts_with("https://")) {
    out.secure = true;
    out.port = INTERNET_DEFAULT_HTTPS_PORT;
    rest.remove_prefix(8);
  } else if (rest.starts_with("http://")) {
    out.secure = false;
    out.port = INTERNET_DEFAULT_HTTP_PORT;
    rest.remove_prefix(7);
  } else {
    return std::nullopt;
  }

  if (rest.empty()) return std::nullopt;

  const std::size_t slash_pos = rest.find('/');
  const std::string_view authority =
      (slash_pos == std::string_view::npos) ? rest : rest.substr(0, slash_pos);
  std::string path =
      (slash_pos == std::string_view::npos) ? std::string{} : std::string(rest.substr(slash_pos));

  if (authority.empty()) return std::nullopt;

  if (authority.front() == '[') {
    const std::size_t close_bracket = authority.find(']');
    if (close_bracket == std::string_view::npos) return std::nullopt;
    out.host = std::string(authority.substr(1, close_bracket - 1));
    if (close_bracket + 1 < authority.size()) {
      if (authority[close_bracket + 1] != ':') return std::nullopt;
      const auto port_str = authority.substr(close_bracket + 2);
      unsigned int parsed_port = 0;
      const auto res = std::from_chars(
          port_str.data(), port_str.data() + port_str.size(), parsed_port);
      if (res.ec != std::errc{} || parsed_port == 0 || parsed_port > 65535) {
        return std::nullopt;
      }
      out.port = static_cast<INTERNET_PORT>(parsed_port);
    }
  } else {
    const std::size_t colon_pos = authority.rfind(':');
    if (colon_pos != std::string_view::npos) {
      out.host = std::string(authority.substr(0, colon_pos));
      const auto port_str = authority.substr(colon_pos + 1);
      unsigned int parsed_port = 0;
      const auto res = std::from_chars(
          port_str.data(), port_str.data() + port_str.size(), parsed_port);
      if (res.ec != std::errc{} || parsed_port == 0 || parsed_port > 65535) {
        return std::nullopt;
      }
      out.port = static_cast<INTERNET_PORT>(parsed_port);
    } else {
      out.host = std::string(authority);
    }
  }

  while (!path.empty() && path.back() == '/') {
    path.pop_back();
  }
  out.base_path = path;
  return out;
}

class GoogleTtsAnnouncementBackend final : public IAnnouncementBackend {
 public:
  GoogleTtsAnnouncementBackend(TtsBackendOptions options, AudioWavePlaybackFn player)
      : options_(std::move(options)), player_(std::move(player)) {}

  [[nodiscard]] const char* name() const noexcept override { return "google_tts"; }

  [[nodiscard]] std::string render_endpoint() const override {
    return options_.google_base_url + "/v1beta/models/" + options_.google_model;
  }

  void deliver(const AnnouncementRequest& request, std::stop_token stop) override {
    deliver(request, stop, nullptr);
  }

  void deliver(const AnnouncementRequest& request,
               std::stop_token stop,
               const PlaybackStartedCallback& on_playback_started) override {
    if (stop.stop_requested()) return;

    // 1. Check persistent SHA-256 disk cache
    const std::string cache_key =
        "dvo-tts-v1|" + options_.google_model + "|" + options_.google_voice + "|" + request.text;
    const std::string cache_hash = compute_sha256_hex(cache_key);
    const auto cache_file = options_.cache_dir / (cache_hash + ".wav");

    if (const auto cached = try_read_wav_cache(cache_file)) {
      if (stop.stop_requested()) return;
      if (on_playback_started) on_playback_started();
      if (player_) player_(*cached, stop);
      return;
    }

    // 2. Resolve API key from environment variable
    std::string api_key = read_env_var_trimmed(options_.google_api_key_env);
    if (api_key.empty() && options_.google_api_key_env == "GEMINI_API_KEY") {
      api_key = read_env_var_trimmed("GOOGLE_API_KEY");
    }
    if (api_key.empty()) {
      throw std::runtime_error("missing Gemini TTS API key in environment");
    }

    // 3. Parse base URL and construct endpoint
    const auto parsed_url = parse_base_url(options_.google_base_url);
    if (!parsed_url) {
      throw std::runtime_error("invalid google_base_url: " + options_.google_base_url);
    }

    std::string request_path =
        parsed_url->base_path + "/v1beta/models/" + options_.google_model + ":generateContent";

    // 4. Construct request JSON body
    nlohmann::json payload;
    payload["contents"] = nlohmann::json::array({
        {{"parts", nlohmann::json::array({{{"text", request.text}}})}}
    });
    payload["generationConfig"] = {
        {"responseModalities", nlohmann::json::array({"AUDIO"})},
        {"speechConfig", {
            {"voiceConfig", {
                {"prebuiltVoiceConfig", {
                    {"voiceName", options_.google_voice}
                }}
            }}
        }}
    };
    const std::string request_body = payload.dump();

    // 5. Send cancelable WinHTTP request
    ScopedWinHttpHandle session(WinHttpOpen(
        L"dvo-tts/0.1", WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) {
      throw std::runtime_error("WinHttpOpen failed");
    }

    const std::wstring wide_host = utf8_to_wide(parsed_url->host);
    ScopedWinHttpHandle connect(WinHttpConnect(
        session.get(), wide_host.c_str(), parsed_url->port, 0));
    if (!connect) {
      throw std::runtime_error("WinHttpConnect failed");
    }

    DWORD request_flags = parsed_url->secure ? WINHTTP_FLAG_SECURE : 0;
    const std::wstring wide_path = utf8_to_wide(request_path);
    HINTERNET raw_request = WinHttpOpenRequest(
        connect.get(), L"POST", wide_path.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, request_flags);
    if (!raw_request) {
      throw std::runtime_error("WinHttpOpenRequest failed");
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(options_.google_timeout_ms);
    CancelableWinHttpRequest http_request(raw_request, deadline, stop);

    std::wstring headers =
        L"Content-Type: application/json\r\nx-goog-api-key: " + utf8_to_wide(api_key) + L"\r\n";
    std::fill(api_key.begin(), api_key.end(), '\0');
    api_key.clear();

    const auto remaining_ms = [&]() -> std::optional<DWORD> {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return std::nullopt;
      return static_cast<DWORD>(
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    };

    if (const auto ms = remaining_ms()) {
      WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);
    } else {
      std::fill(headers.begin(), headers.end(), L'\0');
      throw std::runtime_error("TTS request exceeded deadline before send");
    }

    const BOOL sent = WinHttpSendRequest(
        http_request.get(),
        headers.c_str(),
        static_cast<DWORD>(headers.size()),
        const_cast<char*>(request_body.data()),
        static_cast<DWORD>(request_body.size()),
        static_cast<DWORD>(request_body.size()),
        0);
    std::fill(headers.begin(), headers.end(), L'\0');
    headers.clear();

    if (!sent) {
      if (stop.stop_requested()) return;
      if (http_request.timed_out()) throw std::runtime_error("TTS request timed out in WinHttpSendRequest");
      throw std::runtime_error("WinHttpSendRequest failed with error " + std::to_string(GetLastError()));
    }

    if (const auto ms = remaining_ms()) {
      WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);
    } else {
      throw std::runtime_error("TTS request exceeded deadline while sending");
    }

    if (!WinHttpReceiveResponse(http_request.get(), nullptr)) {
      if (stop.stop_requested()) return;
      if (http_request.timed_out()) throw std::runtime_error("TTS request timed out in WinHttpReceiveResponse");
      throw std::runtime_error("WinHttpReceiveResponse failed with error " + std::to_string(GetLastError()));
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    if (!WinHttpQueryHeaders(
            http_request.get(),
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX,
            &status_code,
            &status_size,
            WINHTTP_NO_HEADER_INDEX)) {
      throw std::runtime_error("WinHttpQueryHeaders failed");
    }

    std::string response_body;
    constexpr std::size_t kMaxResponseBytes = 8 * 1024 * 1024;
    for (;;) {
      const auto ms = remaining_ms();
      if (!ms) {
        throw std::runtime_error("TTS request exceeded deadline while reading response");
      }
      WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);

      DWORD available = 0;
      if (!WinHttpQueryDataAvailable(http_request.get(), &available)) {
        if (stop.stop_requested()) return;
        throw std::runtime_error("WinHttpQueryDataAvailable failed");
      }
      if (available == 0) break;

      const DWORD to_read = static_cast<DWORD>(std::min<std::size_t>(available, 16384));
      if (response_body.size() + to_read > kMaxResponseBytes) {
        throw std::runtime_error("TTS response exceeded maximum allowed size");
      }
      const std::size_t offset = response_body.size();
      response_body.resize(offset + to_read);
      DWORD downloaded = 0;
      if (!WinHttpReadData(http_request.get(), response_body.data() + offset,
                           to_read, &downloaded)) {
        if (stop.stop_requested()) return;
        throw std::runtime_error("WinHttpReadData failed");
      }
      response_body.resize(offset + downloaded);
      if (downloaded == 0) break;
    }
    http_request.close();

    if (stop.stop_requested()) return;

    if (status_code != 200) {
      throw std::runtime_error("TTS endpoint returned HTTP " + std::to_string(status_code));
    }

    // 6. Parse response JSON and extract audio bytes
    const auto envelope = nlohmann::json::parse(response_body, nullptr, false);
    if (envelope.is_discarded() || !envelope.is_object()) {
      throw std::runtime_error("TTS response is not a valid JSON object");
    }

    std::string b64_data;
    std::string mime_type;
    try {
      const auto& candidates = envelope.at("candidates");
      if (!candidates.is_array() || candidates.empty()) {
        throw std::runtime_error("missing candidates array in Gemini TTS response");
      }
      const auto& parts = candidates.at(0).at("content").at("parts");
      if (!parts.is_array() || parts.empty()) {
        throw std::runtime_error("missing parts array in Gemini TTS response");
      }
      const auto& inline_data = parts.at(0).at("inlineData");
      b64_data = inline_data.at("data").get<std::string>();
      if (inline_data.contains("mimeType") && inline_data["mimeType"].is_string()) {
        mime_type = inline_data["mimeType"].get<std::string>();
      }
    } catch (const std::exception& err) {
      throw std::runtime_error(std::string("failed to parse Gemini TTS payload: ") + err.what());
    }

    const auto decoded_bytes = base64_decode(b64_data);
    const std::uint32_t sample_rate = parse_sample_rate_from_mime(mime_type, 24000);
    const auto wav_bytes = wrap_pcm_to_wav(decoded_bytes, sample_rate);

    // Save to disk cache
    write_wav_cache(options_.cache_dir, cache_hash, wav_bytes);

    if (stop.stop_requested()) return;

    if (on_playback_started) on_playback_started();
    if (player_) player_(wav_bytes, stop);
  }

 private:
  TtsBackendOptions options_;
  AudioWavePlaybackFn player_;
};

#if DVO_HAS_SHERPA
class SherpaKokoroAnnouncementBackend final : public IAnnouncementBackend {
 public:
  SherpaKokoroAnnouncementBackend(TtsBackendOptions options, AudioWavePlaybackFn player)
      : options_(std::move(options)), player_(std::move(player)) {}

  ~SherpaKokoroAnnouncementBackend() override {
    if (tts_) {
      SherpaOnnxDestroyOfflineTts(tts_);
      tts_ = nullptr;
    }
  }

  [[nodiscard]] const char* name() const noexcept override { return "sherpa_kokoro"; }

  [[nodiscard]] std::string render_endpoint() const override {
    return options_.kokoro_model_dir.string();
  }

  void deliver(const AnnouncementRequest& request, std::stop_token stop) override {
    deliver(request, stop, nullptr);
  }

  void deliver(const AnnouncementRequest& request,
               std::stop_token stop,
               const PlaybackStartedCallback& on_playback_started) override {
    if (stop.stop_requested()) return;

    ensure_initialized();

    struct StopContext {
      std::stop_token stop;
    };
    StopContext ctx{stop};

    auto callback = [](const float* /*samples*/, int32_t /*n*/, float /*progress*/, void* arg) -> int32_t {
      auto* sc = static_cast<StopContext*>(arg);
      if (sc && sc->stop.stop_requested()) {
        return 0;
      }
      return 1;
    };

    SherpaOnnxGenerationConfig gen_cfg{};
    std::memset(&gen_cfg, 0, sizeof(gen_cfg));
    gen_cfg.sid = options_.kokoro_speaker_id;
    gen_cfg.speed = options_.kokoro_speed;
    gen_cfg.silence_scale = 0.2f;

    const SherpaOnnxGeneratedAudio* audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(tts_, request.text.c_str(), &gen_cfg, callback, &ctx);

    if (stop.stop_requested()) {
      if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
      return;
    }

    if (!audio) {
      throw std::runtime_error("SherpaOnnxOfflineTtsGenerateWithConfig returned null");
    }

    struct AudioDeleter {
      const SherpaOnnxGeneratedAudio* a;
      ~AudioDeleter() { if (a) SherpaOnnxDestroyOfflineTtsGeneratedAudio(a); }
    } audio_deleter{audio};

    if (audio->n <= 0) {
      throw std::runtime_error("sherpa-onnx generated 0 samples");
    }

    std::vector<std::uint8_t> pcm_bytes(audio->n * sizeof(std::int16_t));
    auto* out_samples = reinterpret_cast<std::int16_t*>(pcm_bytes.data());
    for (std::int32_t i = 0; i < audio->n; ++i) {
      float s = audio->samples[i];
      s = std::clamp(s, -1.0f, 1.0f);
      out_samples[i] = static_cast<std::int16_t>(s * 32767.0f);
    }

    const auto wav_bytes = wrap_pcm_to_wav(
        pcm_bytes, static_cast<std::uint32_t>(audio->sample_rate));

    if (stop.stop_requested()) return;

    if (on_playback_started) on_playback_started();
    if (player_) player_(wav_bytes, stop);
  }

 private:
  void ensure_initialized() {
    std::scoped_lock lock(init_mutex_);
    if (tts_) return;

    const auto& dir = options_.kokoro_model_dir;
    std::filesystem::path model_path = dir / "model.int8.onnx";
    if (!std::filesystem::exists(model_path)) {
      model_path = dir / "model.onnx";
      if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Kokoro model file not found in: " + dir.string());
      }
    }
    const auto voices_path = dir / "voices.bin";
    const auto tokens_path = dir / "tokens.txt";
    const auto data_dir = dir / "espeak-ng-data";
    if (!std::filesystem::exists(voices_path) ||
        !std::filesystem::exists(tokens_path) ||
        !std::filesystem::exists(data_dir)) {
      throw std::runtime_error(
          "Kokoro required files (voices.bin, tokens.txt, espeak-ng-data) not found in: " +
          dir.string());
    }

    std::string lexicon_combined;
    const auto us_en = dir / "lexicon-us-en.txt";
    const auto zh = dir / "lexicon-zh.txt";
    if (std::filesystem::exists(us_en)) {
      lexicon_combined = us_en.string();
    }
    if (std::filesystem::exists(zh)) {
      if (!lexicon_combined.empty()) lexicon_combined += ",";
      lexicon_combined += zh.string();
    }

    SherpaOnnxOfflineTtsConfig config{};
    std::memset(&config, 0, sizeof(config));
    const std::string model_str = model_path.string();
    const std::string voices_str = voices_path.string();
    const std::string tokens_str = tokens_path.string();
    const std::string data_dir_str = data_dir.string();

    config.model.kokoro.model = model_str.c_str();
    config.model.kokoro.voices = voices_str.c_str();
    config.model.kokoro.tokens = tokens_str.c_str();
    config.model.kokoro.data_dir = data_dir_str.c_str();
    if (!lexicon_combined.empty()) {
      config.model.kokoro.lexicon = lexicon_combined.c_str();
    }
    config.model.kokoro.length_scale = 1.0f;
    config.model.num_threads = options_.kokoro_num_threads;
    config.model.provider = "cpu";
    config.model.debug = 0;

    tts_ = SherpaOnnxCreateOfflineTts(&config);
    if (!tts_) {
      throw std::runtime_error("failed to create sherpa-onnx offline TTS engine");
    }
  }

  TtsBackendOptions options_;
  AudioWavePlaybackFn player_;
  std::mutex init_mutex_;
  const SherpaOnnxOfflineTts* tts_{nullptr};
};
#else
class SherpaKokoroAnnouncementBackend final : public IAnnouncementBackend {
 public:
  SherpaKokoroAnnouncementBackend(TtsBackendOptions options, AudioWavePlaybackFn player)
      : options_(std::move(options)), player_(std::move(player)) {}

  [[nodiscard]] const char* name() const noexcept override { return "sherpa_kokoro"; }

  [[nodiscard]] std::string render_endpoint() const override {
    return options_.kokoro_model_dir.string();
  }

  void deliver(const AnnouncementRequest&, std::stop_token) override {
    throw std::runtime_error("sherpa-onnx TTS is not compiled in this build");
  }

  void deliver(const AnnouncementRequest&, std::stop_token,
               const PlaybackStartedCallback&) override {
    throw std::runtime_error("sherpa-onnx TTS is not compiled in this build");
  }

 private:
  TtsBackendOptions options_;
  AudioWavePlaybackFn player_;
};
#endif

class AutoFallbackAnnouncementBackend final : public IAnnouncementBackend {
 public:
  AutoFallbackAnnouncementBackend(const TtsBackendOptions& options,
                                  AudioWavePlaybackFn player)
      : google_backend_(std::make_shared<GoogleTtsAnnouncementBackend>(options, player)),
        kokoro_backend_(std::make_shared<SherpaKokoroAnnouncementBackend>(options, player)),
        log_backend_(std::make_shared<LogAnnouncementBackend>()) {}

  [[nodiscard]] const char* name() const noexcept override { return "auto"; }

  [[nodiscard]] std::string render_endpoint() const override {
    return google_backend_->render_endpoint();
  }

  void deliver(const AnnouncementRequest& request, std::stop_token stop) override {
    deliver(request, stop, nullptr);
  }

  void deliver(const AnnouncementRequest& request,
               std::stop_token stop,
               const PlaybackStartedCallback& on_playback_started) override {
    if (stop.stop_requested()) return;

    bool started_fired = false;
    const auto guarded_on_started = [&]() {
      if (!started_fired) {
        started_fired = true;
        if (on_playback_started) on_playback_started();
      }
    };

    // 1. Try Google Gemini TTS
    bool google_succeeded = false;
    try {
      google_backend_->deliver(request, stop, guarded_on_started);
      google_succeeded = true;
    } catch (...) {
      if (started_fired || stop.stop_requested()) {
        throw;
      }
    }

    if (google_succeeded) return;

    // 2. Try Sherpa Kokoro offline TTS
    bool kokoro_succeeded = false;
    try {
      kokoro_backend_->deliver(request, stop, guarded_on_started);
      kokoro_succeeded = true;
    } catch (...) {
      if (started_fired || stop.stop_requested()) {
        throw;
      }
    }

    if (kokoro_succeeded) return;

    // 3. Fallback to Log backend (safe desktop continuity)
    log_backend_->deliver(request, stop, guarded_on_started);
  }

 private:
  std::shared_ptr<IAnnouncementBackend> google_backend_;
  std::shared_ptr<IAnnouncementBackend> kokoro_backend_;
  std::shared_ptr<IAnnouncementBackend> log_backend_;
};

}  // namespace

void play_wav_bytes_winmm(std::span<const std::uint8_t> wav_bytes, std::stop_token stop) {
  if (stop.stop_requested() || wav_bytes.empty()) return;

  if (wav_bytes.size() < 44 ||
      std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("invalid RIFF WAVE header for winmm playback");
  }

  // Parse fmt and data chunks
  std::optional<WAVEFORMATEX> format_opt;
  std::span<const std::uint8_t> pcm_data;

  std::size_t offset = 12;
  while (offset + 8 <= wav_bytes.size()) {
    char chunk_id[4];
    std::memcpy(chunk_id, wav_bytes.data() + offset, 4);
    std::uint32_t chunk_size = 0;
    std::memcpy(&chunk_size, wav_bytes.data() + offset + 4, 4);
    offset += 8;

    if (offset + chunk_size > wav_bytes.size()) break;

    if (std::memcmp(chunk_id, "fmt ", 4) == 0 && chunk_size >= 16) {
      WAVEFORMATEX wfx{};
      std::memcpy(&wfx.wFormatTag, wav_bytes.data() + offset, 2);
      std::memcpy(&wfx.nChannels, wav_bytes.data() + offset + 2, 2);
      std::memcpy(&wfx.nSamplesPerSec, wav_bytes.data() + offset + 4, 4);
      std::memcpy(&wfx.nAvgBytesPerSec, wav_bytes.data() + offset + 8, 4);
      std::memcpy(&wfx.nBlockAlign, wav_bytes.data() + offset + 12, 2);
      std::memcpy(&wfx.wBitsPerSample, wav_bytes.data() + offset + 14, 2);
      wfx.cbSize = 0;
      format_opt = wfx;
    } else if (std::memcmp(chunk_id, "data", 4) == 0) {
      pcm_data = wav_bytes.subspan(offset, chunk_size);
    }

    offset += chunk_size;
    if (chunk_size % 2 != 0) offset += 1;  // RIFF chunks are word-aligned
  }

  if (!format_opt) {
    throw std::runtime_error("missing fmt chunk in WAV data");
  }
  if (pcm_data.empty()) {
    return;
  }

  HANDLE done_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!done_event) {
    throw std::runtime_error("CreateEventW failed");
  }
  struct EventCloser {
    HANDLE h;
    ~EventCloser() { if (h) CloseHandle(h); }
  } event_closer{done_event};

  HWAVEOUT hwo = nullptr;
  MMRESULT mr = waveOutOpen(&hwo, WAVE_MAPPER, &(*format_opt),
                            reinterpret_cast<DWORD_PTR>(done_event), 0, CALLBACK_EVENT);
  if (mr != MMSYSERR_NOERROR) {
    throw std::runtime_error("waveOutOpen failed with error " + std::to_string(mr));
  }

  WAVEHDR header{};
  header.lpData = const_cast<char*>(reinterpret_cast<const char*>(pcm_data.data()));
  header.dwBufferLength = static_cast<DWORD>(pcm_data.size());

  mr = waveOutPrepareHeader(hwo, &header, sizeof(WAVEHDR));
  if (mr != MMSYSERR_NOERROR) {
    waveOutClose(hwo);
    throw std::runtime_error("waveOutPrepareHeader failed with error " + std::to_string(mr));
  }

  mr = waveOutWrite(hwo, &header, sizeof(WAVEHDR));
  if (mr != MMSYSERR_NOERROR) {
    waveOutUnprepareHeader(hwo, &header, sizeof(WAVEHDR));
    waveOutClose(hwo);
    throw std::runtime_error("waveOutWrite failed with error " + std::to_string(mr));
  }

  std::optional<std::stop_callback<std::function<void()>>> stop_cb;
  if (stop.stop_possible()) {
    stop_cb.emplace(stop, [&] {
      waveOutReset(hwo);
      SetEvent(done_event);
    });
  }

  while (!(header.dwFlags & WHDR_DONE)) {
    if (stop.stop_requested()) {
      waveOutReset(hwo);
      break;
    }
    WaitForSingleObject(done_event, 50);
  }

  if (!(header.dwFlags & WHDR_DONE)) {
    waveOutReset(hwo);
  }
  waveOutUnprepareHeader(hwo, &header, sizeof(WAVEHDR));
  waveOutClose(hwo);
}

std::shared_ptr<IAnnouncementBackend> create_announcement_backend(
    const TtsBackendOptions& options,
    AudioWavePlaybackFn player) {
  if (options.backend == "log") {
    return std::make_shared<LogAnnouncementBackend>();
  }
  if (options.backend == "google_tts") {
    return std::make_shared<GoogleTtsAnnouncementBackend>(options, std::move(player));
  }
  if (options.backend == "sherpa_kokoro") {
    return std::make_shared<SherpaKokoroAnnouncementBackend>(options, std::move(player));
  }
  if (options.backend == "auto") {
    return std::make_shared<AutoFallbackAnnouncementBackend>(options, std::move(player));
  }
  throw std::invalid_argument("unknown announcement backend: " + options.backend);
}

}  // namespace dvo
