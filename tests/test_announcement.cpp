#include <Windows.h>
#include <wincrypt.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "dvo/action_executor.h"
#include "dvo/announcement.h"
#include "dvo/command_parser.h"

namespace {

dvo::AnnouncementRequest request(std::string id,
                                 dvo::ExecutionSource source = dvo::ExecutionSource::live) {
  dvo::AnnouncementRequest value;
  value.announcement_id = std::move(id);
  value.text = "正在执行：暂停音乐";
  value.source = source;
  value.origin = dvo::UtteranceOrigin::followup;
  value.activation_id = "activation-7";
  value.turn_index = 2;
  value.trigger_sample = 15'500;
  value.utterance_id = "utterance-8";
  value.command_id = "command-9";
  value.action_ids = {"action-10"};
  value.timestamp_sample = 16'000;
  return value;
}

class BlockingAnnouncementBackend final : public dvo::IAnnouncementBackend {
 public:
  [[nodiscard]] const char* name() const noexcept override { return "blocking"; }

  void deliver(const dvo::AnnouncementRequest& value,
               std::stop_token stop) override {
    std::unique_lock lock(mutex);
    delivered.push_back(value.announcement_id);
    entered = true;
    entered_cv.notify_all();
    release_cv.wait(lock, stop, [this] { return released; });
  }

  bool wait_entered(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return entered_cv.wait_for(lock, timeout, [this] { return entered; });
  }

  void release() {
    {
      std::scoped_lock lock(mutex);
      released = true;
    }
    release_cv.notify_all();
  }

  std::size_t count() const {
    std::scoped_lock lock(mutex);
    return delivered.size();
  }

  mutable std::mutex mutex;
  std::condition_variable_any entered_cv;
  std::condition_variable_any release_cv;
  std::vector<std::string> delivered;
  bool entered{};
  bool released{};
};

class ThrowOnceAnnouncementBackend final : public dvo::IAnnouncementBackend {
 public:
  [[nodiscard]] const char* name() const noexcept override { return "throw-once"; }

  void deliver(const dvo::AnnouncementRequest&, std::stop_token) override {
    ++calls;
    if (throw_next.exchange(false)) throw std::runtime_error("injected failure");
  }

  std::atomic<std::size_t> calls{};
  std::atomic<bool> throw_next{true};
};

class CountingActionBackend final : public dvo::IActionBackend {
 public:
  dvo::ActionResult execute(const dvo::PlannedAction& action,
                            std::stop_token) override {
    ++calls;
    dvo::ActionResult result;
    result.type = action.type;
    result.status = dvo::ActionStatus::succeeded;
    return result;
  }

  std::atomic<std::size_t> calls{};
};

dvo::CommandPlan command_plan(std::string utterance_id) {
  dvo::CommandParseContext context;
  context.runtime_session_id = "announcement-test-runtime";
  context.utterance_id = std::move(utterance_id);
  context.timestamp_sample = 32000;
  const auto parsed = dvo::CommandParser{}.parse("暂停音乐", context);
  REQUIRE(parsed.ok());
  return *parsed.plan;
}

}  // namespace

TEST_CASE("Action summaries are deterministic and use structured actions") {
  std::vector<dvo::PlannedAction> actions{
      {"play", 1, dvo::ActionType::media_play, {}},
      {"pause", 2, dvo::ActionType::media_pause, {}},
      {"up", 3, dvo::ActionType::master_volume_adjust, 5},
      {"down", 4, dvo::ActionType::master_volume_adjust, -7},
  };
  CHECK(dvo::format_action_summary(actions) ==
        "正在执行：播放音乐，然后暂停音乐，然后增加音量 5%，然后降低音量 7%");
  CHECK(dvo::format_action_summary({}).empty());

  std::vector<dvo::PlannedAction> search_actions{
      {"s1", 1, dvo::ActionType::media_search_play, {}, "晴天",
       dvo::MediaSearchType::song, "周杰伦"},
      {"s2", 2, dvo::ActionType::media_search_play, {}, "稻香",
       dvo::MediaSearchType::song, std::nullopt},
      {"s3", 3, dvo::ActionType::media_search_play, {}, "轻音乐",
       dvo::MediaSearchType::playlist, std::nullopt},
  };
  CHECK(dvo::format_action_summary(search_actions) ==
        "正在执行：播放 周杰伦 的 晴天，然后播放歌曲 稻香，然后播放歌单 轻音乐");

  auto plan = command_plan("summary-correlation");
  plan.activation_id = "activation-summary";
  plan.turn_index = 4;
  plan.origin = dvo::UtteranceOrigin::followup;
  plan.trigger_sample = 31'000;
  const auto announcement = dvo::make_plan_announcement(plan, "announcement-summary");
  CHECK(announcement.announcement_id == "announcement-summary");
  CHECK(announcement.activation_id == plan.activation_id);
  CHECK(announcement.turn_index == plan.turn_index);
  CHECK(announcement.origin == plan.origin);
  CHECK(announcement.trigger_sample == plan.trigger_sample);
  CHECK(announcement.command_id == plan.command_id);
  REQUIRE(announcement.action_ids.size() == 1);
  CHECK(announcement.action_ids.front() == plan.actions.front().action_id);
}

TEST_CASE("Log announcements publish correlated started and finished lifecycle") {
  auto backend = std::make_shared<dvo::LogAnnouncementBackend>();
  std::mutex mutex;
  std::vector<dvo::AnnouncementResult> results;
  dvo::AnnouncementDispatcher dispatcher(
      backend, {}, [&](const dvo::AnnouncementResult& result) {
        std::scoped_lock lock(mutex);
        results.push_back(result);
      });

  const auto value = request("announcement-lifecycle");
  REQUIRE(dispatcher.try_submit(value).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  {
    std::scoped_lock lock(mutex);
    REQUIRE(results.size() == 2);
    CHECK(results[0].status == dvo::AnnouncementStatus::playback_started);
    CHECK(results[1].status == dvo::AnnouncementStatus::playback_finished);
    for (const auto& result : results) {
      CHECK(result.announcement_id == value.announcement_id);
      CHECK(result.activation_id == value.activation_id);
      CHECK(result.turn_index == value.turn_index);
      CHECK(result.origin == value.origin);
      CHECK(result.trigger_sample == value.trigger_sample);
      CHECK(result.utterance_id == value.utterance_id);
      CHECK(result.command_id == value.command_id);
      CHECK(result.action_ids == value.action_ids);
      CHECK(result.timestamp_sample == value.timestamp_sample);
      CHECK(result.backend == "log");
    }
  }
}

TEST_CASE("Announcement queue is bounded and submission never waits for playback") {
  auto backend = std::make_shared<BlockingAnnouncementBackend>();
  dvo::AnnouncementDispatcherConfig config;
  config.queue_capacity = 1;
  dvo::AnnouncementDispatcher dispatcher(backend, config);

  REQUIRE(dispatcher.try_submit(request("one")).accepted());
  REQUIRE(backend->wait_entered(std::chrono::seconds(2)));
  REQUIRE(dispatcher.try_submit(request("two")).accepted());
  CHECK(dispatcher.try_submit(request("three")).status ==
        dvo::AnnouncementSubmitStatus::queue_full);
  backend->release();
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->count() == 2);
}

TEST_CASE("Backend and lifecycle observer failures are isolated") {
  auto backend = std::make_shared<ThrowOnceAnnouncementBackend>();
  std::mutex mutex;
  std::vector<dvo::AnnouncementStatus> terminal;
  std::atomic<bool> throw_observer{true};
  dvo::AnnouncementDispatcher dispatcher(
      backend, {}, [&](const dvo::AnnouncementResult& result) {
        if (throw_observer.exchange(false)) {
          throw std::runtime_error("injected lifecycle observer failure");
        }
        if (result.status != dvo::AnnouncementStatus::playback_started) {
          std::scoped_lock lock(mutex);
          terminal.push_back(result.status);
        }
      });

  REQUIRE(dispatcher.try_submit(request("fails")).accepted());
  REQUIRE(dispatcher.try_submit(request("continues")).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->calls == 2);
  {
    std::scoped_lock lock(mutex);
    REQUIRE(terminal.size() == 2);
    CHECK(terminal[0] == dvo::AnnouncementStatus::playback_failed);
    CHECK(terminal[1] == dvo::AnnouncementStatus::playback_finished);
  }
}

TEST_CASE("Audible backend policy suppresses non-live requests while log audit retains them") {
  auto live_only_backend = std::make_shared<ThrowOnceAnnouncementBackend>();
  live_only_backend->throw_next = false;
  dvo::AnnouncementDispatcherConfig live_only;
  live_only.live_only = true;
  dvo::AnnouncementDispatcher audible(live_only_backend, live_only);

  CHECK(audible.try_submit(request("replay", dvo::ExecutionSource::replay)).status ==
        dvo::AnnouncementSubmitStatus::suppressed);
  CHECK(audible.try_submit(request("benchmark", dvo::ExecutionSource::benchmark)).status ==
        dvo::AnnouncementSubmitStatus::suppressed);
  CHECK(live_only_backend->calls == 0);

  auto log_backend = std::make_shared<dvo::LogAnnouncementBackend>();
  dvo::AnnouncementDispatcher audit(log_backend);
  REQUIRE(audit.try_submit(request("replay-log", dvo::ExecutionSource::replay)).accepted());
  REQUIRE(audit.wait_until_idle(std::chrono::seconds(2)));
}

TEST_CASE("Slow announcement playback cannot block or reorder action execution") {
  auto announcement_backend = std::make_shared<BlockingAnnouncementBackend>();
  dvo::AnnouncementDispatcherConfig announcement_config;
  announcement_config.queue_capacity = 1;
  dvo::AnnouncementDispatcher announcements(announcement_backend, announcement_config);
  auto action_backend = std::make_shared<CountingActionBackend>();
  std::atomic<std::size_t> dropped{};
  std::atomic<std::size_t> sequence{};
  dvo::OrderedActionExecutor executor(
      action_backend, {}, {}, [&](const dvo::CommandPlan& plan) {
        auto value = request("plan-" + std::to_string(++sequence));
        value.command_id = plan.command_id;
        value.utterance_id = plan.utterance_id;
        value.action_ids = {plan.actions.front().action_id};
        if (!announcements.try_submit(std::move(value)).accepted()) ++dropped;
      });

  REQUIRE(executor.try_submit(command_plan("one")).accepted());
  REQUIRE(announcement_backend->wait_entered(std::chrono::seconds(2)));
  REQUIRE(executor.try_submit(command_plan("two")).accepted());
  REQUIRE(executor.try_submit(command_plan("three")).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(action_backend->calls == 3);
  CHECK(dropped >= 1);

  announcement_backend->release();
  REQUIRE(announcements.wait_until_idle(std::chrono::seconds(2)));
}

TEST_CASE("Action summaries format app.open actions with display name or app_id") {
  dvo::PlannedAction named;
  named.action_id = "a1";
  named.sequence = 1;
  named.type = dvo::ActionType::app_open;
  named.app_id = "clash";
  named.app_display_name = "Clash";

  dvo::PlannedAction id_only;
  id_only.action_id = "a2";
  id_only.sequence = 2;
  id_only.type = dvo::ActionType::app_open;
  id_only.app_id = "steam";

  CHECK(dvo::format_action_summary({named, id_only}) ==
        "正在执行：打开应用 Clash，然后打开应用 steam");
}

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

struct ScopedEnvVar {
  std::wstring name;
  ScopedEnvVar(std::wstring n, std::wstring val) : name(std::move(n)) {
    SetEnvironmentVariableW(name.c_str(), val.c_str());
  }
  ~ScopedEnvVar() {
    SetEnvironmentVariableW(name.c_str(), nullptr);
  }
};

struct ScopedTempDir {
  std::filesystem::path path;
  explicit ScopedTempDir(std::string_view prefix) {
    path = std::filesystem::temp_directory_path() /
           (std::string(prefix) + "_" + std::to_string(GetCurrentProcessId()) + "_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

std::string base64_encode_bytes(std::span<const std::uint8_t> data) {
  if (data.empty()) return {};
  DWORD str_len = 0;
  CryptBinaryToStringA(data.data(), static_cast<DWORD>(data.size()),
                       CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF,
                       nullptr, &str_len);
  std::string result(str_len, '\0');
  CryptBinaryToStringA(data.data(), static_cast<DWORD>(data.size()),
                       CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF,
                       result.data(), &str_len);
  while (!result.empty() && (result.back() == '\0' || result.back() == '\r' || result.back() == '\n')) {
    result.pop_back();
  }
  return result;
}

class DelayedAudibleBackend final : public dvo::IAnnouncementBackend {
 public:
  DelayedAudibleBackend(std::chrono::milliseconds synthesis_delay, bool fail_synthesis = false)
      : delay_(synthesis_delay), fail_(fail_synthesis) {}

  [[nodiscard]] const char* name() const noexcept override { return "delayed_audible"; }

  void deliver(const dvo::AnnouncementRequest& r, std::stop_token stop) override {
    deliver(r, stop, nullptr);
  }

  void deliver(const dvo::AnnouncementRequest&,
               std::stop_token stop,
               const PlaybackStartedCallback& on_playback_started) override {
    {
      std::scoped_lock lock(mutex_);
      entered_ = true;
    }
    cv_.notify_all();

    const auto deadline = std::chrono::steady_clock::now() + delay_;
    while (std::chrono::steady_clock::now() < deadline) {
      if (stop.stop_requested()) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (fail_) {
      throw std::runtime_error("synthesis failed intentionally");
    }

    if (stop.stop_requested()) return;

    if (on_playback_started) on_playback_started();
    playback_invoked_ = true;
  }

  bool wait_entered(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return entered_; });
  }

  std::chrono::milliseconds delay_;
  bool fail_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_{false};
  std::atomic<bool> playback_invoked_{false};
};

struct RecordedMockRequest {
  std::string method;
  std::string target;
  std::string api_key_header;
  std::string content_type;
  std::string body;
};

struct MockHttpResponse {
  unsigned status{200};
  std::string content_type{"application/json"};
  std::string body;
  std::chrono::milliseconds delay{0};
};

class LoopbackHttpServer {
 public:
  using Handler = std::function<MockHttpResponse(const RecordedMockRequest&)>;

  explicit LoopbackHttpServer(Handler handler)
      : handler_(std::move(handler)),
        acceptor_(io_, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0)) {
    port_ = acceptor_.local_endpoint().port();
    worker_ = std::jthread([this](std::stop_token stop) { serve(stop); });
  }

  ~LoopbackHttpServer() { stop(); }

  void stop() {
    if (stopped_.exchange(true)) return;
    beast::error_code ec;
    acceptor_.close(ec);
    io_.stop();
    if (worker_.joinable()) {
      worker_.request_stop();
      worker_.join();
    }
  }

  [[nodiscard]] std::uint16_t port() const noexcept { return port_; }
  [[nodiscard]] std::string base_url() const {
    return "http://127.0.0.1:" + std::to_string(port_);
  }

  [[nodiscard]] std::vector<RecordedMockRequest> requests() const {
    std::scoped_lock lock(mutex_);
    return requests_;
  }

 private:
  void serve(std::stop_token stop) {
    acceptor_.non_blocking(true);
    while (!stop.stop_requested() && !stopped_.load(std::memory_order_acquire)) {
      tcp::socket socket(io_);
      beast::error_code ec;
      acceptor_.accept(socket, ec);
      if (ec == asio::error::would_block || ec == asio::error::try_again) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      if (ec) break;

      std::jthread([this, sock = std::move(socket), stop]() mutable {
        beast::tcp_stream stream(std::move(sock));
        stream.expires_after(std::chrono::seconds(5));
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        beast::error_code read_ec;
        http::read(stream, buffer, req, read_ec);
        if (read_ec) return;

        RecordedMockRequest recorded;
        recorded.method = std::string(req.method_string());
        recorded.target = std::string(req.target());
        if (const auto it = req.find("x-goog-api-key"); it != req.end()) {
          recorded.api_key_header = std::string(it->value());
        }
        if (const auto it = req.find(http::field::content_type); it != req.end()) {
          recorded.content_type = std::string(it->value());
        }
        recorded.body = req.body();

        MockHttpResponse reply;
        {
          std::scoped_lock lock(mutex_);
          requests_.push_back(recorded);
          if (handler_) reply = handler_(recorded);
        }

        if (reply.delay > std::chrono::milliseconds::zero()) {
          std::this_thread::sleep_for(reply.delay);
        }

        http::response<http::string_body> res{
            static_cast<http::status>(reply.status), req.version()};
        res.set(http::field::server, "MockGeminiTTS/1.0");
        res.set(http::field::content_type, reply.content_type);
        res.keep_alive(false);
        res.body() = std::move(reply.body);
        res.prepare_payload();

        beast::error_code write_ec;
        http::write(stream, res, write_ec);
        stream.socket().shutdown(tcp::socket::shutdown_both, write_ec);
      }).detach();
    }
  }

  Handler handler_;
  asio::io_context io_;
  tcp::acceptor acceptor_;
  std::uint16_t port_{0};
  std::atomic<bool> stopped_{false};
  mutable std::mutex mutex_;
  std::vector<RecordedMockRequest> requests_;
  std::jthread worker_;
};

}  // namespace

TEST_CASE("Delayed synthesis gating test prevents premature wake gate locking") {
  SECTION("Audible backend does not emit playback_started while synthesis is running") {
    auto backend = std::make_shared<DelayedAudibleBackend>(std::chrono::milliseconds(80));
    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("delayed-1")).accepted());
    REQUIRE(backend->wait_entered(std::chrono::seconds(2)));

    // While synthesis is running in its 80ms delay, playback_started must NOT have been emitted!
    {
      std::scoped_lock lock(mutex);
      CHECK(events.empty());
    }

    REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
    CHECK(backend->playback_invoked_);
    {
      std::scoped_lock lock(mutex);
      REQUIRE(events.size() == 2);
      CHECK(events[0] == dvo::AnnouncementStatus::playback_started);
      CHECK(events[1] == dvo::AnnouncementStatus::playback_finished);
    }
  }

  SECTION("Synthesis failure never emits playback_started") {
    auto backend = std::make_shared<DelayedAudibleBackend>(std::chrono::milliseconds(20), true);
    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("fail-synthesis")).accepted());
    REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
    CHECK_FALSE(backend->playback_invoked_);
    {
      std::scoped_lock lock(mutex);
      REQUIRE(events.size() == 1);
      CHECK(events[0] == dvo::AnnouncementStatus::playback_failed);
    }
  }

  SECTION("Cancellation before synthesis completes never emits playback_started") {
    auto backend = std::make_shared<DelayedAudibleBackend>(std::chrono::milliseconds(300), false);
    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("cancel-synthesis")).accepted());
    REQUIRE(backend->wait_entered(std::chrono::seconds(2)));

    // Stop without draining while synthesis is active
    dispatcher.stop(false);
    CHECK_FALSE(backend->playback_invoked_);
    {
      std::scoped_lock lock(mutex);
      REQUIRE(events.size() == 1);
      CHECK(events[0] == dvo::AnnouncementStatus::playback_cancelled);
    }
  }
}

TEST_CASE("Google TTS synthesis, PCM-to-WAV wrapping, and SHA-256 persistent disk cache hit") {
  ScopedTempDir temp_dir("dvo_tts_test_cache");
  ScopedEnvVar env_var(L"DVO_TEST_GEMINI_API_KEY", L"test-secret-key-1234");

  // Prepare 200 bytes of dummy PCM samples (100 16-bit mono samples)
  std::vector<std::uint8_t> dummy_pcm(200);
  for (std::size_t i = 0; i < dummy_pcm.size(); ++i) {
    dummy_pcm[i] = static_cast<std::uint8_t>(i & 0xFF);
  }
  const std::string b64_pcm = base64_encode_bytes(dummy_pcm);

  LoopbackHttpServer server([&](const RecordedMockRequest& req) -> MockHttpResponse {
    CHECK(req.method == "POST");
    CHECK(req.target.find("test-secret-key-1234") == std::string::npos);
    CHECK(req.api_key_header == "test-secret-key-1234");

    nlohmann::json reply_json;
    reply_json["candidates"] = nlohmann::json::array({
        {{"content", {
            {"parts", nlohmann::json::array({
                {{"inlineData", {
                    {"mimeType", "audio/L16;codec=pcm;rate=24000"},
                    {"data", b64_pcm}
                }}}
            })}
        }}}
    });

    MockHttpResponse resp;
    resp.status = 200;
    resp.body = reply_json.dump();
    return resp;
  });

  std::vector<std::vector<std::uint8_t>> played_wavs;
  std::mutex played_mutex;
  auto test_player = [&](std::span<const std::uint8_t> wav_bytes, std::stop_token) {
    std::scoped_lock lock(played_mutex);
    played_wavs.emplace_back(wav_bytes.begin(), wav_bytes.end());
  };

  dvo::TtsBackendOptions options;
  options.backend = "google_tts";
  options.cache_dir = temp_dir.path;
  options.google_api_key_env = "DVO_TEST_GEMINI_API_KEY";
  options.google_base_url = server.base_url();
  options.google_model = "gemini-3.8-flash-tts";
  options.google_voice = "Kore";
  options.google_timeout_ms = 3000;

  auto backend = dvo::create_announcement_backend(options, test_player);
  REQUIRE(backend != nullptr);
  CHECK(std::string(backend->name()) == "google_tts");

  dvo::AnnouncementDispatcher dispatcher(backend, {});

  // First request: Must make HTTP call, wrap PCM into RIFF WAV, and populate disk cache
  REQUIRE(dispatcher.try_submit(request("tts-req-1")).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(3)));

  CHECK(server.requests().size() == 1);
  {
    std::scoped_lock lock(played_mutex);
    REQUIRE(played_wavs.size() == 1);
    const auto& wav = played_wavs[0];
    REQUIRE(wav.size() == 44 + dummy_pcm.size());
    CHECK(std::memcmp(wav.data(), "RIFF", 4) == 0);
    CHECK(std::memcmp(wav.data() + 8, "WAVE", 4) == 0);
    // Sample rate in RIFF header is at offset 24 (little-endian 32-bit uint)
    std::uint32_t sample_rate = 0;
    std::memcpy(&sample_rate, wav.data() + 24, 4);
    CHECK(sample_rate == 24000);
  }

  // Second identical request: Must hit the SHA-256 disk cache with 0 additional HTTP requests!
  REQUIRE(dispatcher.try_submit(request("tts-req-2")).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(3)));

  CHECK(server.requests().size() == 1);  // Still exactly 1 HTTP request!
  {
    std::scoped_lock lock(played_mutex);
    REQUIRE(played_wavs.size() == 2);
    CHECK(played_wavs[1] == played_wavs[0]);
  }
}

TEST_CASE("Dual-tier auto fallback cascades cleanly without duplicate playback_started") {
  ScopedTempDir temp_dir("dvo_auto_fallback_cache");

  SECTION("Falls back to silent log when API key is missing and kokoro models are absent") {
    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::TtsBackendOptions options;
    options.backend = "auto";
    options.cache_dir = temp_dir.path;
    options.google_api_key_env = "DVO_NONEXISTENT_KEY_99999";
    options.kokoro_model_dir = "models/nonexistent_model_dir_99999";

    auto backend = dvo::create_announcement_backend(options, nullptr);
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("auto-missing-key")).accepted());
    REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));

    std::scoped_lock lock(mutex);
    REQUIRE(events.size() == 1);
    CHECK(events[0] == dvo::AnnouncementStatus::playback_finished);
  }

  SECTION("Falls back cleanly to silent log when HTTP endpoint returns 503") {
    ScopedEnvVar env_var(L"DVO_TEST_AUTO_KEY", L"valid-key");

    LoopbackHttpServer server([](const RecordedMockRequest&) -> MockHttpResponse {
      MockHttpResponse resp;
      resp.status = 503;
      resp.body = "Service Unavailable";
      return resp;
    });

    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::TtsBackendOptions options;
    options.backend = "auto";
    options.cache_dir = temp_dir.path;
    options.google_api_key_env = "DVO_TEST_AUTO_KEY";
    options.google_base_url = server.base_url();
    options.google_timeout_ms = 1000;
    options.kokoro_model_dir = "models/nonexistent_model_dir_99999";

    auto backend = dvo::create_announcement_backend(options, nullptr);
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("auto-http-503")).accepted());
    REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));

    CHECK(server.requests().size() == 1);
    std::scoped_lock lock(mutex);
    REQUIRE(events.size() == 1);
    CHECK(events[0] == dvo::AnnouncementStatus::playback_finished);
  }

  SECTION("Emits playback_started exactly once when auto succeeds via google_tts") {
    ScopedEnvVar env_var(L"DVO_TEST_AUTO_KEY_2", L"valid-key-2");

    std::vector<std::uint8_t> dummy_pcm(200);
    for (std::size_t i = 0; i < dummy_pcm.size(); ++i) {
      dummy_pcm[i] = static_cast<std::uint8_t>(i & 0xFF);
    }
    const std::string b64_pcm = base64_encode_bytes(dummy_pcm);

    LoopbackHttpServer server([&](const RecordedMockRequest&) -> MockHttpResponse {
      nlohmann::json reply_json;
      reply_json["candidates"] = nlohmann::json::array({
          {{"content", {
              {"parts", nlohmann::json::array({
                  {{"inlineData", {
                      {"mimeType", "audio/L16;codec=pcm;rate=24000"},
                      {"data", b64_pcm}
                  }}}
              })}
          }}}
      });

      MockHttpResponse resp;
      resp.status = 200;
      resp.body = reply_json.dump();
      return resp;
    });

    std::vector<std::vector<std::uint8_t>> played_wavs;
    std::mutex played_mutex;
    auto test_player = [&](std::span<const std::uint8_t> wav_bytes, std::stop_token) {
      std::scoped_lock lock(played_mutex);
      played_wavs.emplace_back(wav_bytes.begin(), wav_bytes.end());
    };

    std::mutex mutex;
    std::vector<dvo::AnnouncementStatus> events;
    dvo::TtsBackendOptions options;
    options.backend = "auto";
    options.cache_dir = temp_dir.path;
    options.google_api_key_env = "DVO_TEST_AUTO_KEY_2";
    options.google_base_url = server.base_url();
    options.google_timeout_ms = 1000;
    options.kokoro_model_dir = "models/nonexistent_model_dir_99999";

    auto backend = dvo::create_announcement_backend(options, test_player);
    dvo::AnnouncementDispatcher dispatcher(backend, {}, [&](const dvo::AnnouncementResult& res) {
      std::scoped_lock lock(mutex);
      events.push_back(res.status);
    });

    REQUIRE(dispatcher.try_submit(request("auto-http-200")).accepted());
    REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));

    CHECK(server.requests().size() == 1);
    {
      std::scoped_lock lock(played_mutex);
      CHECK(played_wavs.size() == 1);
    }
    std::scoped_lock lock(mutex);
    REQUIRE(events.size() == 2);
    CHECK(events[0] == dvo::AnnouncementStatus::playback_started);
    CHECK(events[1] == dvo::AnnouncementStatus::playback_finished);
  }
}

