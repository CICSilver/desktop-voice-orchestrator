#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dvo/config.h"
#include "dvo/debug_server.h"
#include "dvo/runtime.h"

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

struct StaticRoot {
  StaticRoot() {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("dvo-debug-server-" + std::to_string(suffix));
    std::filesystem::create_directories(path);
    std::ofstream(path / "index.html", std::ios::binary | std::ios::trunc)
        << "<!doctype html><title>DVO test</title>";
  }
  ~StaticRoot() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  std::filesystem::path path;
};

std::uint16_t port_of(const dvo::DebugServer& server) {
  const auto url = server.url();
  const auto colon = url.find(':', std::string("http://").size());
  const auto slash = url.find('/', colon);
  return static_cast<std::uint16_t>(
      std::stoi(url.substr(colon + 1, slash - colon - 1)));
}

dvo::WebConfig web_config(const std::filesystem::path& root) {
  dvo::WebConfig config;
  config.bind = "127.0.0.1";
  config.port = 0;
  config.static_root = root;
  return config;
}

using TestWebSocket = websocket::stream<beast::tcp_stream>;

std::unique_ptr<TestWebSocket> connect_websocket(asio::io_context& io,
                                                  std::uint16_t port,
                                                  const std::string& token,
                                                  std::string origin) {
  tcp::resolver resolver(io);
  auto stream = std::make_unique<TestWebSocket>(io);
  beast::get_lowest_layer(*stream).expires_after(std::chrono::seconds(3));
  beast::get_lowest_layer(*stream).connect(
      resolver.resolve("127.0.0.1", std::to_string(port)));
  stream->set_option(websocket::stream_base::decorator(
      [origin = std::move(origin)](websocket::request_type& request) {
        request.set(http::field::origin, origin);
      }));
  stream->handshake("127.0.0.1:" + std::to_string(port),
                    "/ws?token=" + token);
  return stream;
}

void abort_socket(TestWebSocket& socket) {
  beast::error_code ignored;
  beast::get_lowest_layer(socket).socket().shutdown(tcp::socket::shutdown_both,
                                                    ignored);
  beast::get_lowest_layer(socket).socket().close(ignored);
}

}  // namespace

TEST_CASE("WebSocket enforces token and same-origin and sends versioned history") {
  StaticRoot root;
  dvo::DebugServer server(128);
  server.start(web_config(root.path), [](const auto&) {
    return nlohmann::json{{"ok", true}};
  });
  const auto port = port_of(server);
  const auto expected_origin = "http://127.0.0.1:" + std::to_string(port);

  SECTION("invalid token is rejected") {
    asio::io_context io;
    CHECK_THROWS(connect_websocket(io, port, "invalid", expected_origin));
  }

  SECTION("cross-origin upgrade is rejected") {
    asio::io_context io;
    CHECK_THROWS(connect_websocket(io, port, server.token(),
                                   "http://attacker.invalid"));
  }

  SECTION("valid clients receive schema sequence and reconnect history") {
    asio::io_context io;
    auto socket = connect_websocket(io, port, server.token(), expected_origin);
    REQUIRE(server.publish("test_event", {{"value", 7}}, 320, "replay"));
    beast::flat_buffer buffer;
    socket->read(buffer);
    const auto first = nlohmann::json::parse(beast::buffers_to_string(buffer.data()));
    CHECK(first["schema_version"] == 1);
    CHECK(first["seq"] == 1);
    CHECK(first["session_id"].is_string());
    CHECK(first["source"] == "replay");
    CHECK(first["type"] == "test_event");
    CHECK(first["timestamp_sample"] == 320);
    CHECK(first["payload"]["value"] == 7);
    abort_socket(*socket);

    auto reconnected =
        connect_websocket(io, port, server.token(), expected_origin);
    buffer.consume(buffer.size());
    reconnected->read(buffer);
    const auto history =
        nlohmann::json::parse(beast::buffers_to_string(buffer.data()));
    CHECK(history["session_id"] == first["session_id"]);
    CHECK(history["seq"] == first["seq"]);
    CHECK(history["type"] == "test_event");
    abort_socket(*reconnected);
  }

  server.stop();
}

TEST_CASE("debug command endpoint is token protected") {
  StaticRoot root;
  dvo::DebugServer server;
  server.start(web_config(root.path), [](const nlohmann::json& command) {
    return nlohmann::json{{"ok", true}, {"echo", command}};
  });
  const auto port = port_of(server);

  const auto request = [&](std::string target) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    stream.expires_after(std::chrono::seconds(3));
    stream.connect(resolver.resolve("127.0.0.1", std::to_string(port)));
    http::request<http::string_body> value{http::verb::post,
                                          std::move(target), 11};
    value.set(http::field::host, "127.0.0.1");
    value.set(http::field::content_type, "application/json");
    value.body() = R"({"action":"ping"})";
    value.prepare_payload();
    http::write(stream, value);
    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(stream, buffer, response);
    return response;
  };

  CHECK(request("/api/command").result() == http::status::unauthorized);
  const auto accepted = request("/api/command?token=" + server.token());
  CHECK(accepted.result() == http::status::ok);
  const auto payload = nlohmann::json::parse(accepted.body());
  CHECK(payload["ok"].get<bool>());
  CHECK(payload["echo"]["action"] == "ping");
  server.stop();
}

TEST_CASE("session audio endpoint is token protected and serves only handler resources") {
  StaticRoot root;
  const auto audio = root.path / "audio.wav";
  std::ofstream(audio, std::ios::binary | std::ios::trunc) << "RIFF-test-audio";
  dvo::DebugServer server;
  server.start(web_config(root.path), {},
               [audio](const nlohmann::json& request)
                   -> std::optional<dvo::DebugServer::FileResource> {
                 if (request.value("session", "") != "session-1" ||
                     request.value("stream", "") != "processed") {
                   return std::nullopt;
                 }
                 return dvo::DebugServer::FileResource{audio, "audio/wav"};
               });
  const auto port = port_of(server);

  const auto request = [&](std::string target) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    stream.expires_after(std::chrono::seconds(3));
    stream.connect(resolver.resolve("127.0.0.1", std::to_string(port)));
    http::request<http::empty_body> value{http::verb::get,
                                         std::move(target), 11};
    value.set(http::field::host, "127.0.0.1");
    http::write(stream, value);
    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(stream, buffer, response);
    return response;
  };

  CHECK(request("/api/session-audio?session=session-1&stream=processed")
            .result() == http::status::unauthorized);
  const auto accepted =
      request("/api/session-audio?token=" + server.token() +
              "&session=session-1&stream=processed");
  CHECK(accepted.result() == http::status::ok);
  CHECK(accepted[http::field::content_type] == "audio/wav");
  CHECK(accepted.body() == "RIFF-test-audio");
  CHECK(request("/api/session-audio?token=" + server.token() +
                "&session=missing&stream=processed")
            .result() == http::status::not_found);
  server.stop();
}

TEST_CASE("slow WebSocket clients use a bounded drop queue") {
  StaticRoot root;
  dvo::DebugServer server(4096);
  server.start(web_config(root.path), {});
  const auto port = port_of(server);
  asio::io_context io;
  auto socket = connect_websocket(
      io, port, server.token(),
      "http://127.0.0.1:" + std::to_string(port));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));

  const std::string bulk(16 * 1024, 'x');
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < 1200; ++index) {
    static_cast<void>(server.publish("telemetry", {{"index", index},
                                                    {"bulk", bulk}}));
  }
  const auto publish_elapsed = std::chrono::steady_clock::now() - started;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(6);
  while (server.slow_client_dropped() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(publish_elapsed < std::chrono::seconds(2));
  CHECK(server.slow_client_dropped() > 0);
  abort_socket(*socket);
  server.stop();
}

TEST_CASE("debug server reports a bind failure during start") {
  StaticRoot root;
  dvo::DebugServer first;
  auto config = web_config(root.path);
  first.start(config, {});
  config.port = port_of(first);
  dvo::DebugServer second;
  CHECK_THROWS_WITH(second.start(config, {}),
                    Catch::Matchers::ContainsSubstring("startup failed"));
  first.stop();
}

namespace {

struct MockOllamaRecord {
  std::string target;
  std::string body;
};

class MockOllamaServer {
 public:
  MockOllamaServer()
      : acceptor_(io_, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0)) {
    port_ = acceptor_.local_endpoint().port();
    worker_ = std::jthread([this](std::stop_token stop) { serve(stop); });
  }

  ~MockOllamaServer() { stop(); }

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
    return "http://127.0.0.1:" + std::to_string(port_) + "/v1";
  }

  [[nodiscard]] std::vector<MockOllamaRecord> requests() const {
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

      std::jthread([this, sock = std::move(socket)]() mutable {
        beast::tcp_stream stream(std::move(sock));
        stream.expires_after(std::chrono::seconds(2));
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        beast::error_code read_ec;
        http::read(stream, buffer, req, read_ec);
        if (read_ec) return;

        {
          std::scoped_lock lock(mutex_);
          requests_.push_back({std::string(req.target()), req.body()});
        }

        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::content_type, "application/json");
        res.keep_alive(false);
        res.body() = R"({"response":"done"})";
        res.prepare_payload();
        beast::error_code write_ec;
        http::write(stream, res, write_ec);
        stream.socket().shutdown(tcp::socket::shutdown_send, write_ec);
      }).detach();
    }
  }

  asio::io_context io_;
  tcp::acceptor acceptor_;
  std::uint16_t port_{};
  std::atomic<bool> stopped_{false};
  mutable std::mutex mutex_;
  std::vector<MockOllamaRecord> requests_;
  std::jthread worker_;
};

}  // namespace

TEST_CASE("runtime router profile hot-switch and mock ollama unload/preload") {
  MockOllamaServer mock_ollama;
  const auto root = std::filesystem::path(DVO_PROJECT_ROOT);
  StaticRoot temp_dir;
  const auto local_toml = temp_dir.path / "local.toml";
  dvo::ConfigStore store(root / "config/default.toml", local_toml, root);
  auto config = store.load();

  config.commands.local_router_base_url = mock_ollama.base_url();
  config.commands.local_router_model = "test-qwen";
  config.commands.game_router_backend = "openai_compatible";
  config.commands.cloud_router_base_url = "https://api.openai.com/v1";
  config.commands.cloud_router_model = "gpt-4o-mini";
  config.commands.router_profile = "reading";
  config.commands.router_backend = "openai_compatible";
  config.commands.router_base_url = mock_ollama.base_url();
  config.commands.router_model = "test-qwen";
  config.commands.parser_mode = "hybrid";

  dvo::VoiceFrontendRuntime runtime(config, store);

  // Initial status check
  auto status_res = runtime.handle_command({{"action", "router.status"}});
  REQUIRE(status_res["ok"].get<bool>());
  CHECK(status_res["router"]["profile"] == "reading");
  CHECK(status_res["router"]["vram_resident"].get<bool>());

  // 1) Switch to game mode (env key unset -> fallback to local_rules, unload sent < 500ms)
  (void)_putenv_s("OPENAI_API_KEY", "");
  auto game_res = runtime.handle_command({
      {"action", "router.set_profile"},
      {"profile", "game"}
  });
  REQUIRE(game_res["ok"].get<bool>());
  CHECK(game_res["vram_unloaded"].get<bool>());
  CHECK(game_res["unload_elapsed_ms"].get<std::uint64_t>() < 500);
  CHECK(game_res["effective_fallback"] == "local_rules");
  CHECK(game_res["router"]["profile"] == "game");
  CHECK_FALSE(game_res["router"]["vram_resident"].get<bool>());
  CHECK(game_res["router"]["router_backend"] == "none");
  CHECK(game_res["router"]["parser_mode"] == "rules");

  auto reqs = mock_ollama.requests();
  REQUIRE(!reqs.empty());
  CHECK(reqs.back().target == "/api/generate");
  auto unload_payload = nlohmann::json::parse(reqs.back().body);
  CHECK(unload_payload["model"] == "test-qwen");
  CHECK(unload_payload["keep_alive"] == 0);

  // 2) Switch back to reading mode (hybrid mode, preload sent)
  auto reading_res = runtime.handle_command({
      {"action", "router.set_profile"},
      {"profile", "reading"},
      {"preload", true}
  });
  REQUIRE(reading_res["ok"].get<bool>());
  CHECK(reading_res["preloaded"].get<bool>());
  CHECK(reading_res["router"]["profile"] == "reading");
  CHECK(reading_res["router"]["vram_resident"].get<bool>());
  CHECK(reading_res["router"]["router_backend"] == "openai_compatible");
  CHECK(reading_res["router"]["router_base_url"] == mock_ollama.base_url());
  CHECK(reading_res["router"]["parser_mode"] == "hybrid");

  reqs = mock_ollama.requests();
  REQUIRE(reqs.size() >= 2);
  CHECK(reqs.back().target == "/api/generate");
  auto preload_payload = nlohmann::json::parse(reqs.back().body);
  CHECK(preload_payload["model"] == "test-qwen");
  CHECK(preload_payload["keep_alive"] == "30m");

  // 3) Switch to game mode with env key set -> effective_fallback is cloud_router
  REQUIRE(_putenv_s("OPENAI_API_KEY", "test-key-mock") == 0);
  auto cloud_game_res = runtime.handle_command({
      {"action", "router.set_profile"},
      {"profile", "game"}
  });
  REQUIRE(cloud_game_res["ok"].get<bool>());
  CHECK(cloud_game_res["effective_fallback"] == "cloud_router");
  CHECK(cloud_game_res["router"]["profile"] == "game");
  CHECK(cloud_game_res["router"]["router_backend"] == "openai_compatible");
  CHECK(cloud_game_res["router"]["router_base_url"] == "https://api.openai.com/v1");
  CHECK(cloud_game_res["router"]["router_model"] == "gpt-4o-mini");
  (void)_putenv_s("OPENAI_API_KEY", "");
}
