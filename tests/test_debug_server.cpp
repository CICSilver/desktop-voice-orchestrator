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

#include "dvo/debug_server.h"

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
