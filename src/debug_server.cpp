#include "dvo/debug_server.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <Windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace dvo {
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

namespace {

struct Subscriber {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::string> queue;
  bool stopped{};
  std::uint64_t dropped{};
};

std::string random_token() {
  std::array<unsigned char, 24> bytes{};
  if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
    throw std::runtime_error("BCryptGenRandom failed");
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

std::string query_value(beast::string_view target, beast::string_view key) {
  const auto question = target.find('?');
  if (question == beast::string_view::npos) return {};
  auto query = target.substr(question + 1);
  while (!query.empty()) {
    const auto amp = query.find('&');
    const auto part = query.substr(0, amp);
    const auto equals = part.find('=');
    if (equals != beast::string_view::npos && part.substr(0, equals) == key) {
      return std::string(part.substr(equals + 1));
    }
    if (amp == beast::string_view::npos) break;
    query.remove_prefix(amp + 1);
  }
  return {};
}

std::string path_only(beast::string_view target) {
  const auto question = target.find('?');
  return std::string(target.substr(0, question));
}

std::string mime_type(const std::filesystem::path& path) {
  const auto extension = path.extension().string();
  if (extension == ".html") return "text/html; charset=utf-8";
  if (extension == ".js") return "text/javascript; charset=utf-8";
  if (extension == ".css") return "text/css; charset=utf-8";
  if (extension == ".svg") return "image/svg+xml";
  if (extension == ".json") return "application/json";
  return "application/octet-stream";
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("file not found");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

template <typename Body>
void add_security_headers(http::response<Body>& response) {
  response.set(http::field::cache_control, "no-store");
  response.set("X-Content-Type-Options", "nosniff");
  response.set("Content-Security-Policy",
               "default-src 'self'; connect-src 'self' ws://127.0.0.1:*; "
               "style-src 'self'; script-src 'self'; img-src 'self' data:");
}

}  // namespace

struct DebugServer::SharedState {
  WebConfig config;
  CommandHandler commands;
  std::string token;
  std::string session_id;
  std::atomic<bool> stopping{};
  std::atomic<std::uint16_t> bound_port{};
  std::mutex startup_mutex;
  std::condition_variable startup_cv;
  bool startup_done{};
  std::string startup_error;
  std::mutex commands_mutex;
  std::condition_variable commands_cv;
  std::size_t active_commands{};
  std::mutex subscribers_mutex;
  std::vector<std::weak_ptr<Subscriber>> subscribers;
  std::deque<std::string> history;
  std::uint64_t sequence{};
};

DebugServer::DebugServer(std::size_t capacity) : outbound_(capacity) {}

DebugServer::~DebugServer() { stop(); }

void DebugServer::start(const WebConfig& config, CommandHandler commands) {
  std::scoped_lock lifecycle_lock(lifecycle_mutex_);
  stop();
  {
    // Events from a previous server instance must never acquire the new
    // instance's session id.
    std::scoped_lock publish_lock(publish_mutex_);
    Outbound stale;
    while (outbound_.try_pop(stale)) {}
    dropped_.store(0, std::memory_order_release);
    slow_client_dropped_.store(0, std::memory_order_release);
  }
  auto state = std::make_shared<SharedState>();
  state->config = config;
  state->commands = std::move(commands);
  state->token = random_token();
  state->session_id = random_token();
  {
    std::scoped_lock state_lock(state_mutex_);
    state_ = state;
  }
  broker_ = std::jthread([this](std::stop_token stop) { broker_loop(stop); });
  acceptor_ = std::jthread([this](std::stop_token stop) { accept_loop(stop); });
  {
    std::unique_lock startup_lock(state->startup_mutex);
    if (!state->startup_cv.wait_for(startup_lock, std::chrono::seconds(3),
                                    [&] { return state->startup_done; })) {
      startup_lock.unlock();
      stop();
      throw std::runtime_error("debug server startup timed out");
    }
    if (!state->startup_error.empty()) {
      const auto error = state->startup_error;
      startup_lock.unlock();
      stop();
      throw std::runtime_error("debug server startup failed: " + error);
    }
  }
}

void DebugServer::stop() {
  std::scoped_lock lifecycle_lock(lifecycle_mutex_);
  std::shared_ptr<SharedState> state;
  {
    std::scoped_lock state_lock(state_mutex_);
    state = state_;
  }
  if (state) {
    {
      // Pair the stopping transition with producer serialization so publish()
      // cannot enqueue after the broker has been joined.
      std::scoped_lock publish_lock(publish_mutex_);
      state->stopping.store(true, std::memory_order_release);
    }
    std::scoped_lock lock(state->subscribers_mutex);
    for (auto& weak : state->subscribers) {
      if (auto subscriber = weak.lock()) {
        std::scoped_lock subscriber_lock(subscriber->mutex);
        subscriber->stopped = true;
        subscriber->cv.notify_all();
      }
    }
  }
  if (acceptor_.joinable()) { acceptor_.request_stop(); acceptor_.join(); }
  if (broker_.joinable()) { broker_.request_stop(); broker_.join(); }
  if (state) {
    // A detached HTTP connection captures only SharedState, except while it is
    // inside the command callback. Waiting for those callbacks is what makes a
    // handler that captures VoiceFrontendRuntime safe to tear down.
    std::unique_lock commands_lock(state->commands_mutex);
    state->commands_cv.wait(commands_lock,
                            [&] { return state->active_commands == 0; });
  }
  {
    std::scoped_lock state_lock(state_mutex_);
    if (state_ == state) state_.reset();
  }
}

bool DebugServer::publish(std::string type, nlohmann::json payload,
                          std::uint64_t timestamp_sample, std::string source) {
  std::shared_ptr<SharedState> state;
  {
    std::scoped_lock state_lock(state_mutex_);
    state = state_;
  }
  if (!state || state->stopping.load(std::memory_order_acquire)) return false;
  std::unique_lock lock(publish_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!outbound_.try_push({std::move(type), std::move(payload), timestamp_sample, std::move(source)})) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

std::string DebugServer::token() const {
  std::scoped_lock state_lock(state_mutex_);
  return state_ ? state_->token : std::string{};
}

std::string DebugServer::url() const {
  std::scoped_lock state_lock(state_mutex_);
  if (!state_) return {};
  return "http://" + state_->config.bind + ':' +
         std::to_string(state_->bound_port.load(std::memory_order_acquire)) +
         "/?token=" + state_->token;
}

void DebugServer::broker_loop(std::stop_token stop) {
  while (!stop.stop_requested()) {
    Outbound outbound;
    if (!outbound_.try_pop(outbound)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    std::shared_ptr<SharedState> state;
    {
      std::scoped_lock state_lock(state_mutex_);
      state = state_;
    }
    if (!state) break;
    std::string_view level = "info";
    if (outbound.type == "capture_error" || outbound.type == "config_rejected" ||
        outbound.type == "asr_error" || outbound.type == "action_failed") level = "error";
    else if (outbound.type == "candidate_rejected" || outbound.type == "candidate_cancelled" ||
             outbound.type == "telemetry_dropped" || outbound.type == "asr_overloaded" ||
             outbound.type == "command_rejected") level = "warn";
    const auto envelope = nlohmann::json{{"schema_version", 1},
                                         {"seq", ++state->sequence},
                                         {"session_id", state->session_id},
                                         {"source", outbound.source},
                                         {"level", level},
                                         {"type", outbound.type},
                                         {"timestamp_sample", outbound.timestamp_sample},
                                         {"payload", std::move(outbound.payload)}}.dump();
    std::scoped_lock lock(state->subscribers_mutex);
    state->history.push_back(envelope);
    while (state->history.size() > 400) state->history.pop_front();
    state->subscribers.erase(
        std::remove_if(state->subscribers.begin(), state->subscribers.end(),
                       [](const auto& weak) { return weak.expired(); }),
        state->subscribers.end());
    for (auto& weak : state->subscribers) {
      if (auto subscriber = weak.lock()) {
        std::scoped_lock subscriber_lock(subscriber->mutex);
        if (subscriber->queue.size() >= 512) {
          subscriber->queue.pop_front();
          ++subscriber->dropped;
          dropped_.fetch_add(1, std::memory_order_relaxed);
          slow_client_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        subscriber->queue.push_back(envelope);
        subscriber->cv.notify_one();
      }
    }
  }
}

void DebugServer::accept_loop(std::stop_token stop) {
  std::shared_ptr<SharedState> state;
  {
    std::scoped_lock state_lock(state_mutex_);
    state = state_;
  }
  if (!state) return;
  try {
    asio::io_context io;
    const tcp::endpoint endpoint{asio::ip::make_address(state->config.bind),
                                 state->config.port};
    tcp::acceptor acceptor(io);
    acceptor.open(endpoint.protocol());
#if defined(_WIN32)
    const BOOL exclusive = TRUE;
    if (::setsockopt(acceptor.native_handle(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                     reinterpret_cast<const char*>(&exclusive),
                     sizeof(exclusive)) == SOCKET_ERROR) {
      throw std::runtime_error("cannot make debug server port exclusive");
    }
#endif
    acceptor.bind(endpoint);
    acceptor.listen(asio::socket_base::max_listen_connections);
    acceptor.non_blocking(true);
    state->bound_port.store(acceptor.local_endpoint().port(),
                            std::memory_order_release);
    {
      std::scoped_lock startup_lock(state->startup_mutex);
      state->startup_done = true;
      state->startup_cv.notify_all();
    }
    while (!stop.stop_requested() && !state->stopping.load()) {
      beast::error_code error;
      tcp::socket socket(io);
      acceptor.accept(socket, error);
      if (error == asio::error::would_block || error == asio::error::try_again) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      if (error) throw beast::system_error(error);
      std::thread([state, socket = std::move(socket)]() mutable {
        try {
          beast::flat_buffer buffer;
          http::request<http::string_body> request;
          http::read(socket, buffer, request);
          const auto request_path = path_only(request.target());
          const bool protected_resource = request_path == "/ws" || request_path.starts_with("/api/");
          const bool authorized = query_value(request.target(), "token") == state->token;
          if (protected_resource && !authorized) {
            http::response<http::string_body> response{http::status::unauthorized, request.version()};
            response.body() = "Unauthorized";
            response.prepare_payload();
            add_security_headers(response);
            http::write(socket, response);
            return;
          }

          if (websocket::is_upgrade(request) && request_path == "/ws") {
            const auto origin = request.find(http::field::origin);
            const auto expected = "http://" + state->config.bind + ':' +
                                  std::to_string(state->bound_port.load(
                                      std::memory_order_acquire));
            if (origin == request.end() || origin->value() != expected) {
              throw std::runtime_error("WebSocket origin rejected");
            }
            websocket::stream<tcp::socket> ws(std::move(socket));
            ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
            ws.accept(request);
            auto subscriber = std::make_shared<Subscriber>();
            {
              std::scoped_lock lock(state->subscribers_mutex);
              subscriber->queue.insert(subscriber->queue.end(), state->history.begin(), state->history.end());
              state->subscribers.push_back(subscriber);
            }
            while (!state->stopping.load()) {
              std::string message;
              {
                std::unique_lock lock(subscriber->mutex);
                subscriber->cv.wait_for(lock, std::chrono::milliseconds(250), [&] {
                  return subscriber->stopped || !subscriber->queue.empty();
                });
                if (subscriber->stopped) break;
                if (subscriber->queue.empty()) continue;
                message = std::move(subscriber->queue.front());
                subscriber->queue.pop_front();
              }
              ws.write(asio::buffer(message));
            }
            beast::error_code ignored;
            ws.close(websocket::close_code::normal, ignored);
            return;
          }

          if (request.method() == http::verb::post && request_path == "/api/command") {
            nlohmann::json output;
            bool admitted{};
            {
              std::scoped_lock commands_lock(state->commands_mutex);
              if (!state->stopping.load(std::memory_order_acquire)) {
                ++state->active_commands;
                admitted = true;
              }
            }
            if (!admitted) {
              output = {{"ok", false}, {"error", "debug server is stopping"}};
            } else {
              try {
                output = state->commands ? state->commands(nlohmann::json::parse(request.body()))
                                         : nlohmann::json{{"ok", false}, {"error", "no command handler"}};
              } catch (const std::exception& e) {
                output = {{"ok", false}, {"error", e.what()}};
              } catch (...) {
                output = {{"ok", false}, {"error", "command handler failed"}};
              }
              {
                std::scoped_lock commands_lock(state->commands_mutex);
                --state->active_commands;
                state->commands_cv.notify_all();
              }
            }
            http::response<http::string_body> response{http::status::ok, request.version()};
            response.set(http::field::content_type, "application/json");
            response.body() = output.dump();
            response.prepare_payload();
            add_security_headers(response);
            http::write(socket, response);
            return;
          }

          auto path = request_path;
          if (path == "/") path = "/index.html";
          if (path.find("..") != std::string::npos || path.find('\\') != std::string::npos) {
            throw std::runtime_error("invalid static path");
          }
          const auto file = state->config.static_root / path.substr(1);
          http::response<http::string_body> response{http::status::ok, request.version()};
          response.set(http::field::content_type, mime_type(file));
          response.body() = read_file(file);
          response.prepare_payload();
          add_security_headers(response);
          http::write(socket, response);
        } catch (...) {
          beast::error_code ignored;
          socket.shutdown(tcp::socket::shutdown_both, ignored);
        }
      }).detach();
    }
  } catch (const std::exception& e) {
    std::scoped_lock startup_lock(state->startup_mutex);
    if (!state->startup_done) {
      state->startup_error = e.what();
      state->startup_done = true;
      state->startup_cv.notify_all();
    }
  }
}

}  // namespace dvo
