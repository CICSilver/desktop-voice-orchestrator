#include <catch2/catch_test_macros.hpp>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dvo/app_catalog.h"
#include "dvo/command_parser.h"
#include "dvo/config.h"
#include "dvo/openai_router.h"
#include "dvo/runtime.h"

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

struct RecordedHttpRequest {
  std::string method;
  std::string target;
  std::string authorization;
  std::string content_type;
  std::string body;
};

struct MockHttpReply {
  unsigned status{200};
  std::string content_type{"application/json"};
  std::string body;
  std::chrono::milliseconds delay{0};
};

class LoopbackOpenAiServer {
 public:
  using Handler = std::function<MockHttpReply(const RecordedHttpRequest&)>;

  explicit LoopbackOpenAiServer(Handler handler)
      : handler_(std::move(handler)),
        acceptor_(io_, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0)) {
    port_ = acceptor_.local_endpoint().port();
    worker_ = std::jthread([this](std::stop_token stop) { serve(stop); });
  }

  ~LoopbackOpenAiServer() { stop(); }

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

  [[nodiscard]] std::string base_url(std::string_view suffix = "/v1") const {
    return "http://127.0.0.1:" + std::to_string(port_) + std::string(suffix);
  }

  [[nodiscard]] std::vector<RecordedHttpRequest> requests() const {
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

        RecordedHttpRequest recorded;
        recorded.method = std::string(req.method_string());
        recorded.target = std::string(req.target());
        if (const auto it = req.find(http::field::authorization); it != req.end()) {
          recorded.authorization = std::string(it->value());
        }
        if (const auto it = req.find(http::field::content_type); it != req.end()) {
          recorded.content_type = std::string(it->value());
        }
        recorded.body = req.body();

        {
          std::scoped_lock lock(mutex_);
          requests_.push_back(recorded);
        }

        const MockHttpReply reply = handler_(recorded);
        if (reply.delay.count() > 0) {
          const auto deadline = std::chrono::steady_clock::now() + reply.delay;
          while (!stop.stop_requested() &&
                 !stopped_.load(std::memory_order_acquire) &&
                 std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }

        http::response<http::string_body> res{
            static_cast<http::status>(reply.status), req.version()};
        res.set(http::field::content_type, reply.content_type);
        res.keep_alive(false);
        res.body() = reply.body;
        res.prepare_payload();
        beast::error_code write_ec;
        http::write(stream, res, write_ec);
        stream.socket().shutdown(tcp::socket::shutdown_send, write_ec);
      }).detach();
    }
  }

  Handler handler_;
  asio::io_context io_;
  tcp::acceptor acceptor_;
  std::uint16_t port_{};
  std::atomic<bool> stopped_{false};
  mutable std::mutex mutex_;
  std::vector<RecordedHttpRequest> requests_;
  std::jthread worker_;
};

// Independent recursive OpenAI Strict JSON Schema assertion in test code to
// cross-check production schema generation.
void assert_strict_openai_json_schema_recursive(const nlohmann::json& node,
                                                const std::string& path = "$") {
  CAPTURE(path);
  REQUIRE(node.is_object());
  if (const auto type_it = node.find("type");
      type_it != node.end() && type_it->is_string() && *type_it == "object") {
    REQUIRE(node.contains("additionalProperties"));
    REQUIRE(node["additionalProperties"].is_boolean());
    CHECK(node["additionalProperties"].get<bool>() == false);

    REQUIRE(node.contains("properties"));
    REQUIRE(node["properties"].is_object());
    REQUIRE(node.contains("required"));
    REQUIRE(node["required"].is_array());

    std::vector<std::string> prop_keys;
    for (const auto& [key, child] : node["properties"].items()) {
      prop_keys.push_back(key);
      if (child.is_object()) {
        assert_strict_openai_json_schema_recursive(child, path + ".properties." + key);
      }
    }
    std::vector<std::string> req_keys;
    for (const auto& item : node["required"]) {
      REQUIRE(item.is_string());
      req_keys.push_back(item.get<std::string>());
    }
    CHECK(req_keys.size() == prop_keys.size());
    for (const auto& key : prop_keys) {
      CAPTURE(key);
      CHECK(std::ranges::find(req_keys, key) != req_keys.end());
    }
  }
  if (const auto items_it = node.find("items");
      items_it != node.end() && items_it->is_object()) {
    assert_strict_openai_json_schema_recursive(*items_it, path + ".items");
  }
}

std::string chat_completion_envelope(const nlohmann::json& content_obj) {
  return nlohmann::json{
      {"id", "chatcmpl-test-1"},
      {"object", "chat.completion"},
      {"choices",
       nlohmann::json::array({
           {{"index", 0},
            {"message",
             {{"role", "assistant"}, {"content", content_obj.dump()}}},
            {"finish_reason", "stop"}},
       })},
  }
      .dump();
}

std::shared_ptr<const dvo::AppCatalog> sample_catalog() {
  return std::make_shared<const dvo::AppCatalog>(std::vector<dvo::AppCatalogEntry>{
      dvo::AppCatalogEntry{
          .id = "clash",
          .display_name = "Clash",
          .source = dvo::AppEntrySource::configured,
          .target = "clash-verge.exe",
          .arguments = {},
          .aliases = {"clash", "代理", "克拉什"},
          .process_names = {"clash-verge.exe"},
      },
      dvo::AppCatalogEntry{
          .id = "steam",
          .display_name = "Steam",
          .source = dvo::AppEntrySource::configured,
          .target = "steam.exe",
          .arguments = {},
          .aliases = {"steam", "蒸汽平台", "死盯"},
          .process_names = {"steam.exe"},
      },
  });
}

dvo::CommandParseContext make_wake_context(std::string id) {
  dvo::CommandParseContext ctx;
  ctx.runtime_session_id = "router-test-session";
  ctx.utterance_id = std::move(id);
  ctx.origin = dvo::UtteranceOrigin::keyword;
  ctx.wake_word = "小克";
  ctx.wake_position = "prefix";
  ctx.source = dvo::ExecutionSource::live;
  ctx.execution_mode = dvo::ExecutionMode::dry_run;
  return ctx;
}

}  // namespace

TEST_CASE("OpenAI router strict JSON schema helper catches violations and validates production schema") {
  const auto schema = dvo::build_openai_router_response_schema();
  CHECK_FALSE(dvo::validate_openai_strict_json_schema(schema).has_value());
  assert_strict_openai_json_schema_recursive(schema);

  auto missing_additional = schema;
  missing_additional.erase("additionalProperties");
  CHECK(dvo::validate_openai_strict_json_schema(missing_additional).has_value());

  auto missing_required_field = schema;
  missing_required_field["properties"]["actions"]["items"]["required"] =
      nlohmann::json::array({"type"});
  CHECK(dvo::validate_openai_strict_json_schema(missing_required_field).has_value());
}

TEST_CASE("OpenAiCompatibleCommandRouter sends strict schema, preserves compound order, and rejects chitchat") {
  REQUIRE(_putenv_s("DVO_TEST_LOOPBACK_NO_KEY", "") == 0);

  LoopbackOpenAiServer server([](const RecordedHttpRequest& req) -> MockHttpReply {
    const auto payload = nlohmann::json::parse(req.body);
    const std::string user_content =
        payload["messages"].at(1).value("content", "");
    if (user_content.find("算了别切了") != std::string::npos ||
        user_content.find("这首歌真好听") != std::string::npos) {
      return MockHttpReply{
          .status = 200,
          .body = chat_completion_envelope(nlohmann::json{{"actions", nlohmann::json::array()}}),
      };
    }
    return MockHttpReply{
        .status = 200,
        .body = chat_completion_envelope(nlohmann::json{
            {"actions",
             nlohmann::json::array({
                 {{"type", "audio.volume.adjust"},
                  {"volume_delta_percent", -5},
                  {"search_query", nullptr},
                  {"search_type", nullptr},
                  {"search_artist", nullptr},
                  {"app_id", nullptr}},
                 {{"type", "app.open"},
                  {"volume_delta_percent", nullptr},
                  {"search_query", nullptr},
                  {"search_type", nullptr},
                  {"search_artist", nullptr},
                  {"app_id", "clash"}},
                 {{"type", "media.search_play"},
                  {"volume_delta_percent", nullptr},
                  {"search_query", "晴天"},
                  {"search_type", "song"},
                  {"search_artist", "周杰伦"},
                  {"app_id", nullptr}},
             })},
        }),
    };
  });

  dvo::AppConfig app_config;
  app_config.commands.parser_mode = "hybrid";
  app_config.commands.router_backend = "openai_compatible";
  app_config.commands.router_base_url = server.base_url("/v1");
  app_config.commands.router_model = "gpt-4o-mini-test";
  app_config.commands.router_api_key_env = "DVO_TEST_LOOPBACK_NO_KEY";
  app_config.commands.router_timeout_ms = 2500;
  app_config.commands.router_structured_outputs = true;

  const auto catalog = sample_catalog();
  const auto parser = dvo::create_command_parser(app_config, catalog);

  // 1. Compound natural-language utterance falls through hybrid fast-path to router.
  const auto compound = parser->parse(
      "小克，先把声音关小一点，再打开挂代理的软件，然后放周杰伦的晴天",
      make_wake_context("compound-1"));
  REQUIRE(compound.ok());
  REQUIRE(compound.plan->actions.size() == 3);
  CHECK(compound.plan->actions[0].sequence == 1);
  CHECK(compound.plan->actions[0].type == dvo::ActionType::master_volume_adjust);
  CHECK(compound.plan->actions[0].volume_delta_percent == -5);

  CHECK(compound.plan->actions[1].sequence == 2);
  CHECK(compound.plan->actions[1].type == dvo::ActionType::app_open);
  CHECK(compound.plan->actions[1].app_id == "clash");
  CHECK(compound.plan->actions[1].app_display_name == "Clash");

  CHECK(compound.plan->actions[2].sequence == 3);
  CHECK(compound.plan->actions[2].type == dvo::ActionType::media_search_play);
  CHECK(compound.plan->actions[2].search_query == "晴天");
  CHECK(compound.plan->actions[2].search_type == dvo::MediaSearchType::song);
  CHECK(compound.plan->actions[2].search_artist == "周杰伦");

  // Inspect actual HTTP request sent over the wire.
  const auto recorded = server.requests();
  REQUIRE(recorded.size() == 1);
  CHECK(recorded[0].method == "POST");
  CHECK(recorded[0].target == "/v1/chat/completions");
  CHECK(recorded[0].authorization == "Bearer local-no-key");
  const auto sent_json = nlohmann::json::parse(recorded[0].body);
  CHECK(sent_json["model"] == "gpt-4o-mini-test");
  CHECK(sent_json["temperature"] == 0);
  REQUIRE(sent_json["messages"].is_array());
  REQUIRE(sent_json["messages"].size() == 2);
  const std::string sys_prompt = sent_json["messages"][0].value("content", "");
  CHECK(sys_prompt.find("clash: Clash") != std::string::npos);
  CHECK(sys_prompt.find("steam: Steam") != std::string::npos);
  REQUIRE(sent_json["response_format"]["type"] == "json_schema");
  CHECK(sent_json["response_format"]["json_schema"]["strict"] == true);
  assert_strict_openai_json_schema_recursive(
      sent_json["response_format"]["json_schema"]["schema"]);

  // 2. Self-negation and chitchat returning {"actions": []} are cleanly rejected.
  const auto self_negated =
      parser->parse("小克，算了别切了", make_wake_context("neg-1"));
  CHECK_FALSE(self_negated.ok());
  REQUIRE(self_negated.error.has_value());
  CHECK(self_negated.error->code == "no_command_intent");

  const auto chitchat =
      parser->parse("小克，这首歌真好听", make_wake_context("chat-1"));
  CHECK_FALSE(chitchat.ok());
  REQUIRE(chitchat.error.has_value());
  CHECK(chitchat.error->code == "no_command_intent");

  // 3. Explicit /chat/completions suffix is not duplicated and json_object fallback works.
  app_config.commands.router_base_url =
      server.base_url("/custom/chat/completions");
  app_config.commands.router_structured_outputs = false;
  const auto fallback_parser = dvo::create_command_parser(app_config, catalog);
  const auto fallback_res = fallback_parser->parse(
      "小克，帮我把代理打开", make_wake_context("fallback-json-obj"));
  REQUIRE(fallback_res.ok());
  const auto all_requests = server.requests();
  REQUIRE(all_requests.size() == 4);
  CHECK(all_requests.back().target == "/custom/chat/completions");
  const auto fallback_body = nlohmann::json::parse(all_requests.back().body);
  CHECK(fallback_body["response_format"] ==
        nlohmann::json{{"type", "json_object"}});
}

TEST_CASE("OpenAiCompatibleCommandRouter enforces missing remote API key, strict deadline timeout, and HTTP/JSON error safety") {
  // 1. Remote URL without API key returns router_missing_api_key immediately without network I/O.
  REQUIRE(_putenv_s("DVO_TEST_MISSING_REMOTE_KEY", "") == 0);
  dvo::OpenAiRouterConfig missing_key_cfg;
  missing_key_cfg.base_url = "https://api.openai.com/v1";
  missing_key_cfg.api_key_env = "DVO_TEST_MISSING_REMOTE_KEY";
  missing_key_cfg.timeout_ms = 2500;
  const dvo::OpenAiCompatibleCommandRouter missing_key_router(missing_key_cfg);

  dvo::LlmRouteRequest req;
  req.raw_text = "小克，把代理打开";
  req.stripped_text = "把代理打开";
  const auto t0 = std::chrono::steady_clock::now();
  const auto missing_res = missing_key_router.route(req);
  const auto missing_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0);
  REQUIRE(missing_res.error.has_value());
  CHECK(missing_res.error->code == "router_missing_api_key");
  CHECK(missing_elapsed.count() < 100);

  // 2. Slow server exceeding router_timeout_ms triggers router_timeout promptly.
  LoopbackOpenAiServer slow_server([](const RecordedHttpRequest&) -> MockHttpReply {
    return MockHttpReply{
        .status = 200,
        .body = chat_completion_envelope(nlohmann::json{{"actions", nlohmann::json::array()}}),
        .delay = std::chrono::milliseconds(700),
    };
  });

  dvo::OpenAiRouterConfig timeout_cfg;
  timeout_cfg.base_url = slow_server.base_url("/v1");
  timeout_cfg.api_key_env = "DVO_TEST_MISSING_REMOTE_KEY";
  timeout_cfg.timeout_ms = 250;
  const dvo::OpenAiCompatibleCommandRouter timeout_router(timeout_cfg);

  const auto timeout_start = std::chrono::steady_clock::now();
  const auto timeout_res = timeout_router.route(req);
  const auto timeout_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - timeout_start);
  REQUIRE(timeout_res.error.has_value());
  CHECK(timeout_res.error->code == "router_timeout");
  CHECK(timeout_elapsed.count() < 600);
  slow_server.stop();

  // 3. HTTP 500 and malformed JSON degrade gracefully and never leak API keys.
  const std::string secret = "sk-dvo-top-secret-never-leak-987654";
  REQUIRE(_putenv_s("DVO_TEST_SECRET_API_KEY", secret.c_str()) == 0);

  std::atomic<int> mode{0};
  LoopbackOpenAiServer error_server([&](const RecordedHttpRequest& r) -> MockHttpReply {
    CHECK(r.authorization == "Bearer " + secret);
    if (mode.load() == 0) {
      return MockHttpReply{
          .status = 500,
          .body = R"({"error":{"message":"upstream gateway exploded"}})",
      };
    }
    return MockHttpReply{
        .status = 200,
        .body = R"({"choices":[{"message":{"content":"{truncated_json"}}]})",
    };
  });

  dvo::OpenAiRouterConfig err_cfg;
  err_cfg.base_url = error_server.base_url("/v1");
  err_cfg.api_key_env = "DVO_TEST_SECRET_API_KEY";
  err_cfg.timeout_ms = 1500;
  const dvo::OpenAiCompatibleCommandRouter err_router(err_cfg);

  mode.store(0);
  const auto http_500 = err_router.route(req);
  REQUIRE(http_500.error.has_value());
  CHECK(http_500.error->code == "router_http_error");
  CHECK(http_500.error->message.find(secret) == std::string::npos);

  mode.store(1);
  const auto bad_json = err_router.route(req);
  REQUIRE(bad_json.error.has_value());
  CHECK(bad_json.error->code == "router_invalid_json");
  CHECK(bad_json.error->message.find(secret) == std::string::npos);

  (void)_putenv_s("DVO_TEST_SECRET_API_KEY", "");
}
