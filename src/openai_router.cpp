#include "dvo/openai_router.h"

#ifndef _WIN32
#error openai_router.cpp must only be built on Windows
#endif

#include <Windows.h>
#include <winhttp.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

namespace dvo {
namespace {

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

// Synchronously cancels an in-flight WinHTTP request handle at an exact
// steady_clock deadline. WinHttpSetTimeouts alone has ~500-1000ms internal
// timer granularity on synchronous WinHttpReceiveResponse, whereas closing the
// request handle from a watchdog thread immediately unblocks pending WinHTTP I/O.
class CancelableWinHttpRequest {
 public:
  CancelableWinHttpRequest(HINTERNET handle,
                           std::chrono::steady_clock::time_point deadline)
      : handle_(handle), deadline_(deadline) {
    if (handle_ != nullptr) {
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
    if (watchdog_.joinable()) {
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
};

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

[[nodiscard]] std::string lower_ascii(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const unsigned char ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      out.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else {
      out.push_back(static_cast<char>(ch));
    }
  }
  return out;
}

struct ParsedEndpointUrl {
  bool secure{};
  std::string host;
  INTERNET_PORT port{};
  std::string request_path;
};

[[nodiscard]] std::optional<ParsedEndpointUrl> parse_chat_completions_endpoint(
    std::string_view raw_url) {
  const std::string trimmed = trim_ascii(raw_url);
  std::string_view rest = trimmed;
  ParsedEndpointUrl out;
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
      slash_pos == std::string_view::npos ? rest : rest.substr(0, slash_pos);
  std::string path =
      slash_pos == std::string_view::npos ? std::string{} : std::string(rest.substr(slash_pos));

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
      if (res.ec != std::errc{} || res.ptr != port_str.data() + port_str.size() ||
          parsed_port == 0 || parsed_port > 65535) {
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
      if (res.ec != std::errc{} || res.ptr != port_str.data() + port_str.size() ||
          parsed_port == 0 || parsed_port > 65535) {
        return std::nullopt;
      }
      out.port = static_cast<INTERNET_PORT>(parsed_port);
    } else {
      out.host = std::string(authority);
    }
  }

  if (out.host.empty()) return std::nullopt;

  // Preserve optional query string if present, while appending /chat/completions to the path.
  std::string query;
  if (const auto qpos = path.find('?'); qpos != std::string::npos) {
    query = path.substr(qpos);
    path.resize(qpos);
  }
  while (!path.empty() && path.back() == '/') {
    path.pop_back();
  }
  constexpr std::string_view kCompletionsSuffix = "/chat/completions";
  if (!std::string_view(path).ends_with(kCompletionsSuffix)) {
    path += kCompletionsSuffix;
  }
  out.request_path = path + query;
  return out;
}

[[nodiscard]] bool is_loopback_host(std::string_view host) {
  const std::string lower = lower_ascii(host);
  return lower == "127.0.0.1" || lower == "localhost" ||
         lower == "::1" || lower.starts_with("127.");
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

[[nodiscard]] std::string safe_body_snippet(std::string_view body) {
  std::string snippet;
  snippet.reserve(160);
  for (const unsigned char ch : body) {
    if (snippet.size() >= 160) break;
    if (ch == '\r' || ch == '\n' || ch == '\t') {
      if (!snippet.empty() && snippet.back() != ' ') snippet.push_back(' ');
    } else if (ch >= 0x20 && ch != 0x7F) {
      snippet.push_back(static_cast<char>(ch));
    }
  }
  return trim_ascii(snippet);
}

[[nodiscard]] std::string strip_markdown_code_fence(std::string_view content) {
  std::string trimmed = trim_ascii(content);
  if (!trimmed.starts_with("```")) return trimmed;
  const std::size_t first_newline = trimmed.find('\n');
  if (first_newline == std::string::npos) return trimmed;
  std::string inner = trimmed.substr(first_newline + 1);
  if (const std::size_t closing = inner.rfind("```"); closing != std::string::npos) {
    inner.resize(closing);
  }
  return trim_ascii(inner);
}

[[nodiscard]] std::optional<std::string> validate_schema_node(
    const nlohmann::json& node,
    const std::string& path) {
  if (!node.is_object()) {
    return path + " must be a JSON object";
  }

  bool is_object_schema = node.contains("properties");
  if (const auto type_it = node.find("type"); type_it != node.end()) {
    if (type_it->is_string() && *type_it == "object") {
      is_object_schema = true;
    } else if (type_it->is_array()) {
      for (const auto& item : *type_it) {
        if (item.is_string() && item == "object") {
          is_object_schema = true;
          break;
        }
      }
    }
  }

  if (is_object_schema) {
    const auto add_props = node.find("additionalProperties");
    if (add_props == node.end() || !add_props->is_boolean() ||
        add_props->get<bool>() != false) {
      return path + " must specify \"additionalProperties\": false";
    }

    const auto props = node.find("properties");
    if (props == node.end() || !props->is_object()) {
      return path + " object schema must define a \"properties\" object";
    }

    const auto req = node.find("required");
    if (req == node.end() || !req->is_array()) {
      return path + " object schema must define a \"required\" array";
    }

    std::unordered_set<std::string> required_keys;
    for (const auto& item : *req) {
      if (!item.is_string()) {
        return path + ".required entries must be strings";
      }
      const auto key = item.get<std::string>();
      if (!required_keys.insert(key).second) {
        return path + ".required contains duplicate key '" + key + "'";
      }
      if (!props->contains(key)) {
        return path + ".required lists unknown property '" + key + "'";
      }
    }

    for (const auto& [prop_name, prop_schema] : props->items()) {
      if (!required_keys.contains(prop_name)) {
        return path + ".required is missing property '" + prop_name + "'";
      }
      if (const auto err =
              validate_schema_node(prop_schema, path + ".properties." + prop_name)) {
        return err;
      }
    }
  }

  if (const auto items = node.find("items"); items != node.end()) {
    if (items->is_object()) {
      if (const auto err = validate_schema_node(*items, path + ".items")) {
        return err;
      }
    }
  }

  for (const char* combinator : {"anyOf", "oneOf", "allOf"}) {
    if (const auto comb = node.find(combinator);
        comb != node.end() && comb->is_array()) {
      for (std::size_t i = 0; i < comb->size(); ++i) {
        if (const auto err = validate_schema_node(
                (*comb)[i], path + "." + combinator + "[" + std::to_string(i) + "]")) {
          return err;
        }
      }
    }
  }

  for (const char* defs_key : {"$defs", "definitions"}) {
    if (const auto defs = node.find(defs_key);
        defs != node.end() && defs->is_object()) {
      for (const auto& [name, sub] : defs->items()) {
        if (const auto err =
                validate_schema_node(sub, path + "." + defs_key + "." + name)) {
          return err;
        }
      }
    }
  }

  return std::nullopt;
}

[[nodiscard]] std::string make_router_version_tag(std::string_view model) {
  std::string tag = "openai:";
  for (const unsigned char ch : model) {
    if (tag.size() >= 60) break;
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' ||
        ch == '/' || ch == ':') {
      tag.push_back(static_cast<char>(ch));
    }
  }
  return tag == "openai:" ? std::string("openai.router") : tag;
}

}  // namespace

nlohmann::json build_openai_router_response_schema() {
  return {
      {"type", "object"},
      {"additionalProperties", false},
      {"properties",
       {{"actions",
         {{"type", "array"},
          {"items",
           {{"type", "object"},
            {"additionalProperties", false},
            {"properties",
             {{"type",
               {{"type", "string"},
                {"enum",
                 {"media.play",
                  "media.pause",
                  "audio.volume.adjust",
                  "media.next",
                  "media.like",
                  "media.play_daily",
                  "media.mode.order",
                  "media.mode.list_loop",
                  "media.mode.single_loop",
                  "media.mode.shuffle",
                  "media.lyrics.show",
                  "media.lyrics.hide",
                  "media.search_play",
                  "app.open"}}}},
              {"volume_delta_percent", {{"type", {"integer", "null"}}}},
              {"search_query", {{"type", {"string", "null"}}}},
              {"search_type",
               {{"type", {"string", "null"}},
                {"enum", {"song", "playlist", nullptr}}}},
              {"search_artist", {{"type", {"string", "null"}}}},
              {"app_id", {{"type", {"string", "null"}}}}}},
            {"required",
             {"type",
              "volume_delta_percent",
              "search_query",
              "search_type",
              "search_artist",
              "app_id"}}}}}}}},
      {"required", {"actions"}},
  };
}

std::optional<std::string> validate_openai_strict_json_schema(
    const nlohmann::json& schema) {
  return validate_schema_node(schema, "$");
}

std::string build_openai_router_system_prompt(
    const LlmRouteRequest& request,
    const AppCatalog* app_catalog) {
  std::ostringstream out;
  out << "你是 Windows 桌面语音编排器的极速意图规划器。请将用户话语转换为严格的 JSON：{\"actions\":[...]}。\n"
      << "核心规则：\n"
      << "1. 日常闲聊、感叹句、歌词背景音、提问、或自我否定/取消意图（如“算了别切了”、“不用打开了”、“这首歌真好听”）必须返回 {\"actions\":[]}。\n"
      << "2. 多步指令必须严格按用户口述先后顺序排列，最多 " << request.max_actions << " 个动作；未使用的字段一律填 null。\n"
      << "3. 受控动作与参数：\n"
      << "   - media.play / media.pause / media.next / media.like / media.play_daily / media.mode.order / media.mode.list_loop / media.mode.single_loop / media.mode.shuffle / media.lyrics.show / media.lyrics.hide：全部可选参数填 null。\n"
      << "   - audio.volume.adjust：volume_delta_percent 填非零整数（范围 -"
      << request.maximum_volume_delta_percent << ".." << request.maximum_volume_delta_percent
      << "，未说明具体数值如“小声点/大声点”时默认填 -" << request.default_volume_delta_percent
      << " 或 " << request.default_volume_delta_percent << "）。\n"
      << "   - media.search_play：search_query 填歌名或歌单关键词，search_type 填 \"song\" 或 \"playlist\"，search_artist 填歌手名或 null。\n"
      << "   - app.open：app_id 只能填写下方已注册应用白名单中的 id（支持根据用途推断，如“挂代理的软件”对应 clash；若不在白名单中则返回空数组）。\n";

  out << "已注册应用白名单（id: 名称 [别名]）：\n";
  if (!app_catalog || app_catalog->entries().empty()) {
    out << "- （当前无已注册应用）";
  } else {
    constexpr std::size_t kMaxPromptApps = 60;
    const auto& entries = app_catalog->entries();
    const std::size_t limit = std::min(entries.size(), kMaxPromptApps);
    for (std::size_t i = 0; i < limit; ++i) {
      const auto& app = entries[i];
      out << "- " << app.id << ": " << app.display_name;
      if (!app.aliases.empty()) {
        out << " (";
        const std::size_t alias_limit = std::min<std::size_t>(app.aliases.size(), 5);
        for (std::size_t j = 0; j < alias_limit; ++j) {
          if (j != 0) out << ", ";
          out << app.aliases[j];
        }
        out << ")";
      }
      if (i + 1 < limit) out << '\n';
    }
  }
  return out.str();
}

OpenAiCompatibleCommandRouter::OpenAiCompatibleCommandRouter(
    OpenAiRouterConfig config,
    std::shared_ptr<const AppCatalog> app_catalog)
    : config_(std::move(config)),
      app_catalog_(std::move(app_catalog)),
      response_schema_(build_openai_router_response_schema()) {
  if (const auto schema_error =
          validate_openai_strict_json_schema(response_schema_)) {
    throw std::logic_error("OpenAI router response schema violated strict mode: " +
                           *schema_error);
  }
}

LlmRouteResult OpenAiCompatibleCommandRouter::route(
    const LlmRouteRequest& request) const {
  const std::string router_tag = make_router_version_tag(config_.model);
  const auto fail = [&](std::string code, std::string message) -> LlmRouteResult {
    return LlmRouteResult{
        {},
        CommandParseError{std::move(code), std::move(message), 0},
        router_tag};
  };

  const auto endpoint = parse_chat_completions_endpoint(config_.base_url);
  if (!endpoint.has_value()) {
    return fail("router_invalid_endpoint",
                "commands.router_base_url is not a valid HTTP/HTTPS endpoint");
  }

  std::string api_key = read_env_var_trimmed(config_.api_key_env);
  if (api_key.empty()) {
    if (is_loopback_host(endpoint->host)) {
      api_key = "local-no-key";
    } else {
      return fail("router_missing_api_key",
                  "environment variable '" + config_.api_key_env +
                      "' is empty or unset for remote router endpoint");
    }
  }

  nlohmann::json payload = {
      {"model", config_.model},
      {"temperature", 0},
      {"max_tokens", 256},
      {"messages",
       nlohmann::json::array({
           {{"role", "system"},
            {"content",
             build_openai_router_system_prompt(request, app_catalog_.get())}},
           {{"role", "user"}, {"content", request.stripped_text}},
       })},
  };
  if (config_.structured_outputs) {
    payload["response_format"] = {
        {"type", "json_schema"},
        {"json_schema",
         {{"name", "dvo_command_plan"},
          {"strict", true},
          {"schema", response_schema_}}},
    };
  } else {
    payload["response_format"] = {{"type", "json_object"}};
  }

  const std::string request_body = payload.dump();

  const auto total_timeout = std::chrono::milliseconds(
      std::clamp<std::uint32_t>(config_.timeout_ms, 200U, 15000U));
  const auto deadline = std::chrono::steady_clock::now() + total_timeout;

  const auto remaining_timeout_ms = [&]() -> std::optional<int> {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return std::nullopt;
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
            .count();
    return static_cast<int>(std::clamp<long long>(ms, 1, 15000));
  };

  ScopedWinHttpHandle session(WinHttpOpen(
      L"desktop-voice-orchestrator/0.2",
      WINHTTP_ACCESS_TYPE_NO_PROXY,
      WINHTTP_NO_PROXY_NAME,
      WINHTTP_NO_PROXY_BYPASS,
      0));
  if (!is_loopback_host(endpoint->host)) {
    session = ScopedWinHttpHandle(WinHttpOpen(
        L"desktop-voice-orchestrator/0.2",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0));
  }
  if (!session) {
    return fail("router_network_error", "WinHttpOpen failed");
  }

  const auto initial_ms = remaining_timeout_ms();
  if (!initial_ms) {
    return fail("router_timeout", "router request exceeded deadline before connect");
  }
  WinHttpSetTimeouts(session.get(), *initial_ms, *initial_ms, *initial_ms,
                     *initial_ms);

  const std::wstring wide_host = utf8_to_wide(endpoint->host);
  ScopedWinHttpHandle connection(
      WinHttpConnect(session.get(), wide_host.c_str(), endpoint->port, 0));
  if (!connection) {
    return fail("router_network_error", "WinHttpConnect failed");
  }

  const std::wstring wide_path = utf8_to_wide(endpoint->request_path);
  const DWORD request_flags = endpoint->secure ? WINHTTP_FLAG_SECURE : 0;
  HINTERNET raw_request = WinHttpOpenRequest(
      connection.get(), L"POST", wide_path.c_str(), nullptr,
      WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, request_flags);
  if (!raw_request) {
    return fail("router_network_error", "WinHttpOpenRequest failed");
  }
  CancelableWinHttpRequest http_request(raw_request, deadline);

  // Construct headers in memory only; never log or surface api_key.
  std::wstring headers =
      L"Content-Type: application/json\r\nAuthorization: Bearer " +
      utf8_to_wide(api_key) + L"\r\n";
  // Scrub api_key string once copied into wide header buffer.
  std::fill(api_key.begin(), api_key.end(), '\0');
  api_key.clear();

  const auto classify_winhttp_failure = [&](const char* step_label) -> LlmRouteResult {
    const DWORD err = GetLastError();
    if (http_request.timed_out() ||
        err == ERROR_WINHTTP_TIMEOUT ||
        err == ERROR_WINHTTP_OPERATION_CANCELLED ||
        err == ERROR_INVALID_HANDLE ||
        std::chrono::steady_clock::now() >= deadline) {
      return fail("router_timeout",
                  std::string("router request timed out during ") + step_label);
    }
    return fail("router_network_error",
                std::string(step_label) + " failed (WinHTTP error " +
                    std::to_string(err) + ")");
  };

  if (const auto ms = remaining_timeout_ms()) {
    WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);
  } else {
    std::fill(headers.begin(), headers.end(), L'\0');
    return fail("router_timeout", "router request exceeded deadline before send");
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
    return classify_winhttp_failure("WinHttpSendRequest");
  }

  if (const auto ms = remaining_timeout_ms()) {
    WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);
  } else {
    return fail("router_timeout", "router request exceeded deadline while sending");
  }

  if (!WinHttpReceiveResponse(http_request.get(), nullptr)) {
    return classify_winhttp_failure("WinHttpReceiveResponse");
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
    return classify_winhttp_failure("WinHttpQueryHeaders");
  }

  std::string response_body;
  constexpr std::size_t kMaxResponseBytes = 256 * 1024;
  for (;;) {
    const auto ms = remaining_timeout_ms();
    if (!ms) {
      return fail("router_timeout", "router request exceeded deadline while reading body");
    }
    WinHttpSetTimeouts(http_request.get(), *ms, *ms, *ms, *ms);

    DWORD available = 0;
    if (!WinHttpQueryDataAvailable(http_request.get(), &available)) {
      return classify_winhttp_failure("WinHttpQueryDataAvailable");
    }
    if (available == 0) break;

    const DWORD to_read = static_cast<DWORD>(std::min<std::size_t>(available, 16384));
    if (response_body.size() + to_read > kMaxResponseBytes) {
      return fail("router_invalid_response",
                  "router HTTP response exceeded maximum allowed size");
    }
    const std::size_t offset = response_body.size();
    response_body.resize(offset + to_read);
    DWORD downloaded = 0;
    if (!WinHttpReadData(http_request.get(), response_body.data() + offset,
                         to_read, &downloaded)) {
      return classify_winhttp_failure("WinHttpReadData");
    }
    response_body.resize(offset + downloaded);
    if (downloaded == 0) break;
  }
  http_request.close();

  if (status_code != 200) {
    std::string msg = "router endpoint returned HTTP " + std::to_string(status_code);
    const std::string snippet = safe_body_snippet(response_body);
    if (!snippet.empty()) {
      msg += ": " + snippet;
    }
    return fail("router_http_error", std::move(msg));
  }

  const auto envelope = nlohmann::json::parse(response_body, nullptr, false);
  if (envelope.is_discarded() || !envelope.is_object()) {
    return fail("router_invalid_json",
                "router HTTP 200 body is not a valid JSON object");
  }

  const auto choices_it = envelope.find("choices");
  if (choices_it == envelope.end() || !choices_it->is_array() ||
      choices_it->empty() || !(*choices_it)[0].is_object()) {
    return fail("router_invalid_response",
                "chat completion response is missing choices[0]");
  }

  const auto message_it = (*choices_it)[0].find("message");
  if (message_it == (*choices_it)[0].end() || !message_it->is_object()) {
    return fail("router_invalid_response",
                "chat completion response is missing choices[0].message");
  }

  const auto content_it = message_it->find("content");
  if (content_it == message_it->end() || content_it->is_null()) {
    if (const auto refusal_it = message_it->find("refusal");
        refusal_it != message_it->end() && refusal_it->is_string() &&
        !refusal_it->get<std::string>().empty()) {
      return LlmRouteResult{{}, std::nullopt, router_tag};
    }
    return fail("router_invalid_response",
                "choices[0].message.content is null or missing");
  }
  if (!content_it->is_string()) {
    return fail("router_invalid_response",
                "choices[0].message.content must be a string");
  }

  const std::string raw_content =
      strip_markdown_code_fence(content_it->get<std::string>());
  const auto plan_json = nlohmann::json::parse(raw_content, nullptr, false);
  if (plan_json.is_discarded() || !plan_json.is_object()) {
    return fail("router_invalid_json",
                "assistant message content is not a valid JSON object");
  }

  const auto actions_it = plan_json.find("actions");
  if (actions_it == plan_json.end() || !actions_it->is_array()) {
    return fail("router_invalid_schema",
                "assistant JSON object must contain an 'actions' array");
  }

  LlmRouteResult out;
  out.router_name = router_tag;
  out.actions.reserve(actions_it->size());

  for (std::size_t i = 0; i < actions_it->size(); ++i) {
    const auto& item = (*actions_it)[i];
    if (!item.is_object()) {
      return LlmRouteResult{
          {},
          CommandParseError{"router_invalid_schema",
                            "each element in actions must be an object",
                            i},
          router_tag};
    }

    const auto type_it = item.find("type");
    if (type_it == item.end() || !type_it->is_string()) {
      return LlmRouteResult{
          {},
          CommandParseError{"invalid_action_type",
                            "action item is missing string 'type'",
                            i},
          router_tag};
    }

    const auto parsed_type = parse_action_type(type_it->get<std::string>());
    if (!parsed_type.has_value()) {
      return LlmRouteResult{
          {},
          CommandParseError{"invalid_action_type",
                            "router returned unsupported action type '" +
                                type_it->get<std::string>() + "'",
                            i},
          router_tag};
    }

    ProposedAction proposed;
    proposed.type = *parsed_type;

    if (const auto vol_it = item.find("volume_delta_percent");
        vol_it != item.end() && !vol_it->is_null()) {
      if (!vol_it->is_number_integer()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_volume_amount",
                              "volume_delta_percent must be an integer or null",
                              i},
            router_tag};
      }
      proposed.volume_delta_percent = vol_it->get<int>();
    }

    if (const auto query_it = item.find("search_query");
        query_it != item.end() && !query_it->is_null()) {
      if (!query_it->is_string()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_search_query",
                              "search_query must be a string or null",
                              i},
            router_tag};
      }
      proposed.search_query = query_it->get<std::string>();
    }

    if (const auto stype_it = item.find("search_type");
        stype_it != item.end() && !stype_it->is_null()) {
      if (!stype_it->is_string()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_search_type",
                              "search_type must be 'song', 'playlist', or null",
                              i},
            router_tag};
      }
      const auto parsed_stype =
          parse_media_search_type(stype_it->get<std::string>());
      if (!parsed_stype.has_value()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_search_type",
                              "search_type must be 'song' or 'playlist'",
                              i},
            router_tag};
      }
      proposed.search_type = *parsed_stype;
    }

    if (const auto artist_it = item.find("search_artist");
        artist_it != item.end() && !artist_it->is_null()) {
      if (!artist_it->is_string()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_search_artist",
                              "search_artist must be a string or null",
                              i},
            router_tag};
      }
      proposed.search_artist = artist_it->get<std::string>();
    }

    if (const auto app_it = item.find("app_id");
        app_it != item.end() && !app_it->is_null()) {
      if (!app_it->is_string()) {
        return LlmRouteResult{
            {},
            CommandParseError{"invalid_app_id",
                              "app_id must be a string or null",
                              i},
            router_tag};
      }
      std::string raw_app_id = trim_ascii(app_it->get<std::string>());
      if (proposed.type == ActionType::app_open && app_catalog_) {
        if (const auto* exact = app_catalog_->find_by_id(raw_app_id)) {
          raw_app_id = exact->id;
          proposed.app_display_name = exact->display_name;
        } else if (const auto* by_alias = app_catalog_->match_alias(raw_app_id)) {
          raw_app_id = by_alias->id;
          proposed.app_display_name = by_alias->display_name;
        }
      }
      proposed.app_id = std::move(raw_app_id);
    }

    out.actions.push_back(std::move(proposed));
  }

  return out;
}

}  // namespace dvo
