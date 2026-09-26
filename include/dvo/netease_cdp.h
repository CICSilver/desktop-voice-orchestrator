#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace dvo {

// Controls the NetEase Cloud Music client from inside its own page through
// the Chrome DevTools Protocol. The channel exists only when cloudmusic.exe
// was started with --remote-debugging-address=127.0.0.1
// --remote-debugging-port=<port>; it accepts local connections only. The
// scripts dispatch the same actions as the client's own buttons (tested with
// client 3.1.41), so nothing is shown and no window is brought forward.
class NeteaseCdpError : public std::runtime_error {
 public:
  enum class Kind {
    unavailable,  // no channel: the client is not running with the port
    starting,     // the channel answers but the client's page is not ready yet
    timeout,
    cancelled,
    protocol,  // unexpected reply
    script,    // the page threw, e.g. an internal name changed in an update
  };
  NeteaseCdpError(Kind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}
  [[nodiscard]] Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

// The websocket path of the client's main page in the DevTools /json target list.
[[nodiscard]] std::optional<std::string> netease_cdp_page_path(const nlohmann::json& targets);

// The value of a Runtime.evaluate reply; throws NeteaseCdpError(script) when
// the page threw.
[[nodiscard]] nlohmann::json netease_cdp_result(const nlohmann::json& reply);

// Evaluates an expression in the client's main page, awaiting a returned
// promise, and returns its value.
[[nodiscard]] nlohmann::json netease_cdp_evaluate(std::uint16_t port, std::string_view expression,
                                                  std::chrono::steady_clock::time_point deadline,
                                                  std::stop_token stop);

// Page scripts. Each returns {ok, ...} describing the player afterwards.
// Only checks that the client's store is reachable.
[[nodiscard]] std::string netease_ready_script();
// Plays today's daily recommendations, as the home page's daily card does.
[[nodiscard]] std::string netease_play_daily_script();
// Sets the play mode: "playOrder", "playCycle", "playOneCycle" or "playRandom".
[[nodiscard]] std::string netease_play_mode_script(std::string_view mode);

}  // namespace dvo
