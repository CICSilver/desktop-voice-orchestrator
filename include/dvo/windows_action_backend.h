#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "dvo/action_executor.h"

namespace dvo {

// See NeteaseConfig; empty paths are resolved when the action runs.
struct NeteaseActionConfig {
  std::filesystem::path ncm_cli;
  std::filesystem::path node;
  std::filesystem::path client_data;
  // Covers the whole action: reading the client's history plus up to five
  // ncm-cli calls of about 1.3 s each (search, verify after the hotkey, like).
  std::uint32_t timeout_ms{10000};
  std::uint16_t cdp_port{9223};
  std::filesystem::path executable;
};

// Starts cloudmusic.exe with its local DevTools channel (minimized, without
// taking the focus) and waits until the channel answers. With `restart`, a
// client running without the channel is closed first; otherwise such a
// client is left alone and false is returned.
[[nodiscard]] bool start_netease_with_control(const NeteaseActionConfig& config, bool restart);

struct WindowsActionConfig {
  // "default" resolves eRender/eConsole for each action. An explicit value is
  // treated as an MMDevice endpoint ID, matching the loopback capture setting.
  std::string render_device_id{"default"};
  // Deadline for each asynchronous GSMTC operation. Core Audio endpoint-volume
  // calls are synchronous and can only report their measured duration.
  std::uint32_t action_timeout_ms{2000};
  NeteaseActionConfig netease;
};

// The returned backend is invoked on OrderedActionExecutor's dedicated worker
// thread. It initializes an MTA there before using GSMTC and Core Audio.
[[nodiscard]] std::shared_ptr<IActionBackend> create_windows_action_backend(
    WindowsActionConfig config = {});

}  // namespace dvo
