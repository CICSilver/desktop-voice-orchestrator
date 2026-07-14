#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "dvo/action_executor.h"

namespace dvo {

struct WindowsActionConfig {
  // "default" resolves eRender/eConsole for each action. An explicit value is
  // treated as an MMDevice endpoint ID, matching the loopback capture setting.
  std::string render_device_id{"default"};
  // Deadline for each asynchronous GSMTC operation. Core Audio endpoint-volume
  // calls are synchronous and can only report their measured duration.
  std::uint32_t action_timeout_ms{2000};
};

// The returned backend is invoked on OrderedActionExecutor's dedicated worker
// thread. It initializes an MTA there before using GSMTC and Core Audio.
[[nodiscard]] std::shared_ptr<IActionBackend> create_windows_action_backend(
    WindowsActionConfig config = {});

}  // namespace dvo
