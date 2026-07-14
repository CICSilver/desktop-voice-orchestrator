#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <stop_token>

#include "dvo/command.h"

namespace dvo {

class IActionBackend {
 public:
  virtual ~IActionBackend() = default;
  [[nodiscard]] virtual ActionResult execute(const PlannedAction& action,
                                             std::stop_token stop) = 0;
};

enum class SubmitStatus {
  accepted,
  queue_full,
  invalid_plan,
  stopped,
};

struct SubmitResult {
  SubmitStatus status{SubmitStatus::invalid_plan};
  std::string error;

  [[nodiscard]] bool accepted() const { return status == SubmitStatus::accepted; }
};

struct ActionExecutorConfig {
  std::size_t queue_capacity{32};
  std::size_t ledger_capacity{4096};
  // dry_run is a global safety ceiling. A replay/benchmark plan is dry-run
  // regardless of this setting.
  ExecutionMode mode{ExecutionMode::live};
};

class OrderedActionExecutor {
 public:
  using ResultCallback = std::function<void(const ActionResult&)>;
  // Runs synchronously after the plan is stored in the bounded queue but
  // before the worker can dequeue it. It is intended for authoritative
  // command_plan/action_queued lifecycle events.
  using AcceptedCallback = std::function<void(const CommandPlan&)>;

  OrderedActionExecutor(std::shared_ptr<IActionBackend> backend,
                        ActionExecutorConfig config = {},
                        ResultCallback results = {});
  ~OrderedActionExecutor();

  OrderedActionExecutor(const OrderedActionExecutor&) = delete;
  OrderedActionExecutor& operator=(const OrderedActionExecutor&) = delete;

  [[nodiscard]] SubmitResult try_submit(CommandPlan plan,
                                        AcceptedCallback accepted = {});
  [[nodiscard]] bool wait_until_idle(std::chrono::milliseconds timeout);
  void stop(bool drain = true);

  [[nodiscard]] std::size_t queue_size() const;
  [[nodiscard]] bool accepting() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* to_string(SubmitStatus status);

}  // namespace dvo
