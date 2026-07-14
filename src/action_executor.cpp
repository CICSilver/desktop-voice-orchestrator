#include "dvo/action_executor.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

namespace dvo {
namespace {

[[nodiscard]] std::optional<std::string> validate_plan(const CommandPlan& plan) {
  if (plan.command_id.empty()) return "command_id is required";
  if (plan.utterance_id.empty()) return "utterance_id is required";
  if (plan.actions.empty()) return "a plan must contain at least one action";
  if (plan.actions.size() > kMaxCommandActions) return "a plan may contain at most 8 actions";

  std::unordered_set<std::string> ids;
  for (std::size_t i = 0; i < plan.actions.size(); ++i) {
    const auto& action = plan.actions[i];
    if (action.sequence != i + 1) return "action sequences must be contiguous and one-based";
    if (action.action_id.empty()) return "action_id is required";
    if (!ids.insert(action.action_id).second) return "action_id values must be unique within a plan";
    if (action.type == ActionType::master_volume_adjust) {
      if (!action.volume_delta_percent || *action.volume_delta_percent == 0 ||
          *action.volume_delta_percent < -20 || *action.volume_delta_percent > 20) {
        return "volume actions require a non-zero delta in [-20, 20]";
      }
    } else if (action.volume_delta_percent) {
      return "only volume actions may carry a volume delta";
    }
  }
  return {};
}

}  // namespace

struct OrderedActionExecutor::Impl {
  Impl(std::shared_ptr<IActionBackend> value, ActionExecutorConfig settings,
       ResultCallback callback)
      : backend(std::move(value)), config(settings), results(std::move(callback)) {
    if (!backend) throw std::invalid_argument("action backend is required");
    if (config.queue_capacity == 0) throw std::invalid_argument("queue capacity must be positive");
    if (config.ledger_capacity == 0) throw std::invalid_argument("ledger capacity must be positive");
    worker = std::jthread([this](std::stop_token stop) { run(stop); });
  }

  ~Impl() { stop(false); }

  SubmitResult submit(CommandPlan plan, const AcceptedCallback& accepted) {
    if (const auto validation = validate_plan(plan)) {
      return {SubmitStatus::invalid_plan, *validation};
    }
    {
      std::scoped_lock lock(mutex);
      if (!accepting) return {SubmitStatus::stopped, "executor is stopped"};
      if (queue.size() >= config.queue_capacity) {
        return {SubmitStatus::queue_full, "executor queue is full"};
      }
      queue.push_back(std::move(plan));
      if (accepted) {
        try {
          accepted(queue.back());
        } catch (...) {
          // Lifecycle observers are diagnostic-only. The accepted plan remains
          // queued even if an observer fails.
        }
      }
    }
    wake.notify_one();
    return {SubmitStatus::accepted, {}};
  }

  void run(std::stop_token stop_token) {
    for (;;) {
      CommandPlan plan;
      {
        std::unique_lock lock(mutex);
        wake.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          if (stopping) break;
          continue;
        }
        if (stopping && !drain_on_stop) break;
        plan = std::move(queue.front());
        queue.pop_front();
        in_flight = true;
      }

      execute_plan(plan, stop_token);

      {
        std::scoped_lock lock(mutex);
        in_flight = false;
        if (queue.empty()) idle.notify_all();
        if (stopping && (!drain_on_stop || queue.empty())) break;
      }
    }
    {
      std::scoped_lock lock(mutex);
      in_flight = false;
      idle.notify_all();
    }
  }

  void execute_plan(const CommandPlan& plan, std::stop_token stop_token) {
    const bool dry_run = config.mode == ExecutionMode::dry_run ||
                         plan.execution_mode == ExecutionMode::dry_run ||
                         plan.source != ExecutionSource::live;

    for (const auto& action : plan.actions) {
      const auto start_time = std::chrono::steady_clock::now();
      ActionResult result;
      result.command_id = plan.command_id;
      result.utterance_id = plan.utterance_id;
      result.action_id = action.action_id;
      result.sequence = action.sequence;
      result.type = action.type;
      result.source = plan.source;
      result.timestamp_sample = plan.timestamp_sample;
      result.requested_volume_delta_percent = action.volume_delta_percent;

      if (ledger.contains(action.action_id)) {
        result.status = ActionStatus::deduplicated;
        result.adapter = "ordered_executor";
        result.error_code = "duplicate_action";
        result.message = "action_id has already reached a terminal attempt";
      } else {
        remember(action.action_id);
        if (stop_token.stop_requested()) {
          result.status = ActionStatus::cancelled;
          result.adapter = "ordered_executor";
          result.error_code = "executor_stopped";
          result.message = "executor stopped before the action started";
        } else {
          auto lifecycle = result;
          lifecycle.status = ActionStatus::started;
          lifecycle.adapter = "ordered_executor";
          lifecycle.message = "action dequeued and terminal execution attempt started";
          publish(lifecycle);

          if (dry_run) {
            result.status = ActionStatus::dry_run;
            result.adapter = "ordered_executor";
            result.message = "external action suppressed by dry-run policy";
          } else {
            try {
              auto backend_result = backend->execute(action, stop_token);
              result = std::move(backend_result);
              result.command_id = plan.command_id;
              result.utterance_id = plan.utterance_id;
              result.action_id = action.action_id;
              result.sequence = action.sequence;
              result.type = action.type;
              result.source = plan.source;
              result.timestamp_sample = plan.timestamp_sample;
              if (!result.requested_volume_delta_percent) {
                result.requested_volume_delta_percent = action.volume_delta_percent;
              }
            } catch (const std::exception& error) {
              result.status = ActionStatus::failed;
              result.adapter = "ordered_executor";
              result.error_code = "backend_exception";
              result.message = error.what();
            } catch (...) {
              result.status = ActionStatus::failed;
              result.adapter = "ordered_executor";
              result.error_code = "backend_exception";
              result.message = "action backend threw a non-standard exception";
            }
          }
        }
      }

      result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_time);
      publish(result);
    }
  }

  void publish(const ActionResult& result) const noexcept {
    if (!results) return;
    try {
      results(result);
    } catch (...) {
      // Result observers are diagnostic-only and cannot break ordering.
    }
  }

  void remember(const std::string& action_id) {
    if (ledger.size() >= config.ledger_capacity) {
      ledger.erase(ledger_order.front());
      ledger_order.pop_front();
    }
    ledger.insert(action_id);
    ledger_order.push_back(action_id);
  }

  bool wait_until_idle(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return idle.wait_for(lock, timeout, [this] { return queue.empty() && !in_flight; });
  }

  void stop(bool drain) {
    std::unique_lock lifecycle_lock(stop_mutex);
    {
      std::scoped_lock lock(mutex);
      if (stopping) {
        // A prior non-draining stop cannot be upgraded to a draining stop.
        drain = drain_on_stop;
      } else {
        accepting = false;
        stopping = true;
        drain_on_stop = drain;
        if (!drain_on_stop) queue.clear();
      }
    }
    if (!drain && worker.joinable()) worker.request_stop();
    wake.notify_all();
    if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) worker.join();
  }

  std::shared_ptr<IActionBackend> backend;
  ActionExecutorConfig config;
  ResultCallback results;
  std::mutex stop_mutex;
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::condition_variable idle;
  std::deque<CommandPlan> queue;
  bool accepting{true};
  bool stopping{};
  bool drain_on_stop{true};
  bool in_flight{};
  std::unordered_set<std::string> ledger;
  std::deque<std::string> ledger_order;
  std::jthread worker;
};

OrderedActionExecutor::OrderedActionExecutor(std::shared_ptr<IActionBackend> backend,
                                             ActionExecutorConfig config,
                                             ResultCallback results)
    : impl_(std::make_unique<Impl>(std::move(backend), config, std::move(results))) {}

OrderedActionExecutor::~OrderedActionExecutor() = default;

SubmitResult OrderedActionExecutor::try_submit(CommandPlan plan,
                                                AcceptedCallback accepted) {
  return impl_->submit(std::move(plan), accepted);
}

bool OrderedActionExecutor::wait_until_idle(std::chrono::milliseconds timeout) {
  return impl_->wait_until_idle(timeout);
}

void OrderedActionExecutor::stop(bool drain) { impl_->stop(drain); }

std::size_t OrderedActionExecutor::queue_size() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->queue.size();
}

bool OrderedActionExecutor::accepting() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->accepting;
}

const char* to_string(SubmitStatus status) {
  switch (status) {
    case SubmitStatus::accepted: return "accepted";
    case SubmitStatus::queue_full: return "queue_full";
    case SubmitStatus::invalid_plan: return "invalid_plan";
    case SubmitStatus::stopped: return "stopped";
  }
  return "unknown";
}

}  // namespace dvo
