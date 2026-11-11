#include "dvo/action_executor.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

namespace dvo {
namespace {

[[nodiscard]] bool is_space_cp(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' ||
         cp == U'\f' || cp == U'\v' || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

[[nodiscard]] std::optional<std::string> normalize_search_field(
    std::string_view input) {
  std::size_t first_non_space = input.size();
  std::size_t last_non_space = 0;
  for (std::size_t offset = 0; offset < input.size();) {
    const auto lead = static_cast<unsigned char>(input[offset]);
    std::size_t length = 1;
    char32_t cp = lead;
    char32_t minimum = 0;
    if (lead <= 0x7F) {
      length = 1;
    } else if ((lead & 0xE0) == 0xC0) {
      length = 2;
      cp = lead & 0x1F;
      minimum = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
      length = 3;
      cp = lead & 0x0F;
      minimum = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
      length = 4;
      cp = lead & 0x07;
      minimum = 0x10000;
    } else {
      return std::nullopt;
    }
    if (offset + length > input.size()) return std::nullopt;
    for (std::size_t i = 1; i < length; ++i) {
      const auto next = static_cast<unsigned char>(input[offset + i]);
      if ((next & 0xC0) != 0x80) return std::nullopt;
      cp = (cp << 6) | (next & 0x3F);
    }
    if (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
      return std::nullopt;
    }
    if ((cp < 0x20 || cp == 0x7F) && !is_space_cp(cp)) return std::nullopt;
    if (!is_space_cp(cp)) {
      if (first_non_space == input.size()) first_non_space = offset;
      last_non_space = offset + length;
    }
    offset += length;
  }
  if (first_non_space >= last_non_space) return std::nullopt;
  const auto trimmed =
      input.substr(first_non_space, last_non_space - first_non_space);
  if (trimmed.empty() || trimmed.size() > kMaxMediaSearchTextBytes) {
    return std::nullopt;
  }
  return std::string(trimmed);
}

[[nodiscard]] std::optional<std::string> validate_and_normalize_plan(
    CommandPlan& plan) {
  if (plan.command_id.empty()) return "command_id is required";
  if (plan.utterance_id.empty()) return "utterance_id is required";
  if (plan.actions.empty()) return "a plan must contain at least one action";
  if (plan.actions.size() > kMaxCommandActions) return "a plan may contain at most 8 actions";

  std::unordered_set<std::string> ids;
  for (std::size_t i = 0; i < plan.actions.size(); ++i) {
    auto& action = plan.actions[i];
    if (action.sequence != i + 1) return "action sequences must be contiguous and one-based";
    if (action.action_id.empty()) return "action_id is required";
    if (!ids.insert(action.action_id).second) return "action_id values must be unique within a plan";
    if (std::string_view(to_string(action.type)) == "unknown") {
      return "action type is unknown";
    }
    if (action.type == ActionType::master_volume_adjust) {
      if (action.search_query || action.search_type || action.search_artist) {
        return "only media.search_play actions may carry search parameters";
      }
      if (!action.volume_delta_percent || *action.volume_delta_percent == 0 ||
          *action.volume_delta_percent < -20 || *action.volume_delta_percent > 20) {
        return "volume actions require a non-zero delta in [-20, 20]";
      }
    } else if (action.type == ActionType::media_search_play) {
      if (action.volume_delta_percent) {
        return "only volume actions may carry a volume delta";
      }
      if (!action.search_query) {
        return "media.search_play requires a non-empty search_query";
      }
      const auto normalized_query = normalize_search_field(*action.search_query);
      if (!normalized_query) {
        return "media.search_play search_query must be valid non-empty UTF-8 within 120 bytes";
      }
      action.search_query = *normalized_query;
      const auto effective_type =
          action.search_type.value_or(MediaSearchType::song);
      if (std::string_view(to_string(effective_type)) == "unknown") {
        return "media.search_play search_type is invalid";
      }
      action.search_type = effective_type;
      if (action.search_artist) {
        const auto normalized_artist =
            normalize_search_field(*action.search_artist);
        if (!normalized_artist) {
          return "media.search_play search_artist must be valid non-empty UTF-8 within 120 bytes";
        }
        action.search_artist = *normalized_artist;
      }
    } else {
      if (action.volume_delta_percent) {
        return "only volume actions may carry a volume delta";
      }
      if (action.search_query || action.search_type || action.search_artist) {
        return "only media.search_play actions may carry search parameters";
      }
    }
  }
  return {};
}

}  // namespace

struct OrderedActionExecutor::Impl {
  Impl(std::shared_ptr<IActionBackend> value, ActionExecutorConfig settings,
       ResultCallback callback, PlanStartedCallback started_callback)
      : backend(std::move(value)),
        config(settings),
        results(std::move(callback)),
        plan_started(std::move(started_callback)) {
    if (!backend) throw std::invalid_argument("action backend is required");
    if (config.queue_capacity == 0) throw std::invalid_argument("queue capacity must be positive");
    if (config.ledger_capacity == 0) throw std::invalid_argument("ledger capacity must be positive");
    worker = std::jthread([this](std::stop_token stop) { run(stop); });
  }

  ~Impl() { stop(false); }

  SubmitResult submit(CommandPlan plan, const AcceptedCallback& accepted) {
    if (const auto validation = validate_and_normalize_plan(plan)) {
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

      publish_plan_started(plan);
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
      result.origin = plan.origin;
      result.activation_id = plan.activation_id;
      result.turn_index = plan.turn_index;
      result.trigger_sample = plan.trigger_sample;
      result.action_id = action.action_id;
      result.sequence = action.sequence;
      result.type = action.type;
      result.source = plan.source;
      result.timestamp_sample = plan.timestamp_sample;
      result.requested_volume_delta_percent = action.volume_delta_percent;
      result.requested_search_query = action.search_query;
      result.requested_search_type = action.search_type;
      result.requested_search_artist = action.search_artist;

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
              result.origin = plan.origin;
              result.activation_id = plan.activation_id;
              result.turn_index = plan.turn_index;
              result.trigger_sample = plan.trigger_sample;
              result.action_id = action.action_id;
              result.sequence = action.sequence;
              result.type = action.type;
              result.source = plan.source;
              result.timestamp_sample = plan.timestamp_sample;
              if (!result.requested_volume_delta_percent) {
                result.requested_volume_delta_percent = action.volume_delta_percent;
              }
              if (!result.requested_search_query) {
                result.requested_search_query = action.search_query;
              }
              if (!result.requested_search_type) {
                result.requested_search_type = action.search_type;
              }
              if (!result.requested_search_artist) {
                result.requested_search_artist = action.search_artist;
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

  void publish_plan_started(const CommandPlan& plan) const noexcept {
    if (!plan_started) return;
    try {
      plan_started(plan);
    } catch (...) {
      // Plan observers are diagnostic-only and cannot break action ordering.
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
  PlanStartedCallback plan_started;
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
                                             ResultCallback results,
                                             PlanStartedCallback plan_started)
    : impl_(std::make_unique<Impl>(std::move(backend), config, std::move(results),
                                  std::move(plan_started))) {}

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
