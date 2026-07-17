#include "dvo/announcement.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace dvo {
namespace {

[[nodiscard]] std::optional<std::string> validate_request(
    const AnnouncementRequest& request) {
  if (request.announcement_id.empty()) return "announcement_id is required";
  if (request.text.empty()) return "announcement text is required";
  if (request.command_id.empty()) return "command_id is required";
  if (request.utterance_id.empty()) return "utterance_id is required";
  if (request.kind == AnnouncementKind::plan_started && request.action_ids.empty()) {
    return "plan announcements require at least one action_id";
  }
  return {};
}

[[nodiscard]] AnnouncementResult lifecycle_result(
    const AnnouncementRequest& request, AnnouncementStatus status,
    const char* backend) {
  AnnouncementResult result;
  result.schema_version = request.schema_version;
  result.announcement_id = request.announcement_id;
  result.kind = request.kind;
  result.status = status;
  result.text = request.text;
  result.source = request.source;
  result.origin = request.origin;
  result.activation_id = request.activation_id;
  result.turn_index = request.turn_index;
  result.trigger_sample = request.trigger_sample;
  result.utterance_id = request.utterance_id;
  result.command_id = request.command_id;
  result.action_ids = request.action_ids;
  result.timestamp_sample = request.timestamp_sample;
  result.backend = backend;
  return result;
}

[[nodiscard]] std::string action_text(const PlannedAction& action) {
  switch (action.type) {
    case ActionType::media_play:
      return "播放音乐";
    case ActionType::media_pause:
      return "暂停音乐";
    case ActionType::master_volume_adjust:
      if (!action.volume_delta_percent || *action.volume_delta_percent == 0) {
        return "调整音量";
      }
      if (*action.volume_delta_percent > 0) {
        return "增加音量 " + std::to_string(*action.volume_delta_percent) + "%";
      }
      const auto magnitude = -static_cast<std::int64_t>(*action.volume_delta_percent);
      return "降低音量 " + std::to_string(magnitude) + "%";
  }
  return "未知操作";
}

}  // namespace

const char* LogAnnouncementBackend::name() const noexcept { return "log"; }

void LogAnnouncementBackend::deliver(const AnnouncementRequest&,
                                     std::stop_token) {}

struct AnnouncementDispatcher::Impl {
  Impl(std::shared_ptr<IAnnouncementBackend> value,
       AnnouncementDispatcherConfig settings, LifecycleCallback callback)
      : backend(std::move(value)), config(settings), lifecycle(std::move(callback)) {
    if (!backend) throw std::invalid_argument("announcement backend is required");
    if (config.queue_capacity == 0) {
      throw std::invalid_argument("announcement queue capacity must be positive");
    }
    worker = std::jthread([this](std::stop_token stop) { run(stop); });
  }

  ~Impl() { stop(false); }

  AnnouncementSubmitResult submit(AnnouncementRequest request) {
    if (const auto validation = validate_request(request)) {
      return {AnnouncementSubmitStatus::invalid_request, *validation};
    }
    if (config.live_only && request.source != ExecutionSource::live) {
      return {AnnouncementSubmitStatus::suppressed,
              "non-live announcement suppressed by backend policy"};
    }
    {
      std::scoped_lock lock(mutex);
      if (!accepting) {
        return {AnnouncementSubmitStatus::stopped,
                "announcement dispatcher is stopped"};
      }
      if (queue.size() >= config.queue_capacity) {
        return {AnnouncementSubmitStatus::queue_full,
                "announcement dispatcher queue is full"};
      }
      queue.push_back(std::move(request));
    }
    wake.notify_one();
    return {AnnouncementSubmitStatus::accepted, {}};
  }

  void run(std::stop_token stop_token) {
    for (;;) {
      AnnouncementRequest request;
      {
        std::unique_lock lock(mutex);
        wake.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          if (stopping) break;
          continue;
        }
        if (stopping && !drain_on_stop) break;
        request = std::move(queue.front());
        queue.pop_front();
        in_flight = true;
      }

      deliver(request, stop_token);

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

  void deliver(const AnnouncementRequest& request, std::stop_token stop_token) noexcept {
    const auto started_at = std::chrono::steady_clock::now();
    auto started = lifecycle_result(request, AnnouncementStatus::playback_started,
                                    backend->name());
    try {
      started.render_endpoint = backend->render_endpoint();
    } catch (...) {
      // Endpoint diagnostics cannot turn delivery into an action-path failure.
    }
    publish(started);

    auto terminal = lifecycle_result(request, AnnouncementStatus::playback_finished,
                                     backend->name());
    terminal.render_endpoint = started.render_endpoint;
    try {
      backend->deliver(request, stop_token);
      if (stop_token.stop_requested()) {
        terminal.status = AnnouncementStatus::playback_cancelled;
        terminal.error_code = "dispatcher_stopped";
        terminal.message = "announcement delivery cancelled while stopping";
      }
    } catch (const std::exception& error) {
      terminal.status = AnnouncementStatus::playback_failed;
      terminal.error_code = "backend_exception";
      terminal.message = error.what();
    } catch (...) {
      terminal.status = AnnouncementStatus::playback_failed;
      terminal.error_code = "backend_exception";
      terminal.message = "announcement backend threw a non-standard exception";
    }
    terminal.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at);
    publish(terminal);
  }

  void publish(const AnnouncementResult& result) const noexcept {
    if (!lifecycle) return;
    try {
      lifecycle(result);
    } catch (...) {
      // Announcement observers are diagnostic-only and cannot stop delivery.
    }
  }

  bool wait_until_idle(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return idle.wait_for(lock, timeout,
                         [this] { return queue.empty() && !in_flight; });
  }

  void stop(bool drain) {
    std::unique_lock lifecycle_lock(stop_mutex);
    {
      std::scoped_lock lock(mutex);
      if (stopping) {
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
    if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) {
      worker.join();
    }
  }

  std::shared_ptr<IAnnouncementBackend> backend;
  AnnouncementDispatcherConfig config;
  LifecycleCallback lifecycle;
  std::mutex stop_mutex;
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::condition_variable idle;
  std::deque<AnnouncementRequest> queue;
  bool accepting{true};
  bool stopping{};
  bool drain_on_stop{true};
  bool in_flight{};
  std::jthread worker;
};

AnnouncementDispatcher::AnnouncementDispatcher(
    std::shared_ptr<IAnnouncementBackend> backend,
    AnnouncementDispatcherConfig config, LifecycleCallback lifecycle)
    : impl_(std::make_unique<Impl>(std::move(backend), config,
                                  std::move(lifecycle))) {}

AnnouncementDispatcher::~AnnouncementDispatcher() = default;

AnnouncementSubmitResult AnnouncementDispatcher::try_submit(
    AnnouncementRequest request) {
  return impl_->submit(std::move(request));
}

bool AnnouncementDispatcher::wait_until_idle(std::chrono::milliseconds timeout) {
  return impl_->wait_until_idle(timeout);
}

void AnnouncementDispatcher::stop(bool drain) { impl_->stop(drain); }

std::size_t AnnouncementDispatcher::queue_size() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->queue.size();
}

bool AnnouncementDispatcher::accepting() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->accepting;
}

std::string format_action_summary(const std::vector<PlannedAction>& actions) {
  if (actions.empty()) return {};
  std::string summary = "正在执行：";
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (i != 0) summary += "，然后";
    summary += action_text(actions[i]);
  }
  return summary;
}

AnnouncementRequest make_plan_announcement(const CommandPlan& plan,
                                           std::string announcement_id) {
  AnnouncementRequest request;
  request.announcement_id = std::move(announcement_id);
  request.text = format_action_summary(plan.actions);
  request.source = plan.source;
  request.origin = plan.origin;
  request.activation_id = plan.activation_id;
  request.turn_index = plan.turn_index;
  request.trigger_sample = plan.trigger_sample;
  request.utterance_id = plan.utterance_id;
  request.command_id = plan.command_id;
  request.timestamp_sample = plan.timestamp_sample;
  request.action_ids.reserve(plan.actions.size());
  for (const auto& action : plan.actions) {
    request.action_ids.push_back(action.action_id);
  }
  return request;
}

const char* to_string(AnnouncementKind kind) {
  switch (kind) {
    case AnnouncementKind::plan_started: return "plan_started";
  }
  return "unknown";
}

const char* to_string(AnnouncementStatus status) {
  switch (status) {
    case AnnouncementStatus::playback_started: return "playback_started";
    case AnnouncementStatus::playback_finished: return "playback_finished";
    case AnnouncementStatus::playback_failed: return "playback_failed";
    case AnnouncementStatus::playback_cancelled: return "playback_cancelled";
  }
  return "unknown";
}

const char* to_string(AnnouncementSubmitStatus status) {
  switch (status) {
    case AnnouncementSubmitStatus::accepted: return "accepted";
    case AnnouncementSubmitStatus::queue_full: return "queue_full";
    case AnnouncementSubmitStatus::invalid_request: return "invalid_request";
    case AnnouncementSubmitStatus::suppressed: return "suppressed";
    case AnnouncementSubmitStatus::stopped: return "stopped";
  }
  return "unknown";
}

}  // namespace dvo
