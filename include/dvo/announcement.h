#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

#include "dvo/command.h"

namespace dvo {

enum class AnnouncementKind {
  plan_started,
};

enum class AnnouncementStatus {
  playback_started,
  playback_finished,
  playback_failed,
  playback_cancelled,
};

enum class AnnouncementSubmitStatus {
  accepted,
  queue_full,
  invalid_request,
  suppressed,
  stopped,
};

struct AnnouncementRequest {
  std::uint32_t schema_version{1};
  std::string announcement_id;
  AnnouncementKind kind{AnnouncementKind::plan_started};
  std::string text;
  ExecutionSource source{ExecutionSource::live};
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
  std::string utterance_id;
  std::string command_id;
  std::vector<std::string> action_ids;
  std::uint64_t timestamp_sample{};
};

struct AnnouncementResult {
  std::uint32_t schema_version{1};
  std::string announcement_id;
  AnnouncementKind kind{AnnouncementKind::plan_started};
  AnnouncementStatus status{AnnouncementStatus::playback_failed};
  std::string text;
  ExecutionSource source{ExecutionSource::live};
  UtteranceOrigin origin{UtteranceOrigin::keyword};
  std::string activation_id;
  std::uint32_t turn_index{};
  std::uint64_t trigger_sample{};
  std::string utterance_id;
  std::string command_id;
  std::vector<std::string> action_ids;
  std::uint64_t timestamp_sample{};
  std::string backend;
  std::string render_endpoint;
  std::string error_code;
  std::string message;
  std::chrono::milliseconds duration{};
};

struct AnnouncementSubmitResult {
  AnnouncementSubmitStatus status{AnnouncementSubmitStatus::invalid_request};
  std::string error;

  [[nodiscard]] bool accepted() const {
    return status == AnnouncementSubmitStatus::accepted;
  }
};

class IAnnouncementBackend {
 public:
  virtual ~IAnnouncementBackend() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  // Audible backends report the endpoint that is also supplied to the
  // loopback/AEC reference path. The log backend intentionally returns empty.
  [[nodiscard]] virtual std::string render_endpoint() const { return {}; }
  virtual void deliver(const AnnouncementRequest& request,
                       std::stop_token stop) = 0;
};

// The log backend deliberately performs no I/O. Its dispatcher lifecycle
// callback is the structured log/event source used by the runtime.
class LogAnnouncementBackend final : public IAnnouncementBackend {
 public:
  [[nodiscard]] const char* name() const noexcept override;
  void deliver(const AnnouncementRequest& request,
               std::stop_token stop) override;
};

struct AnnouncementDispatcherConfig {
  std::size_t queue_capacity{32};
  // Future audible backends set this flag. The log backend leaves it false so
  // replay and benchmark runs retain deterministic text audit events.
  bool live_only{};
};

class AnnouncementDispatcher {
 public:
  using LifecycleCallback = std::function<void(const AnnouncementResult&)>;

  AnnouncementDispatcher(std::shared_ptr<IAnnouncementBackend> backend,
                         AnnouncementDispatcherConfig config = {},
                         LifecycleCallback lifecycle = {});
  ~AnnouncementDispatcher();

  AnnouncementDispatcher(const AnnouncementDispatcher&) = delete;
  AnnouncementDispatcher& operator=(const AnnouncementDispatcher&) = delete;

  // Never waits for backend delivery. A full queue is reported to the caller
  // so it can publish an announcement_dropped diagnostic without delaying
  // action execution.
  [[nodiscard]] AnnouncementSubmitResult try_submit(AnnouncementRequest request);
  [[nodiscard]] bool wait_until_idle(std::chrono::milliseconds timeout);
  void stop(bool drain = true);

  [[nodiscard]] std::size_t queue_size() const;
  [[nodiscard]] bool accepting() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string format_action_summary(
    const std::vector<PlannedAction>& actions);
[[nodiscard]] AnnouncementRequest make_plan_announcement(
    const CommandPlan& plan, std::string announcement_id);
[[nodiscard]] const char* to_string(AnnouncementKind kind);
[[nodiscard]] const char* to_string(AnnouncementStatus status);
[[nodiscard]] const char* to_string(AnnouncementSubmitStatus status);

}  // namespace dvo
