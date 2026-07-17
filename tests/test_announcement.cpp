#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dvo/action_executor.h"
#include "dvo/announcement.h"
#include "dvo/command_parser.h"

namespace {

dvo::AnnouncementRequest request(std::string id,
                                 dvo::ExecutionSource source = dvo::ExecutionSource::live) {
  dvo::AnnouncementRequest value;
  value.announcement_id = std::move(id);
  value.text = "正在执行：暂停音乐";
  value.source = source;
  value.origin = dvo::UtteranceOrigin::followup;
  value.activation_id = "activation-7";
  value.turn_index = 2;
  value.trigger_sample = 15'500;
  value.utterance_id = "utterance-8";
  value.command_id = "command-9";
  value.action_ids = {"action-10"};
  value.timestamp_sample = 16'000;
  return value;
}

class BlockingAnnouncementBackend final : public dvo::IAnnouncementBackend {
 public:
  [[nodiscard]] const char* name() const noexcept override { return "blocking"; }

  void deliver(const dvo::AnnouncementRequest& value,
               std::stop_token stop) override {
    std::unique_lock lock(mutex);
    delivered.push_back(value.announcement_id);
    entered = true;
    entered_cv.notify_all();
    release_cv.wait(lock, stop, [this] { return released; });
  }

  bool wait_entered(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return entered_cv.wait_for(lock, timeout, [this] { return entered; });
  }

  void release() {
    {
      std::scoped_lock lock(mutex);
      released = true;
    }
    release_cv.notify_all();
  }

  std::size_t count() const {
    std::scoped_lock lock(mutex);
    return delivered.size();
  }

  mutable std::mutex mutex;
  std::condition_variable_any entered_cv;
  std::condition_variable_any release_cv;
  std::vector<std::string> delivered;
  bool entered{};
  bool released{};
};

class ThrowOnceAnnouncementBackend final : public dvo::IAnnouncementBackend {
 public:
  [[nodiscard]] const char* name() const noexcept override { return "throw-once"; }

  void deliver(const dvo::AnnouncementRequest&, std::stop_token) override {
    ++calls;
    if (throw_next.exchange(false)) throw std::runtime_error("injected failure");
  }

  std::atomic<std::size_t> calls{};
  std::atomic<bool> throw_next{true};
};

class CountingActionBackend final : public dvo::IActionBackend {
 public:
  dvo::ActionResult execute(const dvo::PlannedAction& action,
                            std::stop_token) override {
    ++calls;
    dvo::ActionResult result;
    result.type = action.type;
    result.status = dvo::ActionStatus::succeeded;
    return result;
  }

  std::atomic<std::size_t> calls{};
};

dvo::CommandPlan command_plan(std::string utterance_id) {
  dvo::CommandParseContext context;
  context.runtime_session_id = "announcement-test-runtime";
  context.utterance_id = std::move(utterance_id);
  context.timestamp_sample = 32000;
  const auto parsed = dvo::CommandParser{}.parse("暂停音乐", context);
  REQUIRE(parsed.ok());
  return *parsed.plan;
}

}  // namespace

TEST_CASE("Action summaries are deterministic and use structured actions") {
  std::vector<dvo::PlannedAction> actions{
      {"play", 1, dvo::ActionType::media_play, {}},
      {"pause", 2, dvo::ActionType::media_pause, {}},
      {"up", 3, dvo::ActionType::master_volume_adjust, 5},
      {"down", 4, dvo::ActionType::master_volume_adjust, -7},
  };
  CHECK(dvo::format_action_summary(actions) ==
        "正在执行：播放音乐，然后暂停音乐，然后增加音量 5%，然后降低音量 7%");
  CHECK(dvo::format_action_summary({}).empty());

  auto plan = command_plan("summary-correlation");
  plan.activation_id = "activation-summary";
  plan.turn_index = 4;
  plan.origin = dvo::UtteranceOrigin::followup;
  plan.trigger_sample = 31'000;
  const auto announcement = dvo::make_plan_announcement(plan, "announcement-summary");
  CHECK(announcement.announcement_id == "announcement-summary");
  CHECK(announcement.activation_id == plan.activation_id);
  CHECK(announcement.turn_index == plan.turn_index);
  CHECK(announcement.origin == plan.origin);
  CHECK(announcement.trigger_sample == plan.trigger_sample);
  CHECK(announcement.command_id == plan.command_id);
  REQUIRE(announcement.action_ids.size() == 1);
  CHECK(announcement.action_ids.front() == plan.actions.front().action_id);
}

TEST_CASE("Log announcements publish correlated started and finished lifecycle") {
  auto backend = std::make_shared<dvo::LogAnnouncementBackend>();
  std::mutex mutex;
  std::vector<dvo::AnnouncementResult> results;
  dvo::AnnouncementDispatcher dispatcher(
      backend, {}, [&](const dvo::AnnouncementResult& result) {
        std::scoped_lock lock(mutex);
        results.push_back(result);
      });

  const auto value = request("announcement-lifecycle");
  REQUIRE(dispatcher.try_submit(value).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  {
    std::scoped_lock lock(mutex);
    REQUIRE(results.size() == 2);
    CHECK(results[0].status == dvo::AnnouncementStatus::playback_started);
    CHECK(results[1].status == dvo::AnnouncementStatus::playback_finished);
    for (const auto& result : results) {
      CHECK(result.announcement_id == value.announcement_id);
      CHECK(result.activation_id == value.activation_id);
      CHECK(result.turn_index == value.turn_index);
      CHECK(result.origin == value.origin);
      CHECK(result.trigger_sample == value.trigger_sample);
      CHECK(result.utterance_id == value.utterance_id);
      CHECK(result.command_id == value.command_id);
      CHECK(result.action_ids == value.action_ids);
      CHECK(result.timestamp_sample == value.timestamp_sample);
      CHECK(result.backend == "log");
    }
  }
}

TEST_CASE("Announcement queue is bounded and submission never waits for playback") {
  auto backend = std::make_shared<BlockingAnnouncementBackend>();
  dvo::AnnouncementDispatcherConfig config;
  config.queue_capacity = 1;
  dvo::AnnouncementDispatcher dispatcher(backend, config);

  REQUIRE(dispatcher.try_submit(request("one")).accepted());
  REQUIRE(backend->wait_entered(std::chrono::seconds(2)));
  REQUIRE(dispatcher.try_submit(request("two")).accepted());
  CHECK(dispatcher.try_submit(request("three")).status ==
        dvo::AnnouncementSubmitStatus::queue_full);
  backend->release();
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->count() == 2);
}

TEST_CASE("Backend and lifecycle observer failures are isolated") {
  auto backend = std::make_shared<ThrowOnceAnnouncementBackend>();
  std::mutex mutex;
  std::vector<dvo::AnnouncementStatus> terminal;
  std::atomic<bool> throw_observer{true};
  dvo::AnnouncementDispatcher dispatcher(
      backend, {}, [&](const dvo::AnnouncementResult& result) {
        if (throw_observer.exchange(false)) {
          throw std::runtime_error("injected lifecycle observer failure");
        }
        if (result.status != dvo::AnnouncementStatus::playback_started) {
          std::scoped_lock lock(mutex);
          terminal.push_back(result.status);
        }
      });

  REQUIRE(dispatcher.try_submit(request("fails")).accepted());
  REQUIRE(dispatcher.try_submit(request("continues")).accepted());
  REQUIRE(dispatcher.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->calls == 2);
  {
    std::scoped_lock lock(mutex);
    REQUIRE(terminal.size() == 2);
    CHECK(terminal[0] == dvo::AnnouncementStatus::playback_failed);
    CHECK(terminal[1] == dvo::AnnouncementStatus::playback_finished);
  }
}

TEST_CASE("Audible backend policy suppresses non-live requests while log audit retains them") {
  auto live_only_backend = std::make_shared<ThrowOnceAnnouncementBackend>();
  live_only_backend->throw_next = false;
  dvo::AnnouncementDispatcherConfig live_only;
  live_only.live_only = true;
  dvo::AnnouncementDispatcher audible(live_only_backend, live_only);

  CHECK(audible.try_submit(request("replay", dvo::ExecutionSource::replay)).status ==
        dvo::AnnouncementSubmitStatus::suppressed);
  CHECK(audible.try_submit(request("benchmark", dvo::ExecutionSource::benchmark)).status ==
        dvo::AnnouncementSubmitStatus::suppressed);
  CHECK(live_only_backend->calls == 0);

  auto log_backend = std::make_shared<dvo::LogAnnouncementBackend>();
  dvo::AnnouncementDispatcher audit(log_backend);
  REQUIRE(audit.try_submit(request("replay-log", dvo::ExecutionSource::replay)).accepted());
  REQUIRE(audit.wait_until_idle(std::chrono::seconds(2)));
}

TEST_CASE("Slow announcement playback cannot block or reorder action execution") {
  auto announcement_backend = std::make_shared<BlockingAnnouncementBackend>();
  dvo::AnnouncementDispatcherConfig announcement_config;
  announcement_config.queue_capacity = 1;
  dvo::AnnouncementDispatcher announcements(announcement_backend, announcement_config);
  auto action_backend = std::make_shared<CountingActionBackend>();
  std::atomic<std::size_t> dropped{};
  std::atomic<std::size_t> sequence{};
  dvo::OrderedActionExecutor executor(
      action_backend, {}, {}, [&](const dvo::CommandPlan& plan) {
        auto value = request("plan-" + std::to_string(++sequence));
        value.command_id = plan.command_id;
        value.utterance_id = plan.utterance_id;
        value.action_ids = {plan.actions.front().action_id};
        if (!announcements.try_submit(std::move(value)).accepted()) ++dropped;
      });

  REQUIRE(executor.try_submit(command_plan("one")).accepted());
  REQUIRE(announcement_backend->wait_entered(std::chrono::seconds(2)));
  REQUIRE(executor.try_submit(command_plan("two")).accepted());
  REQUIRE(executor.try_submit(command_plan("three")).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(action_backend->calls == 3);
  CHECK(dropped >= 1);

  announcement_backend->release();
  REQUIRE(announcements.wait_until_idle(std::chrono::seconds(2)));
}
