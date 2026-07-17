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
#include "dvo/command_parser.h"

namespace {

class FakeBackend final : public dvo::IActionBackend {
 public:
  dvo::ActionResult execute(const dvo::PlannedAction& action,
                            std::stop_token) override {
    {
      std::unique_lock lock(mutex);
      calls.push_back(action.action_id);
      threads.push_back(std::this_thread::get_id());
      entered = true;
      entered_cv.notify_all();
      if (block) release_cv.wait(lock, [this] { return released; });
    }
    dvo::ActionResult result;
    result.type = action.type;
    result.adapter = "fake";
    result.status = fail_first.exchange(false) ? dvo::ActionStatus::failed
                                               : dvo::ActionStatus::succeeded;
    if (result.status == dvo::ActionStatus::failed) result.error_code = "injected_failure";
    return result;
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

  std::size_t call_count() const {
    std::scoped_lock lock(mutex);
    return calls.size();
  }

  std::vector<std::string> call_ids() const {
    std::scoped_lock lock(mutex);
    return calls;
  }

  std::vector<std::thread::id> thread_ids() const {
    std::scoped_lock lock(mutex);
    return threads;
  }

  mutable std::mutex mutex;
  std::condition_variable entered_cv;
  std::condition_variable release_cv;
  std::vector<std::string> calls;
  std::vector<std::thread::id> threads;
  bool block{};
  bool entered{};
  bool released{};
  std::atomic<bool> fail_first{};
};

dvo::CommandPlan plan(std::string id, std::string text = "播放音乐") {
  dvo::CommandParseContext context;
  context.runtime_session_id = "executor-test-runtime";
  context.utterance_id = std::move(id);
  context.config_revision = 4;
  context.recognition_generation = 2;
  context.final_revision = 3;
  context.timestamp_sample = 1234;
  const auto parsed = dvo::CommandParser{}.parse(text, context);
  REQUIRE(parsed.ok());
  return *parsed.plan;
}

}  // namespace

TEST_CASE("Ordered executor invokes all actions serially in plan order") {
  auto backend = std::make_shared<FakeBackend>();
  std::mutex results_mutex;
  std::vector<dvo::ActionResult> results;
  std::atomic<bool> accepted_published{};
  std::atomic<bool> started_before_accepted{};
  dvo::OrderedActionExecutor executor(
      backend, {}, [&](const dvo::ActionResult& result) {
        if (result.status == dvo::ActionStatus::started &&
            !accepted_published.load(std::memory_order_acquire)) {
          started_before_accepted.store(true, std::memory_order_release);
        }
        std::scoped_lock lock(results_mutex);
        results.push_back(result);
      });

  auto command = plan("ordered", "播放音乐暂停音乐增加音量降低音量");
  command.origin = dvo::UtteranceOrigin::followup;
  command.activation_id = "activation-ordered";
  command.turn_index = 3;
  command.trigger_sample = 1200;
  REQUIRE(executor.try_submit(command, [&](const dvo::CommandPlan& accepted) {
            CHECK(accepted.command_id == command.command_id);
            accepted_published.store(true, std::memory_order_release);
          }).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK_FALSE(started_before_accepted.load(std::memory_order_acquire));
  const auto calls = backend->call_ids();
  REQUIRE(calls.size() == 4);
  for (std::size_t i = 0; i < calls.size(); ++i) CHECK(calls[i] == command.actions[i].action_id);
  const auto threads = backend->thread_ids();
  REQUIRE(threads.size() == 4);
  for (const auto id : threads) CHECK(id == threads.front());
  {
    std::scoped_lock lock(results_mutex);
    REQUIRE(results.size() == 8);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(results[i * 2].status == dvo::ActionStatus::started);
      CHECK(results[i * 2 + 1].status == dvo::ActionStatus::succeeded);
      CHECK(results[i * 2].action_id == command.actions[i].action_id);
      CHECK(results[i * 2].origin == command.origin);
      CHECK(results[i * 2].activation_id == command.activation_id);
      CHECK(results[i * 2].turn_index == command.turn_index);
      CHECK(results[i * 2].trigger_sample == command.trigger_sample);
      CHECK(results[i * 2].source == command.source);
      CHECK(results[i * 2].timestamp_sample == command.timestamp_sample);
    }
  }
  executor.stop();
}

TEST_CASE("Plan-started observer runs once after dequeue and before actions") {
  auto backend = std::make_shared<FakeBackend>();
  std::mutex mutex;
  std::vector<std::string> lifecycle;
  dvo::OrderedActionExecutor executor(
      backend, {},
      [&](const dvo::ActionResult& result) {
        if (result.status != dvo::ActionStatus::started) return;
        std::scoped_lock lock(mutex);
        lifecycle.push_back("action:" + result.action_id);
      },
      [&](const dvo::CommandPlan& started) {
        std::scoped_lock lock(mutex);
        lifecycle.push_back("plan:" + started.command_id);
      });

  const auto command = plan("plan-started", "播放音乐暂停音乐");
  REQUIRE(executor.try_submit(command).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  {
    std::scoped_lock lock(mutex);
    REQUIRE(lifecycle.size() == 3);
    CHECK(lifecycle[0] == "plan:" + command.command_id);
    CHECK(lifecycle[1] == "action:" + command.actions[0].action_id);
    CHECK(lifecycle[2] == "action:" + command.actions[1].action_id);
  }
}

TEST_CASE("Plan-started observer exceptions do not cancel action execution") {
  auto backend = std::make_shared<FakeBackend>();
  std::atomic<std::size_t> observer_calls{};
  dvo::OrderedActionExecutor executor(
      backend, {}, {}, [&](const dvo::CommandPlan&) {
        ++observer_calls;
        throw std::runtime_error("injected observer failure");
      });

  REQUIRE(executor.try_submit(plan("throwing-plan-observer")).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(observer_calls == 1);
  CHECK(backend->call_count() == 1);
}

TEST_CASE("Executor ledger makes each action ID an at-most-once terminal attempt") {
  auto backend = std::make_shared<FakeBackend>();
  std::mutex mutex;
  std::vector<dvo::ActionStatus> statuses;
  dvo::OrderedActionExecutor executor(
      backend, {}, [&](const dvo::ActionResult& result) {
        std::scoped_lock lock(mutex);
        statuses.push_back(result.status);
      });
  const auto command = plan("duplicate");
  REQUIRE(executor.try_submit(command).accepted());
  REQUIRE(executor.try_submit(command).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->call_count() == 1);
  {
    std::scoped_lock lock(mutex);
    REQUIRE(statuses.size() == 3);
    CHECK(statuses[0] == dvo::ActionStatus::started);
    CHECK(statuses[1] == dvo::ActionStatus::succeeded);
    CHECK(statuses[2] == dvo::ActionStatus::deduplicated);
  }
}

TEST_CASE("Replay and configured dry-run plans never call the external backend") {
  auto backend = std::make_shared<FakeBackend>();
  std::mutex mutex;
  std::vector<dvo::ActionStatus> statuses;
  dvo::OrderedActionExecutor executor(
      backend, {}, [&](const dvo::ActionResult& result) {
        std::scoped_lock lock(mutex);
        statuses.push_back(result.status);
      });

  auto replay = plan("replay");
  replay.source = dvo::ExecutionSource::replay;
  replay.execution_mode = dvo::ExecutionMode::live;
  auto dry = plan("dry");
  dry.execution_mode = dvo::ExecutionMode::dry_run;
  REQUIRE(executor.try_submit(replay).accepted());
  REQUIRE(executor.try_submit(dry).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->call_count() == 0);
  {
    std::scoped_lock lock(mutex);
    REQUIRE(statuses.size() == 4);
    CHECK(statuses[0] == dvo::ActionStatus::started);
    CHECK(statuses[1] == dvo::ActionStatus::dry_run);
    CHECK(statuses[2] == dvo::ActionStatus::started);
    CHECK(statuses[3] == dvo::ActionStatus::dry_run);
  }

  auto globally_dry_backend = std::make_shared<FakeBackend>();
  dvo::ActionExecutorConfig globally_dry_config;
  globally_dry_config.mode = dvo::ExecutionMode::dry_run;
  dvo::OrderedActionExecutor globally_dry(globally_dry_backend, globally_dry_config);
  REQUIRE(globally_dry.try_submit(plan("globally-dry")).accepted());
  REQUIRE(globally_dry.wait_until_idle(std::chrono::seconds(2)));
  CHECK(globally_dry_backend->call_count() == 0);
}

TEST_CASE("Executor queue is bounded without blocking submitters") {
  auto backend = std::make_shared<FakeBackend>();
  backend->block = true;
  dvo::ActionExecutorConfig config;
  config.queue_capacity = 1;
  dvo::OrderedActionExecutor executor(backend, config);

  REQUIRE(executor.try_submit(plan("one")).accepted());
  REQUIRE(backend->wait_entered(std::chrono::seconds(2)));
  REQUIRE(executor.try_submit(plan("two")).accepted());
  bool overflow_callback{};
  const auto overflow = executor.try_submit(
      plan("three"), [&](const dvo::CommandPlan&) { overflow_callback = true; });
  CHECK(overflow.status == dvo::SubmitStatus::queue_full);
  CHECK_FALSE(overflow_callback);
  backend->release();
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->call_count() == 2);
}

TEST_CASE("A failed action does not disturb the remaining serial order") {
  auto backend = std::make_shared<FakeBackend>();
  backend->fail_first = true;
  std::mutex mutex;
  std::vector<dvo::ActionStatus> statuses;
  dvo::OrderedActionExecutor executor(
      backend, {}, [&](const dvo::ActionResult& result) {
        std::scoped_lock lock(mutex);
        statuses.push_back(result.status);
      });
  REQUIRE(executor.try_submit(plan("failure", "播放音乐暂停音乐")).accepted());
  REQUIRE(executor.wait_until_idle(std::chrono::seconds(2)));
  CHECK(backend->call_count() == 2);
  {
    std::scoped_lock lock(mutex);
    REQUIRE(statuses.size() == 4);
    CHECK(statuses[0] == dvo::ActionStatus::started);
    CHECK(statuses[1] == dvo::ActionStatus::failed);
    CHECK(statuses[2] == dvo::ActionStatus::started);
    CHECK(statuses[3] == dvo::ActionStatus::succeeded);
  }
}

TEST_CASE("Executor validates plans before enqueueing") {
  auto backend = std::make_shared<FakeBackend>();
  dvo::OrderedActionExecutor executor(backend);
  auto invalid = plan("invalid");
  invalid.actions[0].sequence = 2;
  CHECK(executor.try_submit(std::move(invalid)).status == dvo::SubmitStatus::invalid_plan);
  CHECK(backend->call_count() == 0);
  executor.stop();
  CHECK(executor.try_submit(plan("after-stop")).status == dvo::SubmitStatus::stopped);
}
