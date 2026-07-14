#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "dvo/config.h"
#include "dvo/spsc_queue.h"

namespace dvo {

class DebugServer {
 public:
  using CommandHandler = std::function<nlohmann::json(const nlohmann::json&)>;

  explicit DebugServer(std::size_t telemetry_queue_capacity = 4096);
  ~DebugServer();
  DebugServer(const DebugServer&) = delete;
  DebugServer& operator=(const DebugServer&) = delete;

  void start(const WebConfig& config, CommandHandler commands);
  void stop();
  bool publish(std::string type, nlohmann::json payload, std::uint64_t timestamp_sample = 0,
               std::string source = "live");
  [[nodiscard]] std::string token() const;
  [[nodiscard]] std::string url() const;
  [[nodiscard]] std::uint64_t dropped() const { return dropped_.load(std::memory_order_acquire); }

 private:
  struct Outbound {
    std::string type;
    nlohmann::json payload;
    std::uint64_t timestamp_sample{};
    std::string source;
  };
  struct SharedState;

  void broker_loop(std::stop_token stop);
  void accept_loop(std::stop_token stop);

  SpscQueue<Outbound> outbound_;
  // start()/stop() own the worker threads and may call one another. A recursive
  // mutex keeps that lifecycle atomic without holding state_mutex_ while a
  // worker is being joined.
  std::recursive_mutex lifecycle_mutex_;
  mutable std::mutex state_mutex_;
  std::shared_ptr<SharedState> state_;
  std::jthread broker_;
  std::jthread acceptor_;
  std::mutex publish_mutex_;
  std::atomic<std::uint64_t> dropped_{};
};

}  // namespace dvo
