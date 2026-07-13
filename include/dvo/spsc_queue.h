#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace dvo {

template <typename T>
class SpscQueue {
 public:
  explicit SpscQueue(std::size_t capacity)
      : capacity_(capacity + 1), storage_(std::make_unique<std::optional<T>[]>(capacity_)) {}

  SpscQueue(const SpscQueue&) = delete;
  SpscQueue& operator=(const SpscQueue&) = delete;

  bool try_push(const T& value) { return try_emplace(value); }
  bool try_push(T&& value) { return try_emplace(std::move(value)); }

 private:
  template <typename U>
  bool try_emplace(U&& value) {
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto next = increment(tail);
    if (next == head_.load(std::memory_order_acquire)) return false;
    storage_[tail].emplace(std::forward<U>(value));
    tail_.store(next, std::memory_order_release);
    return true;
  }

 public:

  bool try_pop(T& value) {
    const auto head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_acquire)) return false;
    value = std::move(*storage_[head]);
    storage_[head].reset();
    head_.store(increment(head), std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::size_t size() const {
    const auto head = head_.load(std::memory_order_acquire);
    const auto tail = tail_.load(std::memory_order_acquire);
    return tail >= head ? tail - head : capacity_ - head + tail;
  }

  [[nodiscard]] std::size_t capacity() const { return capacity_ - 1; }

 private:
  [[nodiscard]] std::size_t increment(std::size_t value) const {
    return (value + 1) % capacity_;
  }

  const std::size_t capacity_;
  std::unique_ptr<std::optional<T>[]> storage_;
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace dvo
