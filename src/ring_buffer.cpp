#include "dvo/ring_buffer.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace dvo {

TimedRingBuffer::TimedRingBuffer(std::size_t capacity_samples) : data_(capacity_samples) {
  if (capacity_samples == 0) throw std::invalid_argument("ring buffer capacity must be positive");
}

void TimedRingBuffer::push(std::uint64_t first_sample, std::span<const float> samples) {
  if (samples.empty()) return;
  std::unique_lock lock(mutex_);
  if (!initialized_ || first_sample != tail_) {
    head_ = first_sample;
    tail_ = first_sample;
    initialized_ = true;
  }

  if (samples.size() >= data_.size()) {
    const auto kept = samples.last(data_.size());
    const auto new_head = first_sample + samples.size() - data_.size();
    for (std::size_t i = 0; i < kept.size(); ++i) {
      data_[static_cast<std::size_t>((new_head + i) % data_.size())] = kept[i];
    }
    head_ = new_head;
    tail_ = first_sample + samples.size();
    return;
  }

  for (float sample : samples) {
    data_[static_cast<std::size_t>(tail_ % data_.size())] = sample;
    ++tail_;
    if (tail_ - head_ > data_.size()) ++head_;
  }
}

void TimedRingBuffer::reset(std::uint64_t next_sample) {
  std::unique_lock lock(mutex_);
  head_ = next_sample;
  tail_ = next_sample;
  initialized_ = true;
}

RingSlice TimedRingBuffer::slice(SampleSpan span) const {
  std::shared_lock lock(mutex_);
  RingSlice result;
  result.requested = span;
  if (!initialized_ || span.empty()) return result;

  result.actual.start = std::max(span.start, head_);
  result.actual.end = std::min(span.end, tail_);
  result.truncated_left = result.actual.start != span.start;
  result.truncated_right = result.actual.end != span.end;
  if (result.actual.empty()) return result;

  result.samples.reserve(static_cast<std::size_t>(result.actual.size()));
  for (auto index = result.actual.start; index < result.actual.end; ++index) {
    result.samples.push_back(data_[static_cast<std::size_t>(index % data_.size())]);
  }
  return result;
}

RingSlice TimedRingBuffer::slice_cooperative(SampleSpan span,
                                             std::size_t chunk_samples) const {
  if (chunk_samples == 0) {
    throw std::invalid_argument("cooperative ring slice chunk must be positive");
  }

  RingSlice result;
  result.requested = span;
  {
    std::shared_lock lock(mutex_);
    if (!initialized_ || span.empty()) return result;
    result.actual.start = std::max(span.start, head_);
    result.actual.end = std::min(span.end, tail_);
    result.truncated_left = result.actual.start != span.start;
    result.truncated_right = result.actual.end != span.end;
  }
  if (result.actual.empty()) return result;

  result.samples.reserve(static_cast<std::size_t>(result.actual.size()));
  auto cursor = result.actual.start;
  while (cursor < result.actual.end) {
    const auto remaining = result.actual.end - cursor;
    const auto chunk_size = std::min<std::uint64_t>(
        remaining, static_cast<std::uint64_t>(chunk_samples));
    const auto chunk_end = cursor + chunk_size;
    {
      std::shared_lock lock(mutex_);
      if (!initialized_ || head_ > cursor || tail_ < chunk_end) {
        result.samples.clear();
        result.actual.end = result.actual.start;
        result.truncated_left = true;
        result.truncated_right = true;
        return result;
      }
      for (auto index = cursor; index < chunk_end; ++index) {
        result.samples.push_back(
            data_[static_cast<std::size_t>(index % data_.size())]);
      }
    }
    cursor = chunk_end;
    if (cursor < result.actual.end) std::this_thread::yield();
  }
  return result;
}

std::uint64_t TimedRingBuffer::head() const {
  std::shared_lock lock(mutex_);
  return head_;
}

std::uint64_t TimedRingBuffer::tail() const {
  std::shared_lock lock(mutex_);
  return tail_;
}

std::size_t TimedRingBuffer::size() const {
  std::shared_lock lock(mutex_);
  return initialized_ ? static_cast<std::size_t>(tail_ - head_) : 0;
}

}  // namespace dvo
