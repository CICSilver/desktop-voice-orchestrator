#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "dvo/ring_buffer.h"

TEST_CASE("timed ring buffer preserves absolute indices and wraps") {
  dvo::TimedRingBuffer ring(5);
  const std::array first{0.0F, 1.0F, 2.0F, 3.0F};
  ring.push(0, first);
  REQUIRE(ring.head() == 0);
  REQUIRE(ring.tail() == 4);
  REQUIRE(ring.slice({1, 4}).samples == std::vector<float>{1, 2, 3});

  const std::array second{4.0F, 5.0F, 6.0F, 7.0F};
  ring.push(4, second);
  REQUIRE(ring.head() == 3);
  REQUIRE(ring.tail() == 8);
  const auto wrapped = ring.slice({0, 9});
  REQUIRE(wrapped.truncated_left);
  REQUIRE(wrapped.truncated_right);
  REQUIRE(wrapped.actual.start == 3);
  REQUIRE(wrapped.actual.end == 8);
  REQUIRE(wrapped.samples == std::vector<float>{3, 4, 5, 6, 7});
}

TEST_CASE("timed ring buffer snapshots remain coherent during writes") {
  dvo::TimedRingBuffer ring(64);
  std::atomic<bool> done{};
  std::atomic<bool> coherent{true};
  std::thread writer([&] {
    for (std::uint64_t base = 0; base < 4000; base += 4) {
      const std::array values{static_cast<float>(base), static_cast<float>(base + 1),
                              static_cast<float>(base + 2), static_cast<float>(base + 3)};
      ring.push(base, values);
    }
    done.store(true, std::memory_order_release);
  });
  std::thread reader([&] {
    while (!done.load(std::memory_order_acquire)) {
      const auto tail = ring.tail();
      const auto start = tail > 32 ? tail - 32 : 0;
      const auto slice = ring.slice({start, tail});
      for (std::size_t i = 0; i < slice.samples.size(); ++i) {
        if (slice.samples[i] != static_cast<float>(slice.actual.start + i)) {
          coherent.store(false, std::memory_order_release);
          return;
        }
      }
    }
  });
  writer.join();
  reader.join();
  REQUIRE(coherent.load(std::memory_order_acquire));
}

TEST_CASE("timed ring buffer handles oversized writes and discontinuities") {
  dvo::TimedRingBuffer ring(5);
  const std::array samples{10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F};
  ring.push(100, samples);
  REQUIRE(ring.head() == 103);
  REQUIRE(ring.tail() == 108);
  REQUIRE(ring.slice({103, 108}).samples == std::vector<float>{13, 14, 15, 16, 17});

  const std::array replacement{1.0F, 2.0F};
  ring.push(500, replacement);
  REQUIRE(ring.head() == 500);
  REQUIRE(ring.tail() == 502);
  REQUIRE(ring.slice({500, 502}).samples == std::vector<float>{1, 2});
}

TEST_CASE("cooperative ring slices match an atomic slice across small chunks") {
  dvo::TimedRingBuffer ring(2048);
  std::vector<float> samples(1500);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    samples[i] = static_cast<float>(i);
  }
  ring.push(100, samples);

  const auto atomic = ring.slice({123, 1477});
  const auto cooperative = ring.slice_cooperative({123, 1477}, 17);
  CHECK(cooperative.requested.start == atomic.requested.start);
  CHECK(cooperative.requested.end == atomic.requested.end);
  CHECK(cooperative.actual.start == atomic.actual.start);
  CHECK(cooperative.actual.end == atomic.actual.end);
  CHECK(cooperative.truncated_left == atomic.truncated_left);
  CHECK(cooperative.truncated_right == atomic.truncated_right);
  CHECK(cooperative.samples == atomic.samples);

  CHECK_THROWS_AS(ring.slice_cooperative({123, 1477}, 0),
                  std::invalid_argument);
}
