#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "dvo/spsc_queue.h"
#include "dvo/wasapi_capture.h"

namespace {

dvo::RawCapturedPacket raw_packet(dvo::RawSampleEncoding encoding,
                                  std::uint16_t bits,
                                  const std::vector<std::byte>& source) {
  const auto width = static_cast<std::uint16_t>(bits / 8);
  dvo::RawAudioBlockPool pool(1, source.size());
  auto block = pool.try_acquire();
  REQUIRE(block);
  std::memcpy(block->data(), source.data(), source.size());
  REQUIRE(block->set_size(source.size()));

  dvo::RawCapturedPacket packet;
  packet.format = {48000, 1, width, bits, bits, encoding};
  packet.frame_count = static_cast<std::uint32_t>(source.size() / width);
  packet.bytes = std::move(*block);
  packet.qpc_100ns = 123;
  packet.arrival_qpc_100ns = 456;
  packet.device_position = 789;
  packet.stream_epoch = 2;
  packet.sequence = 3;
  return packet;
}

template <typename T, std::size_t N>
std::vector<std::byte> bytes_of(const std::array<T, N>& values) {
  std::vector<std::byte> result(sizeof(values));
  std::memcpy(result.data(), values.data(), sizeof(values));
  return result;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<dvo::RawAudioBlock>);
static_assert(!std::is_copy_constructible_v<dvo::RawCapturedPacket>);

TEST_CASE("raw capture pool leases return automatically after queue rejection") {
  dvo::RawAudioBlockPool pool(2, 16);
  dvo::SpscQueue<dvo::RawCapturedPacket> queue(1);

  dvo::RawCapturedPacket first;
  first.bytes = std::move(*pool.try_acquire());
  dvo::RawCapturedPacket rejected;
  rejected.bytes = std::move(*pool.try_acquire());
  REQUIRE(pool.available() == 0);
  REQUIRE(queue.try_push(std::move(first)));
  REQUIRE_FALSE(queue.try_push(std::move(rejected)));
  rejected = {};
  REQUIRE(pool.available() == 1);

  dvo::RawCapturedPacket popped;
  REQUIRE(queue.try_pop(popped));
  popped = {};
  REQUIRE(pool.available() == 2);
}

TEST_CASE("raw pool state outlives endpoint reopen owner") {
  std::optional<dvo::RawAudioBlock> old_epoch_block;
  {
    dvo::RawAudioBlockPool old_epoch_pool(1, 8);
    old_epoch_block = old_epoch_pool.try_acquire();
    REQUIRE(old_epoch_block);
    REQUIRE(old_epoch_block->set_size(8));
    old_epoch_block->data()[0] = std::byte{0x5a};
  }
  REQUIRE(old_epoch_block->capacity() == 8);
  REQUIRE(old_epoch_block->data()[0] == std::byte{0x5a});
  old_epoch_block.reset();
}

TEST_CASE("silent raw packets materialize zeros without leasing a slot") {
  dvo::RawAudioBlockPool pool(1, 16);
  dvo::RawCapturedPacket silent;
  silent.format = {48000, 2, 8, 32, 32, dvo::RawSampleEncoding::float32};
  silent.frame_count = 2;
  silent.silent = true;
  silent.synthetic = true;
  const auto packet = dvo::materialize_audio_packet(std::move(silent));
  REQUIRE(pool.available() == 1);
  REQUIRE(packet.samples == std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F});
  REQUIRE(packet.synthetic);
}

TEST_CASE("all supported WASAPI encodings materialize on the consumer thread") {
  SECTION("float32 clamps to the normalized range") {
    const std::array values{-1.5F, 0.25F, 2.0F};
    const auto packet = dvo::materialize_audio_packet(raw_packet(
        dvo::RawSampleEncoding::float32, 32, bytes_of(values)));
    REQUIRE(packet.samples[0] == -1.0F);
    REQUIRE(packet.samples[1] == 0.25F);
    REQUIRE(packet.samples[2] == 1.0F);
  }

  SECTION("signed PCM16") {
    const std::array<std::int16_t, 3> values{-32768, 0, 32767};
    const auto packet = dvo::materialize_audio_packet(raw_packet(
        dvo::RawSampleEncoding::pcm_s16, 16, bytes_of(values)));
    REQUIRE(packet.samples[0] == -1.0F);
    REQUIRE(packet.samples[1] == 0.0F);
    REQUIRE(packet.samples[2] == Catch::Approx(32767.0F / 32768.0F));
  }

  SECTION("packed signed PCM24") {
    const std::vector<std::byte> values{
        std::byte{0x00}, std::byte{0x00}, std::byte{0x80},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0xff}, std::byte{0xff}, std::byte{0x7f}};
    const auto packet = dvo::materialize_audio_packet(raw_packet(
        dvo::RawSampleEncoding::pcm_s24, 24, values));
    REQUIRE(packet.samples[0] == -1.0F);
    REQUIRE(packet.samples[1] == 0.0F);
    REQUIRE(packet.samples[2] == Catch::Approx(8388607.0F / 8388608.0F));
  }

  SECTION("signed PCM32") {
    const std::array<std::int32_t, 3> values{
        std::numeric_limits<std::int32_t>::min(), 0,
        std::numeric_limits<std::int32_t>::max()};
    const auto packet = dvo::materialize_audio_packet(raw_packet(
        dvo::RawSampleEncoding::pcm_s32, 32, bytes_of(values)));
    REQUIRE(packet.samples[0] == -1.0F);
    REQUIRE(packet.samples[1] == 0.0F);
    REQUIRE(packet.samples[2] == Catch::Approx(
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) /
        2147483648.0));
  }
}
