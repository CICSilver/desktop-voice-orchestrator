#include "dvo/wasapi_capture.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace dvo {

struct RawAudioBlockPoolState {
  RawAudioBlockPoolState(std::size_t slots, std::size_t slot_bytes)
      : slot_count(slots), bytes_per_slot(slot_bytes),
        storage(std::make_unique<std::byte[]>(slots * slot_bytes)),
        leased(std::make_unique<std::atomic_uint8_t[]>(slots)) {
    for (std::size_t i = 0; i < slot_count; ++i) {
      leased[i].store(0, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::byte* slot_data(std::size_t slot) noexcept {
    return storage.get() + slot * bytes_per_slot;
  }

  void release(std::size_t slot) noexcept {
    if (slot >= slot_count) return;
    if (leased[slot].exchange(0, std::memory_order_release) != 0) {
      leased_count.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  const std::size_t slot_count;
  const std::size_t bytes_per_slot;
  std::unique_ptr<std::byte[]> storage;
  std::unique_ptr<std::atomic_uint8_t[]> leased;
  std::atomic<std::size_t> next_slot{};
  std::atomic<std::size_t> leased_count{};
};

RawAudioBlock::RawAudioBlock(std::shared_ptr<RawAudioBlockPoolState> state,
                             std::size_t slot) noexcept
    : state_(std::move(state)), slot_(slot) {}

RawAudioBlock::~RawAudioBlock() { reset(); }

RawAudioBlock::RawAudioBlock(RawAudioBlock&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), size_(other.size_) {
  other.slot_ = 0;
  other.size_ = 0;
}

RawAudioBlock& RawAudioBlock::operator=(RawAudioBlock&& other) noexcept {
  if (this == &other) return *this;
  reset();
  state_ = std::move(other.state_);
  slot_ = other.slot_;
  size_ = other.size_;
  other.slot_ = 0;
  other.size_ = 0;
  return *this;
}

std::byte* RawAudioBlock::data() noexcept {
  return state_ ? state_->slot_data(slot_) : nullptr;
}

const std::byte* RawAudioBlock::data() const noexcept {
  return state_ ? state_->slot_data(slot_) : nullptr;
}

std::size_t RawAudioBlock::capacity() const noexcept {
  return state_ ? state_->bytes_per_slot : 0;
}

bool RawAudioBlock::set_size(std::size_t size) noexcept {
  if (!state_ || size > state_->bytes_per_slot) return false;
  size_ = size;
  return true;
}

void RawAudioBlock::reset() noexcept {
  if (!state_) return;
  state_->release(slot_);
  state_.reset();
  slot_ = 0;
  size_ = 0;
}

RawAudioBlockPool::RawAudioBlockPool(std::size_t slot_count,
                                     std::size_t bytes_per_slot) {
  if (slot_count == 0 || bytes_per_slot == 0) {
    throw std::invalid_argument("raw audio pool dimensions must be non-zero");
  }
  if (slot_count > std::numeric_limits<std::size_t>::max() / bytes_per_slot) {
    throw std::overflow_error("raw audio pool size overflow");
  }
  // A corrupt endpoint descriptor must not reserve unbounded process memory.
  constexpr std::size_t kMaximumPoolBytes = 512ULL * 1024ULL * 1024ULL;
  if (slot_count * bytes_per_slot > kMaximumPoolBytes) {
    throw std::length_error("raw audio pool exceeds 512 MiB safety bound");
  }
  state_ = std::make_shared<RawAudioBlockPoolState>(slot_count, bytes_per_slot);
}

std::optional<RawAudioBlock> RawAudioBlockPool::try_acquire() const noexcept {
  if (!state_) return std::nullopt;
  const auto count = state_->slot_count;
  const auto first = state_->next_slot.fetch_add(1, std::memory_order_relaxed) % count;
  for (std::size_t offset = 0; offset < count; ++offset) {
    const auto slot = (first + offset) % count;
    std::uint8_t expected{};
    if (state_->leased[slot].compare_exchange_strong(
            expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
      state_->leased_count.fetch_add(1, std::memory_order_relaxed);
      return RawAudioBlock(state_, slot);
    }
  }
  return std::nullopt;
}

std::size_t RawAudioBlockPool::capacity() const noexcept {
  return state_ ? state_->slot_count : 0;
}

std::size_t RawAudioBlockPool::available() const noexcept {
  if (!state_) return 0;
  return state_->slot_count - state_->leased_count.load(std::memory_order_relaxed);
}

std::size_t RawAudioBlockPool::bytes_per_slot() const noexcept {
  return state_ ? state_->bytes_per_slot : 0;
}

namespace {

std::size_t sample_width(RawSampleEncoding encoding) {
  switch (encoding) {
    case RawSampleEncoding::float32: return 4;
    case RawSampleEncoding::pcm_s16: return 2;
    case RawSampleEncoding::pcm_s24: return 3;
    case RawSampleEncoding::pcm_s32: return 4;
  }
  throw std::invalid_argument("unsupported raw sample encoding");
}

float decode_sample(const std::byte* source, RawSampleEncoding encoding) {
  switch (encoding) {
    case RawSampleEncoding::float32: {
      float value{};
      std::memcpy(&value, source, sizeof(value));
      return std::clamp(value, -1.0F, 1.0F);
    }
    case RawSampleEncoding::pcm_s16: {
      std::int16_t value{};
      std::memcpy(&value, source, sizeof(value));
      return static_cast<float>(value) / 32768.0F;
    }
    case RawSampleEncoding::pcm_s24: {
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(source);
      std::uint32_t packed = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U);
      if ((packed & 0x00800000U) != 0) packed |= 0xff000000U;
      return static_cast<float>(static_cast<std::int32_t>(packed)) / 8388608.0F;
    }
    case RawSampleEncoding::pcm_s32: {
      std::int32_t value{};
      std::memcpy(&value, source, sizeof(value));
      return static_cast<float>(static_cast<double>(value) / 2147483648.0);
    }
  }
  throw std::invalid_argument("unsupported raw sample encoding");
}

}  // namespace

AudioPacket materialize_audio_packet(RawCapturedPacket packet) {
  const auto& format = packet.format;
  if (format.sample_rate == 0 || format.channels == 0 || format.block_align == 0) {
    throw std::invalid_argument("raw audio packet has an invalid format");
  }
  const auto width = sample_width(format.encoding);
  const auto packed_frame_bytes = width * static_cast<std::size_t>(format.channels);
  if (format.block_align < packed_frame_bytes) {
    throw std::invalid_argument("raw audio block alignment is smaller than one frame");
  }
  if (packet.frame_count > std::numeric_limits<std::size_t>::max() /
                               static_cast<std::size_t>(format.block_align)) {
    throw std::overflow_error("raw audio packet byte size overflow");
  }
  if (packet.frame_count > std::numeric_limits<std::size_t>::max() /
                               static_cast<std::size_t>(format.channels)) {
    throw std::overflow_error("raw audio packet sample count overflow");
  }

  const auto required_bytes = static_cast<std::size_t>(packet.frame_count) *
                              static_cast<std::size_t>(format.block_align);
  const auto sample_count = static_cast<std::size_t>(packet.frame_count) *
                            static_cast<std::size_t>(format.channels);
  if (!packet.silent && (!packet.bytes || packet.bytes.size() < required_bytes)) {
    throw std::invalid_argument("raw audio packet does not own enough source bytes");
  }

  AudioPacket result;
  result.stream = packet.stream;
  result.format = {format.sample_rate, format.channels};
  result.qpc_100ns = packet.qpc_100ns;
  result.arrival_qpc_100ns = packet.arrival_qpc_100ns;
  result.device_position = packet.device_position;
  result.stream_epoch = packet.stream_epoch;
  result.sequence = packet.sequence;
  result.silent = packet.silent;
  result.discontinuity = packet.discontinuity;
  result.timestamp_error = packet.timestamp_error;
  result.synthetic = packet.synthetic;
  result.samples.resize(sample_count, 0.0F);
  if (packet.silent) return result;

  const auto* source = packet.bytes.data();
  for (std::size_t frame = 0; frame < packet.frame_count; ++frame) {
    const auto* frame_source = source + frame * format.block_align;
    for (std::size_t channel = 0; channel < format.channels; ++channel) {
      result.samples[frame * format.channels + channel] =
          decode_sample(frame_source + channel * width, format.encoding);
    }
  }
  return result;
}

}  // namespace dvo
