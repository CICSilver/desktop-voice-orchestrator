#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "dvo/audio_types.h"

namespace dvo {

// Immutable description copied from WAVEFORMATEX before the realtime capture
// loop begins. No COM-owned format pointer crosses the GetBuffer boundary.
enum class RawSampleEncoding { float32, pcm_s16, pcm_s24, pcm_s32 };

struct RawAudioFormat {
  std::uint32_t sample_rate{};
  std::uint16_t channels{};
  std::uint16_t block_align{};
  std::uint16_t bits_per_sample{};
  std::uint16_t valid_bits_per_sample{};
  RawSampleEncoding encoding{RawSampleEncoding::float32};
};

struct RawAudioBlockPoolState;

// Move-only slot lease. Destroying a lease on either the producer or consumer
// thread returns the fixed-size slot to its pool without allocating or locking.
class RawAudioBlock {
 public:
  RawAudioBlock() = default;
  ~RawAudioBlock();
  RawAudioBlock(const RawAudioBlock&) = delete;
  RawAudioBlock& operator=(const RawAudioBlock&) = delete;
  RawAudioBlock(RawAudioBlock&& other) noexcept;
  RawAudioBlock& operator=(RawAudioBlock&& other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }
  [[nodiscard]] std::byte* data() noexcept;
  [[nodiscard]] const std::byte* data() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] bool set_size(std::size_t size) noexcept;
  void reset() noexcept;

 private:
  friend class RawAudioBlockPool;
  RawAudioBlock(std::shared_ptr<RawAudioBlockPoolState> state,
                std::size_t slot) noexcept;

  std::shared_ptr<RawAudioBlockPoolState> state_;
  std::size_t slot_{};
  std::size_t size_{};
};

// Storage is allocated once, before IAudioClient::Start. Slot acquisition is
// an atomic CAS scan because rejected queue pushes may release on the capture
// thread while accepted packets release on the processing thread.
class RawAudioBlockPool {
 public:
  RawAudioBlockPool(std::size_t slot_count, std::size_t bytes_per_slot);
  RawAudioBlockPool(const RawAudioBlockPool&) = delete;
  RawAudioBlockPool& operator=(const RawAudioBlockPool&) = delete;
  RawAudioBlockPool(RawAudioBlockPool&&) noexcept = default;
  RawAudioBlockPool& operator=(RawAudioBlockPool&&) noexcept = default;

  [[nodiscard]] std::optional<RawAudioBlock> try_acquire() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t available() const noexcept;
  [[nodiscard]] std::size_t bytes_per_slot() const noexcept;

 private:
  std::shared_ptr<RawAudioBlockPoolState> state_;
};

struct RawCapturedPacket {
  AudioStreamKind stream{AudioStreamKind::microphone};
  RawAudioFormat format{};
  std::uint32_t frame_count{};
  RawAudioBlock bytes;
  std::uint64_t qpc_100ns{};
  std::uint64_t arrival_qpc_100ns{};
  std::uint64_t device_position{};
  std::uint64_t stream_epoch{};
  std::uint64_t sequence{};
  // Number of non-silent WASAPI packets skipped because no raw pool slot was
  // available since the preceding callback. Runtime telemetry accounts it
  // without requiring logging or allocation on the capture thread.
  std::uint64_t pool_drops_before{};
  bool silent{};
  bool discontinuity{};
  bool timestamp_error{};
  bool synthetic{};

  RawCapturedPacket() = default;
  RawCapturedPacket(const RawCapturedPacket&) = delete;
  RawCapturedPacket& operator=(const RawCapturedPacket&) = delete;
  RawCapturedPacket(RawCapturedPacket&&) noexcept = default;
  RawCapturedPacket& operator=(RawCapturedPacket&&) noexcept = default;
};

// The only allocation in this conversion is AudioPacket::samples, and callers
// invoke it on the processing thread after removing the raw packet from SPSC.
[[nodiscard]] AudioPacket materialize_audio_packet(RawCapturedPacket packet);

struct AudioDeviceInfo {
  std::string id;
  std::string name;
  AudioStreamKind kind{};
  bool is_default{};
};

struct WasapiCaptureConfig {
  AudioStreamKind kind{AudioStreamKind::microphone};
  std::string device_id{"default"};
  bool follow_default{true};
  bool request_raw{false};
  bool request_post_volume_loopback{false};
  // Runtime sets this to at least its ingress queue capacity plus spare slots.
  std::size_t raw_pool_slots{256};
};

class WasapiCapture {
 public:
  // false means the bounded runtime queue rejected the packet. The capture
  // loop marks the next accepted packet discontinuous and never waits.
  using PacketCallback = std::function<bool(RawCapturedPacket)>;
  using EventCallback = std::function<void(std::string, std::string)>;

  WasapiCapture() = default;
  ~WasapiCapture();
  WasapiCapture(const WasapiCapture&) = delete;
  WasapiCapture& operator=(const WasapiCapture&) = delete;

  void start(WasapiCaptureConfig config, PacketCallback packets, EventCallback events);
  void stop();
  [[nodiscard]] bool running() const { return running_.load(std::memory_order_acquire); }

  static std::vector<AudioDeviceInfo> list_devices();

 private:
  void run(std::stop_token stop, WasapiCaptureConfig config, PacketCallback packets,
           EventCallback events);
  void capture_once(std::stop_token stop, const WasapiCaptureConfig& config,
                    const PacketCallback& packets, const EventCallback& events,
                    std::uint64_t stream_epoch);

  std::jthread thread_;
  std::atomic<bool> running_{};
};

}  // namespace dvo
