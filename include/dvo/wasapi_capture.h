#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "dvo/audio_types.h"

namespace dvo {

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
};

class WasapiCapture {
 public:
  using PacketCallback = std::function<void(AudioPacket)>;
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
                    const PacketCallback& packets, const EventCallback& events);

  std::jthread thread_;
  std::atomic<bool> running_{};
};

}  // namespace dvo
