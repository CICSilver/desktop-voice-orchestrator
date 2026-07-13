#include "dvo/wasapi_capture.h"

#include <Windows.h>
#include <audioclient.h>
#include <avrt.h>
#include <propkeydef.h>
#include <functiondiscoverykeys_devpkey.h>
#include <ksmedia.h>
#include <mmdeviceapi.h>
#include <propvarutil.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dvo {
namespace {

using Microsoft::WRL::ComPtr;

class ComApartment {
 public:
  ComApartment() {
    const auto hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    uninitialize_ = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) throw std::runtime_error("CoInitializeEx failed");
  }
  ~ComApartment() { if (uninitialize_) CoUninitialize(); }
 private:
  bool uninitialize_{};
};

class Handle {
 public:
  explicit Handle(HANDLE handle = nullptr) : handle_(handle) {}
  ~Handle() { if (handle_) CloseHandle(handle_); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  [[nodiscard]] HANDLE get() const { return handle_; }
 private:
  HANDLE handle_{};
};

std::string utf8(const wchar_t* value) {
  if (!value) return {};
  const auto size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
  std::string result(static_cast<std::size_t>(std::max(0, size)), '\0');
  if (size > 0) WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr, nullptr);
  if (!result.empty()) result.pop_back();
  return result;
}

std::wstring wide(const std::string& value) {
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) throw std::runtime_error("invalid UTF-8 device id");
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                      result.data(), size);
  return result;
}

void check(HRESULT hr, const char* operation) {
  if (FAILED(hr)) {
    char message[128];
    std::snprintf(message, sizeof(message), "%s failed (HRESULT 0x%08lx)", operation,
                  static_cast<unsigned long>(hr));
    throw std::runtime_error(message);
  }
}

std::uint64_t qpc_now_100ns() {
  LARGE_INTEGER counter{};
  LARGE_INTEGER frequency{};
  QueryPerformanceCounter(&counter);
  QueryPerformanceFrequency(&frequency);
  return static_cast<std::uint64_t>(static_cast<long double>(counter.QuadPart) * 10000000.0L /
                                    static_cast<long double>(frequency.QuadPart));
}

ComPtr<IMMDevice> get_device(IMMDeviceEnumerator* enumerator, const WasapiCaptureConfig& config) {
  ComPtr<IMMDevice> device;
  const auto flow = config.kind == AudioStreamKind::loopback ? eRender : eCapture;
  if (config.device_id.empty() || config.device_id == "default") {
    check(enumerator->GetDefaultAudioEndpoint(flow, eConsole, &device), "GetDefaultAudioEndpoint");
  } else {
    const auto id = wide(config.device_id);
    check(enumerator->GetDevice(id.c_str(), &device), "GetDevice");
  }
  return device;
}

bool is_float(const WAVEFORMATEX* format) {
  if (format->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) return true;
  if (format->wFormatTag != WAVE_FORMAT_EXTENSIBLE) return false;
  return IsEqualGUID(reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(format)->SubFormat,
                     KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
}

bool is_pcm(const WAVEFORMATEX* format) {
  if (format->wFormatTag == WAVE_FORMAT_PCM) return true;
  if (format->wFormatTag != WAVE_FORMAT_EXTENSIBLE) return false;
  return IsEqualGUID(reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(format)->SubFormat,
                     KSDATAFORMAT_SUBTYPE_PCM);
}

std::vector<float> convert_samples(const BYTE* data, UINT32 frames, const WAVEFORMATEX* format,
                                   bool silent) {
  const auto count = static_cast<std::size_t>(frames) * format->nChannels;
  std::vector<float> result(count, 0.0F);
  if (silent || !data) return result;
  if (is_float(format) && format->wBitsPerSample == 32) {
    std::memcpy(result.data(), data, result.size() * sizeof(float));
  } else if (is_pcm(format) && format->wBitsPerSample == 16) {
    const auto* source = reinterpret_cast<const std::int16_t*>(data);
    for (std::size_t i = 0; i < count; ++i) result[i] = static_cast<float>(source[i]) / 32768.0F;
  } else if (is_pcm(format) && format->wBitsPerSample == 24) {
    for (std::size_t i = 0; i < count; ++i) {
      const auto* p = data + i * 3;
      std::int32_t value = static_cast<std::int32_t>(p[0]) |
                           (static_cast<std::int32_t>(p[1]) << 8) |
                           (static_cast<std::int32_t>(p[2]) << 16);
      if (value & 0x800000) value |= ~0xffffff;
      result[i] = static_cast<float>(value) / 8388608.0F;
    }
  } else if (is_pcm(format) && format->wBitsPerSample == 32) {
    const auto* source = reinterpret_cast<const std::int32_t*>(data);
    for (std::size_t i = 0; i < count; ++i) result[i] = static_cast<float>(source[i] / 2147483648.0);
  } else {
    throw std::runtime_error("unsupported WASAPI sample format");
  }
  for (auto& value : result) value = std::clamp(value, -1.0F, 1.0F);
  return result;
}

std::string device_name(IMMDevice* device) {
  ComPtr<IPropertyStore> properties;
  check(device->OpenPropertyStore(STGM_READ, &properties), "OpenPropertyStore");
  PROPVARIANT value;
  PropVariantInit(&value);
  const auto hr = properties->GetValue(PKEY_Device_FriendlyName, &value);
  std::string result = SUCCEEDED(hr) && value.vt == VT_LPWSTR ? utf8(value.pwszVal) : "Unnamed device";
  PropVariantClear(&value);
  return result;
}

std::string device_id(IMMDevice* device) {
  LPWSTR id{};
  check(device->GetId(&id), "IMMDevice::GetId");
  const auto result = utf8(id);
  CoTaskMemFree(id);
  return result;
}

}  // namespace

WasapiCapture::~WasapiCapture() { stop(); }

void WasapiCapture::start(WasapiCaptureConfig config, PacketCallback packets, EventCallback events) {
  stop();
  running_.store(true, std::memory_order_release);
  thread_ = std::jthread([this, config = std::move(config), packets = std::move(packets),
                          events = std::move(events)](std::stop_token stop) mutable {
    run(stop, std::move(config), std::move(packets), std::move(events));
  });
}

void WasapiCapture::stop() {
  if (thread_.joinable()) {
    thread_.request_stop();
    thread_.join();
  }
  running_.store(false, std::memory_order_release);
}

void WasapiCapture::run(std::stop_token stop, WasapiCaptureConfig config,
                        PacketCallback packets, EventCallback events) {
  std::chrono::milliseconds backoff{500};
  while (!stop.stop_requested()) {
    try {
      capture_once(stop, config, packets, events);
      backoff = std::chrono::milliseconds{500};
    } catch (const std::exception& e) {
      events("capture_error", std::string(to_string(config.kind)) + ": " + e.what());
      const auto deadline = std::chrono::steady_clock::now() + backoff;
      while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      backoff = std::min(backoff * 2, std::chrono::milliseconds{5000});
    }
  }
  running_.store(false, std::memory_order_release);
}

void WasapiCapture::capture_once(std::stop_token stop, const WasapiCaptureConfig& config,
                                 const PacketCallback& packets, const EventCallback& events) {
  ComApartment apartment;
  DWORD task_index{};
  HANDLE mmcss = AvSetMmThreadCharacteristicsW(L"Audio", &task_index);

  ComPtr<IMMDeviceEnumerator> enumerator;
  check(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                         IID_PPV_ARGS(&enumerator)), "CoCreateInstance(MMDeviceEnumerator)");
  auto device = get_device(enumerator.Get(), config);
  const auto id = device_id(device.Get());
  events("capture_started", std::string(to_string(config.kind)) + ": " + device_name(device.Get()));

  ComPtr<IAudioClient> client;
  check(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                         reinterpret_cast<void**>(client.GetAddressOf())), "Activate(IAudioClient)");
  WAVEFORMATEX* raw_format{};
  check(client->GetMixFormat(&raw_format), "GetMixFormat");
  std::unique_ptr<WAVEFORMATEX, decltype(&CoTaskMemFree)> format(raw_format, CoTaskMemFree);
  const DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                      (config.kind == AudioStreamKind::loopback ? AUDCLNT_STREAMFLAGS_LOOPBACK : 0);
  check(client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, format.get(), nullptr),
        "IAudioClient::Initialize");
  Handle audio_event(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!audio_event.get()) throw std::runtime_error("CreateEvent failed");
  check(client->SetEventHandle(audio_event.get()), "SetEventHandle");

  ComPtr<IAudioCaptureClient> capture;
  check(client->GetService(IID_PPV_ARGS(capture.GetAddressOf())), "GetService(IAudioCaptureClient)");
  check(client->Start(), "IAudioClient::Start");

  constexpr std::uint64_t idle_packet_100ns = 100000;  // 10 ms
  std::uint64_t next_idle_qpc = qpc_now_100ns() + idle_packet_100ns;
  std::uint64_t next_idle_position{};
  const auto idle_frames = std::max<UINT32>(1, format->nSamplesPerSec / 100);

  const auto emit_idle_loopback = [&] {
    if (config.kind != AudioStreamKind::loopback) return;
    const auto now = qpc_now_100ns();
    // Bound catch-up work after a debugger pause. The discontinuity flag tells
    // downstream stateful components not to bridge an unrepresented gap.
    std::uint32_t emitted{};
    while (now >= next_idle_qpc && emitted < 5) {
      AudioPacket packet;
      packet.stream = config.kind;
      packet.format = {format->nSamplesPerSec, format->nChannels};
      packet.qpc_100ns = next_idle_qpc;
      packet.device_position = next_idle_position;
      packet.silent = true;
      packet.synthetic = true;
      packet.discontinuity = emitted == 0 && now > next_idle_qpc + idle_packet_100ns * 5;
      packet.samples.assign(static_cast<std::size_t>(idle_frames) * format->nChannels, 0.0F);
      packets(std::move(packet));
      next_idle_qpc += idle_packet_100ns;
      next_idle_position += idle_frames;
      ++emitted;
    }
    if (now > next_idle_qpc + idle_packet_100ns * 5) next_idle_qpc = now + idle_packet_100ns;
  };

  while (!stop.stop_requested()) {
    const auto wait = WaitForSingleObject(audio_event.get(), 10);
    if (wait == WAIT_TIMEOUT) {
      emit_idle_loopback();
      continue;
    }
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("WaitForSingleObject failed");
    UINT32 next{};
    check(capture->GetNextPacketSize(&next), "GetNextPacketSize");
    while (next != 0) {
      BYTE* data{};
      UINT32 frames{};
      DWORD packet_flags{};
      UINT64 device_position{};
      UINT64 qpc{};
      check(capture->GetBuffer(&data, &frames, &packet_flags, &device_position, &qpc),
            "IAudioCaptureClient::GetBuffer");
      AudioPacket packet;
      packet.stream = config.kind;
      packet.format = {format->nSamplesPerSec, format->nChannels};
      packet.qpc_100ns = qpc;
      packet.device_position = device_position;
      packet.silent = (packet_flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
      packet.discontinuity = (packet_flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) != 0;
      packet.samples = convert_samples(data, frames, format.get(), packet.silent);
      check(capture->ReleaseBuffer(frames), "IAudioCaptureClient::ReleaseBuffer");
      if (config.kind == AudioStreamKind::loopback) {
        next_idle_qpc = qpc + static_cast<std::uint64_t>(frames) * 10000000ULL /
                                  format->nSamplesPerSec;
        next_idle_position = device_position + frames;
      }
      packets(std::move(packet));
      check(capture->GetNextPacketSize(&next), "GetNextPacketSize");
    }
    if (next == 0) emit_idle_loopback();
  }
  client->Stop();
  if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
  events("capture_stopped", std::string(to_string(config.kind)) + ": " + id);
}

std::vector<AudioDeviceInfo> WasapiCapture::list_devices() {
  ComApartment apartment;
  ComPtr<IMMDeviceEnumerator> enumerator;
  check(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                         IID_PPV_ARGS(&enumerator)), "CoCreateInstance(MMDeviceEnumerator)");
  std::vector<AudioDeviceInfo> result;
  for (const auto [flow, kind] : {std::pair{eCapture, AudioStreamKind::microphone},
                                  std::pair{eRender, AudioStreamKind::loopback}}) {
    ComPtr<IMMDevice> default_device;
    std::string default_id;
    if (SUCCEEDED(enumerator->GetDefaultAudioEndpoint(flow, eConsole, &default_device))) {
      default_id = device_id(default_device.Get());
    }
    ComPtr<IMMDeviceCollection> collection;
    check(enumerator->EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE, &collection), "EnumAudioEndpoints");
    UINT count{};
    check(collection->GetCount(&count), "IMMDeviceCollection::GetCount");
    for (UINT i = 0; i < count; ++i) {
      ComPtr<IMMDevice> device;
      check(collection->Item(i, &device), "IMMDeviceCollection::Item");
      auto id = device_id(device.Get());
      result.push_back({id, device_name(device.Get()), kind, id == default_id});
    }
  }
  return result;
}

}  // namespace dvo
