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
#include <limits>
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

class MmcssRegistration {
 public:
  explicit MmcssRegistration(const wchar_t* task_name) {
    DWORD task_index{};
    handle_ = AvSetMmThreadCharacteristicsW(task_name, &task_index);
  }
  ~MmcssRegistration() {
    if (handle_) AvRevertMmThreadCharacteristics(handle_);
  }
  MmcssRegistration(const MmcssRegistration&) = delete;
  MmcssRegistration& operator=(const MmcssRegistration&) = delete;

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

std::string hresult_hex(HRESULT hr) {
  char result[16];
  std::snprintf(result, sizeof(result), "0x%08lx", static_cast<unsigned long>(hr));
  return result;
}

std::uint64_t qpc_now_100ns() {
  LARGE_INTEGER counter{};
  static const auto frequency = [] {
    LARGE_INTEGER value{};
    QueryPerformanceFrequency(&value);
    return value.QuadPart;
  }();
  QueryPerformanceCounter(&counter);
  return static_cast<std::uint64_t>(static_cast<long double>(counter.QuadPart) * 10000000.0L /
                                    static_cast<long double>(frequency));
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

RawAudioFormat describe_format(const WAVEFORMATEX* format) {
  if (!format || format->nSamplesPerSec == 0 || format->nChannels == 0 ||
      format->nBlockAlign == 0) {
    throw std::runtime_error("invalid WASAPI mix format");
  }
  RawAudioFormat result;
  result.sample_rate = format->nSamplesPerSec;
  result.channels = format->nChannels;
  result.block_align = format->nBlockAlign;
  result.bits_per_sample = format->wBitsPerSample;
  result.valid_bits_per_sample = format->wBitsPerSample;
  if (format->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
      format->cbSize >= sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)) {
    const auto* extensible = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(format);
    if (extensible->Samples.wValidBitsPerSample != 0) {
      result.valid_bits_per_sample = extensible->Samples.wValidBitsPerSample;
    }
  }

  if (is_float(format) && format->wBitsPerSample == 32) {
    result.encoding = RawSampleEncoding::float32;
  } else if (is_pcm(format) && format->wBitsPerSample == 16) {
    result.encoding = RawSampleEncoding::pcm_s16;
  } else if (is_pcm(format) && format->wBitsPerSample == 24) {
    result.encoding = RawSampleEncoding::pcm_s24;
  } else if (is_pcm(format) && format->wBitsPerSample == 32) {
    result.encoding = RawSampleEncoding::pcm_s32;
  } else {
    throw std::runtime_error("unsupported WASAPI sample format");
  }
  const auto bytes_per_sample = static_cast<std::size_t>(format->wBitsPerSample / 8);
  if (bytes_per_sample == 0 ||
      format->nBlockAlign < bytes_per_sample * format->nChannels) {
    throw std::runtime_error("invalid WASAPI block alignment");
  }
  return result;
}

class CaptureBufferLease {
 public:
  CaptureBufferLease(IAudioCaptureClient* capture, UINT32 frames) noexcept
      : capture_(capture), frames_(frames) {}
  ~CaptureBufferLease() {
    if (capture_) (void)capture_->ReleaseBuffer(frames_);
  }
  CaptureBufferLease(const CaptureBufferLease&) = delete;
  CaptureBufferLease& operator=(const CaptureBufferLease&) = delete;

  void release_checked() {
    auto* capture = capture_;
    capture_ = nullptr;
    check(capture->ReleaseBuffer(frames_), "IAudioCaptureClient::ReleaseBuffer");
  }

 private:
  IAudioCaptureClient* capture_{};
  UINT32 frames_{};
};

[[nodiscard]] std::size_t checked_packet_bytes(UINT32 frames,
                                               std::uint16_t block_align) {
  if (frames > std::numeric_limits<std::size_t>::max() /
                   static_cast<std::size_t>(block_align)) {
    throw std::overflow_error("WASAPI packet byte size overflow");
  }
  return static_cast<std::size_t>(frames) * block_align;
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
  std::uint64_t stream_epoch{};
  while (!stop.stop_requested()) {
    try {
      capture_once(stop, config, packets, events, ++stream_epoch);
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
                                 const PacketCallback& packets, const EventCallback& events,
                                 std::uint64_t stream_epoch) {
  ComApartment apartment;

  ComPtr<IMMDeviceEnumerator> enumerator;
  check(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                         IID_PPV_ARGS(&enumerator)), "CoCreateInstance(MMDeviceEnumerator)");
  auto device = get_device(enumerator.Get(), config);
  const auto id = device_id(device.Get());
  events("capture_started", std::string(to_string(config.kind)) + ": " + device_name(device.Get()));

  const auto activate_client = [&] {
    ComPtr<IAudioClient> result;
    check(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                           reinterpret_cast<void**>(result.GetAddressOf())),
          "Activate(IAudioClient)");
    return result;
  };
  using WaveFormat = std::unique_ptr<WAVEFORMATEX, decltype(&CoTaskMemFree)>;
  const auto get_mix_format = [](IAudioClient* audio_client) {
    WAVEFORMATEX* raw_format{};
    check(audio_client->GetMixFormat(&raw_format), "GetMixFormat");
    return WaveFormat(raw_format, CoTaskMemFree);
  };

  constexpr auto post_volume_loopback = static_cast<AUDCLNT_STREAMOPTIONS>(0x8);
  const auto requested_option =
      config.kind == AudioStreamKind::microphone && config.request_raw
          ? AUDCLNT_STREAMOPTIONS_RAW
          : (config.kind == AudioStreamKind::loopback && config.request_post_volume_loopback
                 ? post_volume_loopback
                 : AUDCLNT_STREAMOPTIONS_NONE);

  auto client = activate_client();
  bool requested_option_applied{};
  std::string capability_fallback_reason;
  if (requested_option != AUDCLNT_STREAMOPTIONS_NONE) {
    ComPtr<IAudioClient2> client2;
    const auto query = client.As(&client2);
    if (SUCCEEDED(query)) {
      AudioClientProperties properties{};
      properties.cbSize = sizeof(properties);
      properties.bIsOffload = FALSE;
      properties.eCategory = config.kind == AudioStreamKind::microphone
                                 ? AudioCategory_Communications
                                 : AudioCategory_Media;
      // SDK 10.0.22621 does not name POST_VOLUME_LOOPBACK yet. Unsupported
      // systems can reject either this call or the later Initialize call.
      properties.Options = requested_option;
      const auto property_result = client2->SetClientProperties(&properties);
      if (SUCCEEDED(property_result)) {
        requested_option_applied = true;
      } else {
        capability_fallback_reason =
            "SetClientProperties rejected requested stream option (HRESULT " +
            hresult_hex(property_result) + ")";
        // SetClientProperties is allowed to partially alter client state even
        // when it fails. Reactivate so the fallback is guaranteed to be plain.
        client = activate_client();
      }
    } else {
      capability_fallback_reason =
          "IAudioClient2 unavailable (HRESULT " + hresult_hex(query) + ")";
    }
  }

  auto format = get_mix_format(client.Get());
  const DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                      (config.kind == AudioStreamKind::loopback ? AUDCLNT_STREAMFLAGS_LOOPBACK : 0);
  const auto initialize_result =
      client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, format.get(), nullptr);
  if (FAILED(initialize_result) && requested_option_applied) {
    capability_fallback_reason =
        "requested stream option initialization failed (HRESULT " +
        hresult_hex(initialize_result) + ")";
    // Client properties cannot be undone after initialization is attempted.
    // A fresh activation is required before ordinary shared-mode capture.
    client = activate_client();
    format = get_mix_format(client.Get());
    check(client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, format.get(), nullptr),
          "IAudioClient::Initialize (capability fallback)");
    requested_option_applied = false;
  } else {
    check(initialize_result, "IAudioClient::Initialize");
  }

  RawAudioFormat raw_format;
  try {
    raw_format = describe_format(format.get());
  } catch (const std::exception& error) {
    if (!requested_option_applied) throw;
    capability_fallback_reason =
        std::string("requested stream option returned an unsupported format: ") +
        error.what();
    // A RAW request can succeed while exposing a device-native encoding the
    // frontend does not support. Validate before Start and retry with a fresh
    // ordinary shared-mode client, rather than failing after GetBuffer.
    client = activate_client();
    format = get_mix_format(client.Get());
    check(client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0,
                             format.get(), nullptr),
          "IAudioClient::Initialize (format fallback)");
    requested_option_applied = false;
    raw_format = describe_format(format.get());
  }

  if (requested_option_applied) {
    events("capture_capability",
           std::string(to_string(config.kind)) + ": requested stream option active");
  } else if (!capability_fallback_reason.empty()) {
    events("capture_capability_fallback",
           std::string(to_string(config.kind)) + ": " + capability_fallback_reason +
               "; ordinary shared capture active");
  }
  Handle audio_event(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!audio_event.get()) throw std::runtime_error("CreateEvent failed");
  check(client->SetEventHandle(audio_event.get()), "SetEventHandle");

  UINT32 endpoint_buffer_frames{};
  check(client->GetBufferSize(&endpoint_buffer_frames), "IAudioClient::GetBufferSize");
  const auto slot_bytes = checked_packet_bytes(endpoint_buffer_frames,
                                               raw_format.block_align);
  const auto pool_slots = std::clamp<std::size_t>(config.raw_pool_slots, 2, 4096);
  RawAudioBlockPool raw_pool(pool_slots, slot_bytes);

  REFERENCE_TIME default_device_period{};
  check(client->GetDevicePeriod(&default_device_period, nullptr),
        "IAudioClient::GetDevicePeriod");

  ComPtr<IAudioCaptureClient> capture;
  check(client->GetService(IID_PPV_ARGS(capture.GetAddressOf())), "GetService(IAudioCaptureClient)");
  // All format parsing and fixed-pool allocation is complete before this
  // thread is promoted to MMCSS and capture starts.
  MmcssRegistration mmcss(L"Audio");
  check(client->Start(), "IAudioClient::Start");

  constexpr std::uint64_t idle_packet_100ns = 100000;  // 10 ms
  // Allow two actual endpoint periods for a late event, while keeping the
  // total interval + confirmation watermark at least 30 ms (the default AEC
  // alignment budget). The synthetic packets themselves remain 10 ms long.
  const auto idle_confirmation_100ns = std::max<std::uint64_t>(
      idle_packet_100ns * 2,
      static_cast<std::uint64_t>(std::max<REFERENCE_TIME>(0, default_device_period)) * 2);
  // `next_idle_qpc` is the start of the first not-yet-accounted render
  // interval. Do not publish that interval until its entire 10 ms duration is
  // in the past; a normal WASAPI packet is timestamped at its first frame and
  // can arrive near the end of the interval.
  std::uint64_t next_idle_qpc = qpc_now_100ns();
  std::uint64_t next_idle_position{};
  std::uint64_t packet_sequence{};
  std::uint64_t pending_pool_drops{};
  bool pending_discontinuity{};
  const auto idle_frames = std::max<UINT32>(1, raw_format.sample_rate / 100);

  const auto emit_idle_loopback = [&] {
    if (config.kind != AudioStreamKind::loopback) return;
    const auto now = qpc_now_100ns();
    // Bound catch-up work after a debugger pause. The discontinuity flag tells
    // downstream stateful components not to bridge an unrepresented gap.
    std::uint32_t emitted{};
    while (now >= next_idle_qpc + idle_packet_100ns +
                      idle_confirmation_100ns &&
           emitted < 5) {
      RawCapturedPacket packet;
      packet.stream = config.kind;
      packet.format = raw_format;
      packet.frame_count = idle_frames;
      packet.qpc_100ns = next_idle_qpc;
      packet.arrival_qpc_100ns = now;
      packet.device_position = next_idle_position;
      packet.stream_epoch = stream_epoch;
      packet.sequence = ++packet_sequence;
      packet.silent = true;
      packet.synthetic = true;
      packet.discontinuity = pending_discontinuity ||
                             (emitted == 0 &&
                              now > next_idle_qpc + idle_packet_100ns * 5);
      packet.pool_drops_before = pending_pool_drops;
      const auto accepted = packets(std::move(packet));
      // The callback has observed the accumulated pool-drop count even if the
      // queue rejected this packet, so do not report it twice.
      pending_pool_drops = 0;
      pending_discontinuity = !accepted;
      next_idle_qpc += idle_packet_100ns;
      next_idle_position += idle_frames;
      ++emitted;
    }
    if (now > next_idle_qpc + idle_packet_100ns * 5) next_idle_qpc = now + idle_packet_100ns;
  };

  const bool monitor_default = config.follow_default &&
                               (config.device_id.empty() ||
                                config.device_id == "default");
  auto next_default_check = std::chrono::steady_clock::now() +
                            std::chrono::seconds(1);
  bool default_changed{};
  while (!stop.stop_requested()) {
    const auto wait = WaitForSingleObject(audio_event.get(), 10);
    if (wait == WAIT_TIMEOUT) {
      emit_idle_loopback();
    } else {
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
        CaptureBufferLease lease(capture.Get(), frames);
        RawCapturedPacket packet;
        packet.stream = config.kind;
        packet.format = raw_format;
        packet.frame_count = frames;
        packet.qpc_100ns = qpc;
        packet.arrival_qpc_100ns = qpc_now_100ns();
        packet.device_position = device_position;
        packet.stream_epoch = stream_epoch;
        packet.sequence = ++packet_sequence;
        packet.silent = (packet_flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
        packet.discontinuity = pending_discontinuity ||
                               (packet_flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) != 0;
        packet.timestamp_error =
            (packet_flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR) != 0;
        packet.pool_drops_before = pending_pool_drops;

        const auto packet_bytes = checked_packet_bytes(frames, raw_format.block_align);
        if (!packet.silent) {
          auto block = raw_pool.try_acquire();
          if (!block) {
            lease.release_checked();
            pending_discontinuity = true;
            ++pending_pool_drops;
            if (config.kind == AudioStreamKind::loopback) {
              next_idle_qpc = qpc + static_cast<std::uint64_t>(frames) *
                                        10000000ULL / raw_format.sample_rate;
              next_idle_position = device_position + frames;
            }
            check(capture->GetNextPacketSize(&next), "GetNextPacketSize");
            continue;
          }
          if (packet_bytes > block->capacity()) {
            lease.release_checked();
            throw std::runtime_error("WASAPI packet exceeds preallocated raw slot");
          }
          std::memcpy(block->data(), data, packet_bytes);
          if (!block->set_size(packet_bytes)) {
            lease.release_checked();
            throw std::runtime_error("failed to commit raw capture slot");
          }
          packet.bytes = std::move(*block);
        }
        lease.release_checked();
        if (config.kind == AudioStreamKind::loopback) {
          next_idle_qpc = qpc + static_cast<std::uint64_t>(frames) *
                                    10000000ULL / raw_format.sample_rate;
          next_idle_position = device_position + frames;
        }
        const auto accepted = packets(std::move(packet));
        pending_pool_drops = 0;
        pending_discontinuity = !accepted;
        check(capture->GetNextPacketSize(&next), "GetNextPacketSize");
      }
      // Never speculate about silence immediately after draining real data.
      // The WAIT_TIMEOUT path above is the only producer of idle packets.
    }

    // Polling once a second avoids allocations/COM calls inside the
    // GetBuffer/ReleaseBuffer critical section while making follow_default
    // effective even when the old endpoint remains alive.
    if (monitor_default && std::chrono::steady_clock::now() >= next_default_check) {
      next_default_check = std::chrono::steady_clock::now() +
                           std::chrono::seconds(1);
      const auto flow = config.kind == AudioStreamKind::loopback ? eRender : eCapture;
      ComPtr<IMMDevice> current_default;
      check(enumerator->GetDefaultAudioEndpoint(flow, eConsole, &current_default),
            "GetDefaultAudioEndpoint (follow default)");
      if (device_id(current_default.Get()) != id) {
        default_changed = true;
        break;
      }
    }
  }
  client->Stop();
  if (default_changed) {
    events("capture_default_changed",
           std::string(to_string(config.kind)) +
               ": reopening the new default endpoint");
  }
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
