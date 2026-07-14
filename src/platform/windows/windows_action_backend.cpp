#include "dvo/windows_action_backend.h"

#ifndef _WIN32
#error windows_action_backend.cpp must only be built on Windows
#endif

#include <Windows.h>
#include <endpointvolume.h>
#include <mmdeviceapi.h>
#include <roapi.h>
#include <wrl/client.h>

#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Media.Control.h>
#include <winrt/base.h>

#include "dvo/volume_math.h"

namespace dvo {
namespace {

using Microsoft::WRL::ComPtr;
namespace foundation = winrt::Windows::Foundation;
namespace media = winrt::Windows::Media::Control;

constexpr GUID kVolumeEventContext{
    0x73b5ca28, 0x4fc8, 0x44db, {0x8d, 0xc2, 0x8c, 0xdf, 0x55, 0x92, 0x2f, 0x0b}};

class ThreadApartment {
 public:
  ThreadApartment() {
    const auto result = RoInitialize(RO_INIT_MULTITHREADED);
    if (FAILED(result)) winrt::check_hresult(result);
    initialized_ = true;
  }
  ~ThreadApartment() {
    if (initialized_) RoUninitialize();
  }
  ThreadApartment(const ThreadApartment&) = delete;
  ThreadApartment& operator=(const ThreadApartment&) = delete;

 private:
  bool initialized_{};
};

void ensure_apartment() {
  static thread_local ThreadApartment apartment;
  (void)apartment;
}

[[nodiscard]] std::wstring wide(std::string_view value) {
  if (value.empty()) return {};
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) winrt::check_hresult(HRESULT_FROM_WIN32(GetLastError()));
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), size) != size) {
    winrt::check_hresult(HRESULT_FROM_WIN32(GetLastError()));
  }
  return result;
}

[[nodiscard]] std::string hresult_message(const winrt::hresult_error& error) {
  std::ostringstream output;
  output << winrt::to_string(error.message()) << " (HRESULT 0x" << std::hex << std::uppercase
         << static_cast<std::uint32_t>(error.code()) << ')';
  return output.str();
}

[[nodiscard]] ActionResult failure(ActionType type, std::string adapter,
                                   std::string code, std::string message) {
  ActionResult result;
  result.type = type;
  result.status = ActionStatus::failed;
  result.adapter = std::move(adapter);
  result.error_code = std::move(code);
  result.message = std::move(message);
  return result;
}

class ActionTimeoutError final : public std::runtime_error {
 public:
  ActionTimeoutError() : std::runtime_error("Windows asynchronous action exceeded its deadline") {}
};

class ActionCancelledError final : public std::runtime_error {
 public:
  ActionCancelledError() : std::runtime_error("Windows asynchronous action was cancelled") {}
};

template <typename AsyncOperation>
[[nodiscard]] auto wait_async(AsyncOperation operation,
                              std::chrono::steady_clock::time_point deadline,
                              std::stop_token stop) {
  for (;;) {
    const auto status = operation.Status();
    if (status == foundation::AsyncStatus::Completed) return operation.GetResults();
    if (status == foundation::AsyncStatus::Canceled) throw ActionCancelledError{};
    if (status == foundation::AsyncStatus::Error) {
      winrt::check_hresult(operation.ErrorCode());
    }

    if (stop.stop_requested()) {
      try {
        operation.Cancel();
      } catch (...) {
        // Cancellation is best effort; the executor still reports cancellation.
      }
      throw ActionCancelledError{};
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      try {
        operation.Cancel();
      } catch (...) {
        // Preserve the deadline result even if the WinRT object rejects Cancel().
      }
      throw ActionTimeoutError{};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

class WindowsActionBackend final : public IActionBackend {
 public:
  explicit WindowsActionBackend(WindowsActionConfig config) : config_(std::move(config)) {}

  ActionResult execute(const PlannedAction& action, std::stop_token stop) override {
    if (stop.stop_requested()) {
      ActionResult result;
      result.type = action.type;
      result.status = ActionStatus::cancelled;
      result.adapter = "windows";
      result.error_code = "executor_stopped";
      result.message = "execution was cancelled before the Windows API call";
      return result;
    }
    try {
      ensure_apartment();
      switch (action.type) {
        case ActionType::media_play: return control_media(true, stop);
        case ActionType::media_pause: return control_media(false, stop);
        case ActionType::master_volume_adjust:
          if (!action.volume_delta_percent || *action.volume_delta_percent == 0 ||
              *action.volume_delta_percent < -20 || *action.volume_delta_percent > 20) {
            return failure(action.type, "windows.endpoint_volume", "invalid_volume_delta",
                           "volume delta must be a non-zero value in [-20, 20]");
          }
          return adjust_volume(*action.volume_delta_percent);
      }
    } catch (const ActionTimeoutError& error) {
      return failure(action.type, "windows.gsmtc", "action_timeout", error.what());
    } catch (const ActionCancelledError& error) {
      ActionResult result;
      result.type = action.type;
      result.status = ActionStatus::cancelled;
      result.adapter = "windows.gsmtc";
      result.error_code = "action_cancelled";
      result.message = error.what();
      return result;
    } catch (const winrt::hresult_error& error) {
      return failure(action.type,
                     action.type == ActionType::master_volume_adjust
                         ? "windows.endpoint_volume"
                         : "windows.gsmtc",
                     "windows_hresult", hresult_message(error));
    } catch (const std::exception& error) {
      return failure(action.type, "windows", "windows_adapter_error", error.what());
    }
    return failure(action.type, "windows", "unsupported_action", "unsupported action type");
  }

 private:
  [[nodiscard]] ActionResult control_media(bool play, std::stop_token stop) const {
    const auto type = play ? ActionType::media_play : ActionType::media_pause;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.action_timeout_ms);
    auto manager = wait_async(
        media::GlobalSystemMediaTransportControlsSessionManager::RequestAsync(), deadline, stop);
    if (!manager) {
      return failure(type, "windows.gsmtc", "media_manager_unavailable",
                     "GSMTC session manager is unavailable");
    }
    auto session = manager.GetCurrentSession();
    if (!session) {
      return failure(type, "windows.gsmtc", "no_media_session",
                     "Windows has no current controllable media session");
    }

    ActionResult result;
    result.type = type;
    result.adapter = "windows.gsmtc";
    result.target_id = winrt::to_string(session.SourceAppUserModelId());

    auto playback = session.GetPlaybackInfo();
    if (!playback) {
      return failure(type, "windows.gsmtc", "playback_info_unavailable",
                     "the current media session did not provide playback information");
    }
    const auto status = playback.PlaybackStatus();
    if ((play && status == media::GlobalSystemMediaTransportControlsSessionPlaybackStatus::Playing) ||
        (!play && status == media::GlobalSystemMediaTransportControlsSessionPlaybackStatus::Paused)) {
      result.status = ActionStatus::noop;
      result.verified = true;
      result.message = play ? "media session is already playing" : "media session is already paused";
      return result;
    }

    const auto controls = playback.Controls();
    if (!controls || (play && !controls.IsPlayEnabled()) || (!play && !controls.IsPauseEnabled())) {
      result.status = ActionStatus::failed;
      result.error_code = "control_not_supported";
      result.message = play ? "the current session does not enable Play"
                            : "the current session does not enable Pause";
      return result;
    }

    if (stop.stop_requested()) throw ActionCancelledError{};
    if (std::chrono::steady_clock::now() >= deadline) throw ActionTimeoutError{};
    const bool accepted = play ? wait_async(session.TryPlayAsync(), deadline, stop)
                               : wait_async(session.TryPauseAsync(), deadline, stop);
    if (!accepted) {
      result.status = ActionStatus::failed;
      result.error_code = "session_rejected";
      result.message = "the current media session rejected the transport request";
      return result;
    }
    result.status = ActionStatus::succeeded;
    result.message = play ? "Play request accepted by the current media session"
                          : "Pause request accepted by the current media session";
    // TryPlayAsync/TryPauseAsync reports whether the request was accepted. The
    // target application may publish its new playback status asynchronously.
    result.verified = false;
    return result;
  }

  [[nodiscard]] ComPtr<IMMDevice> render_device() const {
    ComPtr<IMMDeviceEnumerator> enumerator;
    winrt::check_hresult(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                         IID_PPV_ARGS(&enumerator)));
    ComPtr<IMMDevice> device;
    if (config_.render_device_id.empty() || config_.render_device_id == "default") {
      winrt::check_hresult(
          enumerator->GetDefaultAudioEndpoint(eRender, eConsole, device.GetAddressOf()));
    } else {
      const auto id = wide(config_.render_device_id);
      winrt::check_hresult(enumerator->GetDevice(id.c_str(), device.GetAddressOf()));
    }
    return device;
  }

  [[nodiscard]] ActionResult adjust_volume(int delta_percent) const {
    auto device = render_device();
    LPWSTR raw_id{};
    winrt::check_hresult(device->GetId(&raw_id));
    const std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> owned_id(raw_id, CoTaskMemFree);

    ComPtr<IAudioEndpointVolume> volume;
    winrt::check_hresult(device->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr,
                                         reinterpret_cast<void**>(volume.GetAddressOf())));

    float before{};
    BOOL mute_before{};
    winrt::check_hresult(volume->GetMasterVolumeLevelScalar(&before));
    winrt::check_hresult(volume->GetMute(&mute_before));
    const auto adjustment = calculate_volume_adjustment(before, delta_percent);

    ActionResult result;
    result.type = ActionType::master_volume_adjust;
    result.adapter = "windows.endpoint_volume";
    result.target_id = raw_id ? winrt::to_string(winrt::hstring{raw_id}) : std::string{};
    result.requested_volume_delta_percent = delta_percent;
    result.volume_before = adjustment.current;
    result.volume_requested = adjustment.requested;
    result.clamped = adjustment.clamped;
    result.mute_before = mute_before != FALSE;

    if (adjustment.noop) {
      result.status = ActionStatus::noop;
      result.volume_after = adjustment.current;
      result.mute_after = mute_before != FALSE;
      result.verified = true;
      result.message = delta_percent > 0 ? "volume is already at the upper boundary"
                                         : "volume is already at the lower boundary";
      return result;
    }

    winrt::check_hresult(volume->SetMasterVolumeLevelScalar(
        static_cast<float>(adjustment.target), &kVolumeEventContext));

    float after{};
    BOOL mute_after{};
    winrt::check_hresult(volume->GetMasterVolumeLevelScalar(&after));
    winrt::check_hresult(volume->GetMute(&mute_after));
    result.volume_after = after;
    result.mute_after = mute_after != FALSE;
    result.verified = std::abs(static_cast<double>(after) - adjustment.target) <= 0.011 &&
                      mute_before == mute_after;
    if (!result.verified) {
      result.status = ActionStatus::failed;
      result.error_code = "volume_verification_failed";
      result.message = "endpoint volume read-back or mute-state verification failed";
      return result;
    }
    result.status = ActionStatus::succeeded;
    result.message = "endpoint master volume changed and was verified by read-back";
    return result;
  }

  WindowsActionConfig config_;
};

}  // namespace

std::shared_ptr<IActionBackend> create_windows_action_backend(WindowsActionConfig config) {
  return std::make_shared<WindowsActionBackend>(std::move(config));
}

}  // namespace dvo
