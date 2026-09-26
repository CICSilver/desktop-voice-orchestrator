#include "dvo/windows_action_backend.h"

#ifndef _WIN32
#error windows_action_backend.cpp must only be built on Windows
#endif

#include <Windows.h>
#include <TlHelp32.h>
#include <endpointvolume.h>
#include <mmdeviceapi.h>
#include <roapi.h>
#include <winsqlite/winsqlite3.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.Control.h>
#include <winrt/base.h>

#include "dvo/netease.h"
#include "dvo/netease_cdp.h"
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

[[nodiscard]] const char* adapter_for(ActionType type) {
  switch (type) {
    case ActionType::master_volume_adjust: return "windows.endpoint_volume";
    case ActionType::media_like: return "netease.ncm_cli";
    case ActionType::media_play_daily:
    case ActionType::media_mode_order:
    case ActionType::media_mode_list_loop:
    case ActionType::media_mode_single_loop:
    case ActionType::media_mode_shuffle: return "netease.cdp";
    default: return "windows.gsmtc";
  }
}

[[nodiscard]] std::string lower_ascii(std::string value) {
  std::ranges::transform(value, value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

[[nodiscard]] std::filesystem::path environment_path(const wchar_t* name) {
  const auto size = GetEnvironmentVariableW(name, nullptr, 0);
  if (size == 0) return {};
  std::wstring value(size, L'\0');
  value.resize(GetEnvironmentVariableW(name, value.data(), size));
  return value;
}

// Quotes one argument for the MSVC/CommandLineToArgvW rules that node.exe
// uses. The process is started directly, never through cmd.exe, so song
// titles cannot be interpreted as shell syntax.
[[nodiscard]] std::wstring quote_argument(const std::wstring& value) {
  if (!value.empty() && value.find_first_of(L" \t\n\v\"") == std::wstring::npos) return value;
  std::wstring result{L'"'};
  for (auto it = value.begin();; ++it) {
    std::size_t backslashes{};
    while (it != value.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == value.end()) {
      result.append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      result.append(backslashes * 2 + 1, L'\\');
    } else {
      result.append(backslashes, L'\\');
    }
    result.push_back(*it);
  }
  result.push_back(L'"');
  return result;
}

struct ProcessOutput {
  DWORD exit_code{};
  std::string out;
  std::string err;
};

void read_pipe(HANDLE pipe, std::string& output) {
  constexpr std::size_t kMaxOutput = 8 * 1024 * 1024;
  std::array<char, 8192> buffer{};
  for (;;) {
    DWORD read{};
    if (!ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) ||
        read == 0) {
      return;
    }
    if (output.size() < kMaxOutput) output.append(buffer.data(), read);
  }
}

// Runs a console program without a window and collects its output. The
// process and anything it starts live in a job that is killed on timeout,
// cancellation or return.
[[nodiscard]] ProcessOutput run_process(const std::filesystem::path& executable,
                                        const std::vector<std::wstring>& arguments,
                                        std::chrono::steady_clock::time_point deadline,
                                        std::stop_token stop) {
  SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  const auto make_pipe = [&](winrt::handle& read_end, winrt::handle& write_end) {
    HANDLE read_handle{};
    HANDLE write_handle{};
    winrt::check_bool(CreatePipe(&read_handle, &write_handle, &inheritable, 0));
    read_end.attach(read_handle);
    write_end.attach(write_handle);
    winrt::check_bool(SetHandleInformation(read_handle, HANDLE_FLAG_INHERIT, 0));
  };
  winrt::handle out_read, out_write, err_read, err_write;
  make_pipe(out_read, out_write);
  make_pipe(err_read, err_write);
  winrt::handle input{CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  &inheritable, OPEN_EXISTING, 0, nullptr)};
  if (input.get() == INVALID_HANDLE_VALUE) {
    input.detach();
    winrt::throw_last_error();
  }

  winrt::handle job{CreateJobObjectW(nullptr, nullptr)};
  if (!job) winrt::throw_last_error();
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  winrt::check_bool(SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation,
                                            &limits, sizeof(limits)));

  // Only the three standard handles are inherited.
  SIZE_T attribute_size{};
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
  std::vector<std::byte> attribute_buffer(attribute_size);
  auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_buffer.data());
  winrt::check_bool(InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_size));
  std::array<HANDLE, 3> inherited{input.get(), out_write.get(), err_write.get()};
  const auto attribute_ok = UpdateProcThreadAttribute(
      attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
      inherited.size() * sizeof(HANDLE), nullptr, nullptr);
  if (!attribute_ok) {
    DeleteProcThreadAttributeList(attributes);
    winrt::throw_last_error();
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = out_write.get();
  startup.StartupInfo.hStdError = err_write.get();
  startup.lpAttributeList = attributes;

  auto command_line = quote_argument(executable.wstring());
  for (const auto& argument : arguments) command_line += L' ' + quote_argument(argument);
  PROCESS_INFORMATION information{};
  const auto created = CreateProcessW(
      executable.c_str(), command_line.data(), nullptr, nullptr, TRUE,
      CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED |
          EXTENDED_STARTUPINFO_PRESENT,
      nullptr, nullptr, &startup.StartupInfo, &information);
  const auto create_error = GetLastError();
  DeleteProcThreadAttributeList(attributes);
  if (!created) winrt::throw_hresult(HRESULT_FROM_WIN32(create_error));
  winrt::handle process{information.hProcess};
  winrt::handle thread{information.hThread};
  if (!AssignProcessToJobObject(job.get(), process.get())) {
    const auto error = GetLastError();
    TerminateProcess(process.get(), 1);
    winrt::throw_hresult(HRESULT_FROM_WIN32(error));
  }
  ResumeThread(thread.get());
  // The child holds its own copies; the pipes close when it and its children exit.
  out_write.close();
  err_write.close();
  input.close();

  ProcessOutput output;
  {
    // Declared after the handles they read, so they are joined first.
    std::jthread out_reader([&] { read_pipe(out_read.get(), output.out); });
    std::jthread err_reader([&] { read_pipe(err_read.get(), output.err); });
    for (;;) {
      if (WaitForSingleObject(process.get(), 20) == WAIT_OBJECT_0) break;
      if (stop.stop_requested()) {
        TerminateJobObject(job.get(), 1);
        throw ActionCancelledError{};
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        TerminateJobObject(job.get(), 1);
        throw ActionTimeoutError{};
      }
    }
    // Anything the program left running would otherwise keep the pipes open.
    TerminateJobObject(job.get(), 0);
  }
  winrt::check_bool(GetExitCodeProcess(process.get(), &output.exit_code));
  return output;
}

// NetEase's global "like" hotkey (设置 → 快捷键 → 喜欢歌曲, default Ctrl+Alt+L;
// global hotkeys must be enabled in the client).
constexpr UINT kLikeHotkeyModifiers = MOD_CONTROL | MOD_ALT;
constexpr WORD kLikeHotkeyKey = 'L';

// True when some program holds the like hotkey: probing it with RegisterHotKey
// fails with ERROR_HOTKEY_ALREADY_REGISTERED. A successful probe is undone at
// once, and the keystroke is then not sent, because it would reach whatever
// window has the focus.
[[nodiscard]] bool like_hotkey_taken() {
  constexpr int kProbeId = 0x0D0A;
  if (RegisterHotKey(nullptr, kProbeId, kLikeHotkeyModifiers | MOD_NOREPEAT, kLikeHotkeyKey)) {
    UnregisterHotKey(nullptr, kProbeId);
    return false;
  }
  return GetLastError() == ERROR_HOTKEY_ALREADY_REGISTERED;
}

[[nodiscard]] bool press_like_hotkey() {
  const std::array<WORD, 3> keys{VK_CONTROL, VK_MENU, kLikeHotkeyKey};
  std::array<INPUT, 6> inputs{};
  for (std::size_t i = 0; i < keys.size(); ++i) {
    inputs[i].type = INPUT_KEYBOARD;
    inputs[i].ki.wVk = keys[i];
    inputs[inputs.size() - 1 - i].type = INPUT_KEYBOARD;
    inputs[inputs.size() - 1 - i].ki.wVk = keys[i];
    inputs[inputs.size() - 1 - i].ki.dwFlags = KEYEVENTF_KEYUP;
  }
  return SendInput(static_cast<UINT>(inputs.size()), inputs.data(), sizeof(INPUT)) ==
         inputs.size();
}

void sleep_for_action(std::chrono::milliseconds duration,
                      std::chrono::steady_clock::time_point deadline, std::stop_token stop) {
  const auto until = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < until) {
    if (stop.stop_requested()) throw ActionCancelledError{};
    if (std::chrono::steady_clock::now() >= deadline) throw ActionTimeoutError{};
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

// The NetEase client's newest play-history entry, which is the current song.
[[nodiscard]] std::optional<NeteaseTrack> newest_history_track(
    const std::filesystem::path& database) {
  if (!std::filesystem::exists(database)) {
    throw std::runtime_error("NetEase client database not found: " + database.string());
  }
  sqlite3* db{};
  const auto path = database.u8string();
  const auto opened = sqlite3_open_v2(reinterpret_cast<const char*>(path.c_str()), &db,
                                      SQLITE_OPEN_READONLY, nullptr);
  const std::unique_ptr<sqlite3, void (*)(sqlite3*)> guard(
      db, [](sqlite3* value) { sqlite3_close(value); });
  if (opened != SQLITE_OK) {
    throw std::runtime_error(std::string("cannot open the NetEase client database: ") +
                             (db ? sqlite3_errmsg(db) : "out of memory"));
  }
  // The client writes to the database while it plays.
  sqlite3_busy_timeout(db, 1000);
  sqlite3_stmt* raw_statement{};
  if (sqlite3_prepare_v2(db, "SELECT jsonStr FROM historyTracks ORDER BY playtime DESC LIMIT 1",
                         -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(std::string("cannot read the NetEase play history: ") +
                             sqlite3_errmsg(db));
  }
  const std::unique_ptr<sqlite3_stmt, void (*)(sqlite3_stmt*)> statement(
      raw_statement, [](sqlite3_stmt* value) { sqlite3_finalize(value); });
  const auto step = sqlite3_step(statement.get());
  if (step == SQLITE_DONE) return std::nullopt;
  if (step != SQLITE_ROW) {
    throw std::runtime_error(std::string("cannot read the NetEase play history: ") +
                             sqlite3_errmsg(db));
  }
  const auto* text = reinterpret_cast<const char*>(sqlite3_column_text(statement.get(), 0));
  if (!text) return std::nullopt;
  return parse_netease_history_track(
      std::string_view(text, static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 0))));
}

[[nodiscard]] std::vector<DWORD> netease_processes() {
  std::vector<DWORD> result;
  winrt::handle snapshot{CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
  if (snapshot.get() == INVALID_HANDLE_VALUE) {
    snapshot.detach();
    return result;
  }
  PROCESSENTRY32W entry{};
  entry.dwSize = sizeof(entry);
  for (auto more = Process32FirstW(snapshot.get(), &entry); more;
       more = Process32NextW(snapshot.get(), &entry)) {
    if (_wcsicmp(entry.szExeFile, L"cloudmusic.exe") == 0) {
      result.push_back(entry.th32ProcessID);
    }
  }
  return result;
}

// cloudmusic.exe as configured, else the program registered for orpheus:// links.
[[nodiscard]] std::filesystem::path netease_executable(const NeteaseActionConfig& config) {
  if (!config.executable.empty()) return config.executable;
  DWORD size{};
  constexpr auto key = L"orpheus\\shell\\open\\command";
  if (RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr, RRF_RT_REG_SZ, nullptr, nullptr, &size) ==
      ERROR_SUCCESS) {
    std::wstring value(size / sizeof(wchar_t), L'\0');
    if (RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr, RRF_RT_REG_SZ, nullptr, value.data(),
                     &size) == ERROR_SUCCESS) {
      value.resize(wcslen(value.c_str()));
      std::filesystem::path path = value.starts_with(L'"')
                                       ? value.substr(1, value.find(L'"', 1) - 1)
                                       : value.substr(0, value.find(L' '));
      if (std::filesystem::exists(path)) return path;
    }
  }
  throw std::runtime_error("cloudmusic.exe was not found; set netease.executable");
}

// Minimized and not activated, so a game or video keeps the focus. The client
// is not tied to this process: it keeps running after voice_frontend exits.
void launch_netease(const std::filesystem::path& executable, std::uint16_t port) {
  auto command_line = quote_argument(executable.wstring()) +
                      L" --remote-debugging-address=127.0.0.1 --remote-debugging-port=" +
                      std::to_wstring(port);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW;
  startup.wShowWindow = SW_SHOWMINNOACTIVE;
  PROCESS_INFORMATION information{};
  const auto directory = executable.parent_path().wstring();
  DWORD flags = CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB;
  auto created = CreateProcessW(executable.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                                flags, nullptr, directory.c_str(), &startup, &information);
  if (!created && GetLastError() == ERROR_ACCESS_DENIED) {
    // This process sits in a job that forbids breakaway.
    flags &= ~static_cast<DWORD>(CREATE_BREAKAWAY_FROM_JOB);
    created = CreateProcessW(executable.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                             flags, nullptr, directory.c_str(), &startup, &information);
  }
  if (!created) winrt::throw_last_error();
  CloseHandle(information.hThread);
  CloseHandle(information.hProcess);
}

// Polls until the client's store answers through the channel.
void wait_for_netease_control(std::uint16_t port, std::chrono::steady_clock::time_point deadline,
                              std::stop_token stop) {
  const auto script = netease_ready_script();
  for (;;) {
    try {
      const auto attempt = std::min(deadline, std::chrono::steady_clock::now() +
                                                  std::chrono::seconds(3));
      (void)netease_cdp_evaluate(port, script, attempt, stop);
      return;
    } catch (const NeteaseCdpError& error) {
      const auto retry = error.kind() == NeteaseCdpError::Kind::unavailable ||
                         error.kind() == NeteaseCdpError::Kind::starting ||
                         error.kind() == NeteaseCdpError::Kind::script;
      if (!retry || std::chrono::steady_clock::now() >= deadline) throw;
    }
    sleep_for_action(std::chrono::milliseconds(500), deadline, stop);
  }
}

[[nodiscard]] const char* netease_mode_name(ActionType type) {
  switch (type) {
    case ActionType::media_mode_order: return "playOrder";
    case ActionType::media_mode_list_loop: return "playCycle";
    case ActionType::media_mode_single_loop: return "playOneCycle";
    case ActionType::media_mode_shuffle: return "playRandom";
    default: return "";
  }
}

// A cold client start takes several seconds before its page is ready.
constexpr std::chrono::seconds kNeteaseStartTimeout{30};

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
        case ActionType::media_next: return skip_next(stop);
        case ActionType::media_like: return like_current_song(stop);
        case ActionType::media_play_daily:
        case ActionType::media_mode_order:
        case ActionType::media_mode_list_loop:
        case ActionType::media_mode_single_loop:
        case ActionType::media_mode_shuffle: return control_netease(action.type, stop);
      }
    } catch (const ActionTimeoutError& error) {
      return failure(action.type, adapter_for(action.type), "action_timeout", error.what());
    } catch (const ActionCancelledError& error) {
      ActionResult result;
      result.type = action.type;
      result.status = ActionStatus::cancelled;
      result.adapter = adapter_for(action.type);
      result.error_code = "action_cancelled";
      result.message = error.what();
      return result;
    } catch (const winrt::hresult_error& error) {
      return failure(action.type, adapter_for(action.type), "windows_hresult",
                     hresult_message(error));
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

  [[nodiscard]] ActionResult skip_next(std::stop_token stop) const {
    constexpr auto type = ActionType::media_next;
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
    const auto playback = session.GetPlaybackInfo();
    const auto controls = playback ? playback.Controls() : nullptr;
    if (!controls || !controls.IsNextEnabled()) {
      result.status = ActionStatus::failed;
      result.error_code = "control_not_supported";
      result.message = "the current session does not enable Next";
      return result;
    }
    if (!wait_async(session.TrySkipNextAsync(), deadline, stop)) {
      result.status = ActionStatus::failed;
      result.error_code = "session_rejected";
      result.message = "the current media session rejected the Next request";
      return result;
    }
    result.status = ActionStatus::succeeded;
    result.message = "Next request accepted by the current media session";
    return result;
  }

  // Likes the NetEase client's current song through ncm-cli. The song is
  // identified by id, never by title alone: the client's newest play-history
  // entry must describe the song in its media session, and only the search
  // result with that exact id is liked. ncm-cli's like sets the state instead
  // of toggling it, and an already-liked song is left alone.
  [[nodiscard]] ActionResult like_current_song(std::stop_token stop) const {
    constexpr auto type = ActionType::media_like;
    constexpr auto adapter = "netease.ncm_cli";
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.netease.timeout_ms);
    auto manager = wait_async(
        media::GlobalSystemMediaTransportControlsSessionManager::RequestAsync(), deadline, stop);
    if (!manager) {
      return failure(type, adapter, "media_manager_unavailable",
                     "GSMTC session manager is unavailable");
    }
    // The NetEase session, even while another application owns the current one.
    media::GlobalSystemMediaTransportControlsSession session{nullptr};
    for (const auto& candidate : manager.GetSessions()) {
      if (lower_ascii(winrt::to_string(candidate.SourceAppUserModelId())).find("cloudmusic") !=
          std::string::npos) {
        session = candidate;
        break;
      }
    }
    if (!session) {
      return failure(type, adapter, "netease_not_running",
                     "the NetEase Cloud Music client has no media session");
    }
    const auto properties = wait_async(session.TryGetMediaPropertiesAsync(), deadline, stop);
    const auto title = properties ? winrt::to_string(properties.Title()) : std::string{};
    const auto artist = properties ? winrt::to_string(properties.Artist()) : std::string{};
    if (title.empty()) {
      return failure(type, adapter, "no_current_song",
                     "the NetEase media session does not name a song");
    }

    const auto track = newest_history_track(client_data() / "Library" / "webdb.dat");
    if (!track || !netease_track_matches_media(*track, title, artist)) {
      return failure(type, adapter, "song_unidentified",
                     "the client's newest play-history entry is not the playing song (" +
                         title + " - " + artist + ")");
    }

    ActionResult result;
    result.type = type;
    result.adapter = adapter;
    result.target_id = track->id;
    const auto search = [&](const std::string& keyword) {
      const auto response = run_ncm_cli(
          {L"search", L"song", L"--keyword", wide(keyword), L"--limit", L"30"}, deadline, stop);
      if (!response.ok()) throw std::runtime_error(response.error);
      return find_ncm_search_hit(response.body, track->id);
    };
    auto keyword = title + " " + track->artists.front();
    auto hit = search(keyword);
    if (!hit) {
      keyword = title;
      hit = search(keyword);
    }
    if (!hit) {
      return failure(type, adapter, "song_not_found",
                     "ncm-cli search did not return song " + track->id + " (" + title + ")");
    }
    if (hit->liked) {
      result.status = ActionStatus::noop;
      result.verified = true;
      result.message = "already liked: " + title + " - " + artist;
      return result;
    }

    // A like made only through ncm-cli is not shown by the client until it
    // refreshes, so the client's own hotkey goes first: the client then likes
    // the song itself and lights its heart. The hotkey toggles the client's
    // local idea of the state, so it is pressed at most once, only for a song
    // the server reports as not liked, and the result is checked; if the
    // client's state was stale, ncm-cli likes the song.
    if (like_hotkey_taken() && press_like_hotkey()) {
      for (int check = 0; check < 2; ++check) {
        sleep_for_action(std::chrono::milliseconds(500), deadline, stop);
        const auto after = search(keyword);
        if (after && after->liked) {
          result.status = ActionStatus::succeeded;
          result.verified = true;
          result.message = "liked in the client: " + title + " - " + artist;
          return result;
        }
      }
    }
    const auto liked = run_ncm_cli({L"song", L"like", L"--songId", wide(hit->encrypted_id)},
                                   deadline, stop);
    if (!liked.ok()) {
      result.status = ActionStatus::failed;
      result.error_code = "ncm_cli_error";
      result.message = liked.error;
      return result;
    }
    result.status = ActionStatus::succeeded;
    result.message = "liked through ncm-cli (the client shows it after a refresh): " + title +
                     " - " + artist;
    return result;
  }

  // Daily recommendations and play modes, through the client's DevTools
  // channel. A client that is not running is started with the channel; one
  // running without it is left alone, since restarting it would cut off
  // whatever it is playing.
  [[nodiscard]] ActionResult control_netease(ActionType type, std::stop_token stop) const {
    constexpr auto adapter = "netease.cdp";
    const auto port = config_.netease.cdp_port;
    const auto script = type == ActionType::media_play_daily
                            ? netease_play_daily_script()
                            : netease_play_mode_script(netease_mode_name(type));
    const auto evaluate = [&](std::chrono::steady_clock::time_point deadline) {
      return netease_cdp_evaluate(port, script, deadline, stop);
    };
    ActionResult result;
    result.type = type;
    result.adapter = adapter;
    nlohmann::json value;
    try {
      try {
        value = evaluate(std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(config_.netease.timeout_ms));
      } catch (const NeteaseCdpError& error) {
        const bool running = !netease_processes().empty();
        if (error.kind() == NeteaseCdpError::Kind::unavailable && running) {
          return failure(type, adapter, "netease_control_unavailable",
                         "the NetEase client is running without its control channel; close it "
                         "and start it with `voice_frontend netease-start`");
        }
        if (error.kind() != NeteaseCdpError::Kind::unavailable &&
            error.kind() != NeteaseCdpError::Kind::starting) {
          throw;
        }
        const auto deadline = std::chrono::steady_clock::now() + kNeteaseStartTimeout;
        if (!running) launch_netease(netease_executable(config_.netease), port);
        wait_for_netease_control(port, deadline, stop);
        value = evaluate(deadline);
      }
    } catch (const NeteaseCdpError& error) {
      switch (error.kind()) {
        case NeteaseCdpError::Kind::timeout: throw ActionTimeoutError{};
        case NeteaseCdpError::Kind::cancelled: throw ActionCancelledError{};
        default:
          return failure(type, adapter, "netease_control_failed", error.what());
      }
    }

    if (!value.is_object() || !value.value("ok", false)) {
      return failure(type, adapter, "netease_control_failed",
                     "the NetEase client did not reach the requested state: " + value.dump());
    }
    result.verified = true;
    if (type == ActionType::media_play_daily) {
      result.status = ActionStatus::succeeded;
      result.target_id = value.value("id", "");
      result.message = "playing daily recommendations (" +
                       std::to_string(value.value("queue", 0)) + " songs): " +
                       value.value("name", "");
      return result;
    }
    result.target_id = value.value("mode", "");
    if (!value.value("changed", false)) {
      result.status = ActionStatus::noop;
      result.message = "play mode is already " + result.target_id;
      return result;
    }
    result.status = ActionStatus::succeeded;
    result.message = "play mode " + value.value("before", "") + " -> " + result.target_id;
    return result;
  }

  [[nodiscard]] NcmResponse run_ncm_cli(std::vector<std::wstring> arguments,
                                        std::chrono::steady_clock::time_point deadline,
                                        std::stop_token stop) const {
    arguments.insert(arguments.begin(), ncm_cli_script().wstring());
    arguments.insert(arguments.end(), {L"--output", L"json"});
    const auto output = run_process(node_executable(), arguments, deadline, stop);
    auto response = parse_ncm_response(output.out);
    if (!response.ok() && !output.err.empty()) {
      response.error += " | " + output.err.substr(0, 400);
    }
    return response;
  }

  [[nodiscard]] std::filesystem::path node_executable() const {
    const auto& configured = config_.netease.node;
    if (!configured.empty() && configured.has_parent_path()) return configured;
    const auto name = configured.empty() ? std::wstring{L"node"} : configured.wstring();
    std::wstring found(MAX_PATH, L'\0');
    const auto length = SearchPathW(nullptr, name.c_str(), L".exe",
                                    static_cast<DWORD>(found.size()), found.data(), nullptr);
    if (length == 0 || length >= found.size()) {
      throw std::runtime_error("node.exe was not found on PATH; set netease.node");
    }
    found.resize(length);
    return found;
  }

  [[nodiscard]] std::filesystem::path ncm_cli_script() const {
    auto script = config_.netease.ncm_cli;
    if (script.empty()) {
      script = environment_path(L"APPDATA") / "npm" / "node_modules" / "@music163" /
               "ncm-cli" / "dist" / "index.js";
    }
    if (!std::filesystem::exists(script)) {
      throw std::runtime_error("ncm-cli was not found at " + script.string() +
                               "; install @music163/ncm-cli or set netease.ncm_cli");
    }
    return script;
  }

  [[nodiscard]] std::filesystem::path client_data() const {
    if (!config_.netease.client_data.empty()) return config_.netease.client_data;
    return environment_path(L"LOCALAPPDATA") / "NetEase" / "CloudMusic";
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

bool start_netease_with_control(const NeteaseActionConfig& config, bool restart) {
  const std::stop_source never;
  const auto deadline = std::chrono::steady_clock::now() + kNeteaseStartTimeout;
  try {
    (void)netease_cdp_evaluate(config.cdp_port, netease_ready_script(),
                               std::chrono::steady_clock::now() + std::chrono::seconds(3),
                               never.get_token());
    return true;
  } catch (const NeteaseCdpError& error) {
    if (error.kind() == NeteaseCdpError::Kind::starting) {
      wait_for_netease_control(config.cdp_port, deadline, never.get_token());
      return true;
    }
  }
  if (const auto running = netease_processes(); !running.empty()) {
    if (!restart) return false;
    for (const auto pid : running) {
      if (const auto process = OpenProcess(PROCESS_TERMINATE, FALSE, pid)) {
        TerminateProcess(process, 0);
        CloseHandle(process);
      }
    }
    for (int wait = 0; wait < 100 && !netease_processes().empty(); ++wait) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  launch_netease(netease_executable(config), config.cdp_port);
  wait_for_netease_control(config.cdp_port, deadline, never.get_token());
  return true;
}

std::shared_ptr<IActionBackend> create_windows_action_backend(WindowsActionConfig config) {
  return std::make_shared<WindowsActionBackend>(std::move(config));
}

}  // namespace dvo
