#include <Windows.h>
#include <shellapi.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "dvo/config.h"
#include "dvo/runtime.h"
#include "dvo/wasapi_capture.h"

namespace {
std::atomic<bool> stop_requested{};

BOOL WINAPI console_handler(DWORD event) {
  if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT || event == CTRL_BREAK_EVENT) {
    stop_requested.store(true, std::memory_order_release);
    return TRUE;
  }
  return FALSE;
}

std::filesystem::path project_root() {
  return std::filesystem::path(DVO_PROJECT_ROOT);
}

void print_usage() {
  std::cout << "voice_frontend <list-devices|live|replay|benchmark> [session] [--no-browser]\n";
}
}  // namespace

int main(int argc, char** argv) {
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCtrlHandler(console_handler, TRUE);
  try {
    if (argc < 2) { print_usage(); return 2; }
    const std::string mode = argv[1];
    if (mode == "list-devices") {
      nlohmann::json result = nlohmann::json::array();
      for (const auto& device : dvo::WasapiCapture::list_devices()) {
        result.push_back({{"kind", dvo::to_string(device.kind)}, {"id", device.id},
                          {"name", device.name}, {"default", device.is_default}});
      }
      std::cout << result.dump(2) << '\n';
      return 0;
    }

    const auto root = project_root();
    dvo::ConfigStore store(root / "config/default.toml", root / "config/local.toml", root);
    auto config = store.load();
    dvo::VoiceFrontendRuntime runtime(config, store);
    const bool no_browser = std::find_if(argv + 2, argv + argc,
        [](const char* value) { return std::string_view(value) == "--no-browser"; }) != argv + argc;

    if (mode == "benchmark") {
      if (argc < 3) throw std::invalid_argument("benchmark requires a session path");
      std::cout << runtime.run_benchmark(argv[2]).dump(2) << '\n';
      return 0;
    }
    if (mode == "replay") {
      if (argc < 3) throw std::invalid_argument("replay requires a session path");
      runtime.start_replay(argv[2]);
    } else if (mode == "live") {
      runtime.start_live();
    } else {
      print_usage();
      return 2;
    }

    std::cout << "Debug UI: " << runtime.debug_url() << std::endl;
    if (!no_browser && !runtime.debug_url().empty()) {
      const auto url = runtime.debug_url();
      ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
    while (!stop_requested.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    runtime.stop();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }
}
