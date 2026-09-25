#include <Windows.h>
#include <shellapi.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/config.h"
#include "dvo/evaluation.h"
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
  std::cout << "voice_frontend <list-devices|live|replay|benchmark|evaluate> [session] "
               "[--no-browser] [--web-port=<port>]\n"
               "  benchmark <session> [--events=<file.ndjson>]\n"
               "  evaluate <labeled session | directory of sessions> [--out=<report.json>]\n"
               "           [--since=<YYYYMMDD-HHMMSS>]   (only sessions recorded from then on)\n"
               "  parse [--wake=<wake word>]   (one recognized text per stdin line)\n";
}

std::optional<std::string> option_value(int argc, char** argv, std::string_view name) {
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument.starts_with(name) && argument.size() > name.size() &&
        argument[name.size()] == '=') {
      return std::string(argument.substr(name.size() + 1));
    }
  }
  return std::nullopt;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << text;
  if (!output) throw std::runtime_error("cannot write " + path.string());
}

// Training recordings (labels "purpose": "train") must never be scored, or the
// evaluation would measure data the models were fitted to.
bool training_session(const std::filesystem::path& session) {
  try {
    std::ifstream stream(session / "labels.json", std::ios::binary);
    const auto labels = nlohmann::json::parse(stream);
    return labels.is_object() && labels.value("purpose", "") == "train";
  } catch (const std::exception&) {
    return false;  // load_evaluation_labels() reports malformed files
  }
}

std::vector<std::filesystem::path> labeled_sessions(const std::filesystem::path& target) {
  std::vector<std::filesystem::path> sessions;
  if (std::filesystem::is_regular_file(target / "labels.json")) {
    if (!training_session(target)) sessions.push_back(target);
    return sessions;
  }
  if (!std::filesystem::is_directory(target)) {
    throw std::invalid_argument(target.string() + " is not a directory");
  }
  for (const auto& entry : std::filesystem::directory_iterator(target)) {
    if (entry.is_directory() && std::filesystem::is_regular_file(entry.path() / "labels.json") &&
        !training_session(entry.path())) {
      sessions.push_back(entry.path());
    }
  }
  std::ranges::sort(sessions);
  return sessions;
}

// argv is in the ANSI code page on Windows; non-ASCII options such as a
// Chinese wake word must come from the UTF-16 command line instead.
std::optional<std::string> utf8_option_value(std::string_view name) {
  int count{};
  wchar_t** wide = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!wide) return std::nullopt;
  std::optional<std::string> result;
  for (int index = 2; index < count && !result; ++index) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, nullptr, 0, nullptr, nullptr);
    std::string argument(size > 0 ? static_cast<std::size_t>(size - 1) : 0, '\0');
    if (size > 1) {
      WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, argument.data(), size, nullptr, nullptr);
    }
    if (argument.starts_with(name) && argument.size() > name.size() && argument[name.size()] == '=') {
      result = argument.substr(name.size() + 1);
    }
  }
  LocalFree(wide);
  return result;
}

// Runs recognized texts through the configured command parser, dry-run, so
// alternative recognizers or parser changes can be scored offline.
int run_parse(const dvo::AppConfig& config) {
  const auto wake = utf8_option_value("--wake").value_or("");
  const dvo::CommandParser parser(
      static_cast<int>(config.commands.default_volume_step_percent),
      static_cast<int>(config.commands.max_spoken_volume_step_percent),
      dvo::command_grammar(config.commands));
  std::string line;
  std::uint64_t index{};
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    dvo::CommandParseContext context;
    context.runtime_session_id = "parse-cli";
    context.utterance_id = "parse-" + std::to_string(++index);
    context.origin = wake.empty() ? dvo::UtteranceOrigin::followup : dvo::UtteranceOrigin::keyword;
    context.wake_word = wake;
    context.source = dvo::ExecutionSource::replay;
    context.execution_mode = dvo::ExecutionMode::dry_run;
    const auto parsed = parser.parse(line, context);
    nlohmann::json result{{"text", line}, {"ok", parsed.ok()}};
    if (parsed.ok()) {
      auto actions = nlohmann::json::array();
      for (const auto& action : parsed.plan->actions) {
        actions.push_back({{"type", dvo::to_string(action.type)},
                           {"volume_delta_percent",
                            action.volume_delta_percent
                                ? nlohmann::json(*action.volume_delta_percent)
                                : nlohmann::json(nullptr)}});
      }
      result["normalized_text"] = parsed.plan->normalized_text;
      result["actions"] = std::move(actions);
    } else if (parsed.error) {
      result["error"] = {{"code", parsed.error->code}, {"message", parsed.error->message}};
    }
    std::cout << result.dump() << '\n';
  }
  return 0;
}

int run_evaluation(int argc, char** argv, const dvo::AppConfig& config,
                   const dvo::ConfigStore& store) {
  if (argc < 3) throw std::invalid_argument("evaluate requires a session or directory path");
  auto sessions = labeled_sessions(argv[2]);
  // Session names start with their creation time, so a batch recorded later
  // can be scored apart from the tuning batch without moving files around.
  if (const auto since = option_value(argc, argv, "--since")) {
    std::erase_if(sessions, [&](const std::filesystem::path& session) {
      return session.filename().string() < *since;
    });
  }
  if (sessions.empty()) {
    std::cerr << "no evaluation session with labels.json under " << argv[2]
              << " (training sessions are skipped)\n";
    return 1;
  }
  std::vector<nlohmann::json> reports;
  for (const auto& session : sessions) {
    std::cerr << "evaluating " << session.filename().string() << " ..." << std::endl;
    const auto labels = dvo::load_evaluation_labels(session / "labels.json");
    dvo::VoiceFrontendRuntime runtime(config, store);
    // The "after AEC" reference must come from this replay's AEC, not the
    // processed.wav recorded under whatever configuration was live then.
    const auto processed = std::filesystem::temp_directory_path() /
                           ("dvo-processed-" + session.filename().string() + ".wav");
    auto run = runtime.run_benchmark_detailed(session, processed);
    const auto& utterances = run.metrics["utterances"];
    const auto reference =
        dvo::compute_session_reference(session, labels, config, utterances, processed);
    std::error_code removed;
    std::filesystem::remove(processed, removed);
    auto report = dvo::evaluate_labeled_session(labels, utterances, reference);
    report["session"]["name"] = session.filename().string();
    report["session"]["path"] = session.string();
    report["benchmark_wait"] = run.metrics["benchmark_wait"];
    report["config"] = store.to_public_json(config);
    write_text(session / "evaluation.json", report.dump(2) + "\n");
    reports.push_back(std::move(report));
  }
  const auto aggregate = dvo::aggregate_evaluations(reports);
  std::cout << dvo::format_evaluation_summary(reports, aggregate);
  if (const auto out = option_value(argc, argv, "--out")) {
    write_text(*out, nlohmann::json{{"aggregate", aggregate}, {"sessions", reports}}.dump(2) + "\n");
    std::cout << "\nreport: " << *out << "\n";
  }
  return 0;
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
    for (int index = 2; index < argc; ++index) {
      const std::string_view argument = argv[index];
      constexpr std::string_view prefix = "--web-port=";
      if (argument.starts_with(prefix)) {
        const auto port = std::stoi(std::string(argument.substr(prefix.size())));
        if (port <= 0 || port > 65535) throw std::invalid_argument("web port must be in [1, 65535]");
        config.web.port = static_cast<std::uint16_t>(port);
      }
    }
    dvo::VoiceFrontendRuntime runtime(config, store);
    const bool no_browser = std::find_if(argv + 2, argv + argc,
        [](const char* value) { return std::string_view(value) == "--no-browser"; }) != argv + argc;

    if (mode == "benchmark") {
      if (argc < 3) throw std::invalid_argument("benchmark requires a session path");
      const auto run = runtime.run_benchmark_detailed(argv[2]);
      if (const auto events = option_value(argc, argv, "--events")) {
        std::string lines;
        for (const auto& event : run.events) lines += event.dump() + '\n';
        write_text(*events, lines);
      }
      std::cout << run.metrics.dump(2) << '\n';
      return 0;
    }
    if (mode == "evaluate") return run_evaluation(argc, argv, config, store);
    if (mode == "parse") return run_parse(config);
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
