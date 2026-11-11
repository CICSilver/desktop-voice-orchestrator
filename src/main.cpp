#include <Windows.h>
#include <shellapi.h>

#include <array>
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

#include "dvo/app_catalog.h"
#include "dvo/config.h"
#include "dvo/evaluation.h"
#include "dvo/runtime.h"
#include "dvo/wasapi_capture.h"
#include "dvo/windows_action_backend.h"

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
  std::cout << "voice_frontend <list-devices|list-apps|live|replay|benchmark|evaluate> [session] "
               "[--no-browser] [--web-port=<port>]\n"
               "  list-apps   (prints the merged local application catalog as JSON)\n"
               "  benchmark <session> [--events=<file.ndjson>]\n"
               "  evaluate <labeled session | directory of sessions> [--out=<report.json>]\n"
               "           [--since=<YYYYMMDD-HHMMSS>]   (only sessions recorded from then on)\n"
               "  parse [--wake=<wake word>] [--mode=rules|hybrid|router]\n"
               "        [--router-url=<url>] [--router-model=<model>]\n"
               "        (one recognized text per stdin line)\n"
               "  act <next|like|daily|order|list-loop|single-loop|shuffle|lyrics-on|lyrics-off>\n"
               "  act open-app <id_or_alias>\n"
               "  act search-play <query> [--playlist] [--artist=<artist>]\n"
               "      (runs one action for real, bypassing speech)\n"
               "  netease-start [--restart]   (starts NetEase Cloud Music with its control channel;\n"
               "      --restart closes a client running without it first)\n";
}

dvo::NeteaseActionConfig netease_action_config(const dvo::AppConfig& config) {
  return {config.netease.ncm_cli,    config.netease.node,     config.netease.client_data,
          config.netease.timeout_ms, config.netease.cdp_port, config.netease.executable};
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

std::vector<std::string> utf8_command_line_args() {
  int count{};
  wchar_t** wide = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!wide) return {};
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, nullptr, 0, nullptr, nullptr);
    std::string argument(size > 0 ? static_cast<std::size_t>(size - 1) : 0, '\0');
    if (size > 1) {
      WideCharToMultiByte(CP_UTF8, 0, wide[index], -1, argument.data(), size, nullptr, nullptr);
    }
    args.push_back(std::move(argument));
  }
  LocalFree(wide);
  return args;
}

// argv is in the ANSI code page on Windows; non-ASCII options such as a
// Chinese wake word must come from the UTF-16 command line instead.
std::optional<std::string> utf8_option_value(std::string_view name) {
  const auto args = utf8_command_line_args();
  for (std::size_t index = 2; index < args.size(); ++index) {
    const std::string_view argument = args[index];
    if (argument.starts_with(name) && argument.size() > name.size() &&
        argument[name.size()] == '=') {
      return std::string(argument.substr(name.size() + 1));
    }
  }
  return std::nullopt;
}

// Executes one action through the live Windows backend, for checking an
// adapter (for example the NetEase like) without speaking.
int run_action(const dvo::AppConfig& config, std::string_view name) {
  std::shared_ptr<const dvo::AppCatalog> catalog;
  dvo::PlannedAction action;
  action.action_id = "act-cli";
  action.sequence = 1;
  if (name == "open-app") {
    catalog = std::make_shared<const dvo::AppCatalog>(
        dvo::AppCatalog::from_config(config.apps));
    const auto args = utf8_command_line_args();
    std::string query;
    for (std::size_t index = 3; index < args.size(); ++index) {
      const std::string_view arg = args[index];
      if (arg.starts_with("--")) {
        throw std::invalid_argument("unknown option for act open-app: " +
                                    std::string(arg));
      }
      if (!query.empty()) query.push_back(' ');
      query.append(arg);
    }
    if (query.empty()) {
      throw std::invalid_argument("act open-app requires <id_or_alias>");
    }
    action.type = dvo::ActionType::app_open;
    if (const auto* by_id = catalog->find_by_id(query)) {
      action.app_id = by_id->id;
      action.app_display_name = by_id->display_name;
    } else if (const auto* by_alias = catalog->match_alias(query)) {
      action.app_id = by_alias->id;
      action.app_display_name = by_alias->display_name;
    } else {
      action.app_id = query;
    }
  } else if (name == "search-play") {
    const auto args = utf8_command_line_args();
    dvo::MediaSearchType search_type = dvo::MediaSearchType::song;
    std::optional<std::string> artist;
    std::string query;
    for (std::size_t index = 3; index < args.size(); ++index) {
      const std::string_view arg = args[index];
      if (arg == "--playlist") {
        search_type = dvo::MediaSearchType::playlist;
      } else if (arg == "--song") {
        search_type = dvo::MediaSearchType::song;
      } else if (arg.starts_with("--artist=")) {
        auto value = std::string(arg.substr(std::string_view{"--artist="}.size()));
        if (!value.empty()) artist = std::move(value);
      } else if (arg.starts_with("--")) {
        throw std::invalid_argument("unknown option for act search-play: " +
                                    std::string(arg));
      } else {
        if (!query.empty()) query.push_back(' ');
        query.append(arg);
      }
    }
    if (query.empty()) {
      throw std::invalid_argument(
          "act search-play requires a <query> [--playlist] [--artist=<artist>]");
    }
    action.type = dvo::ActionType::media_search_play;
    action.search_query = std::move(query);
    action.search_type = search_type;
    action.search_artist = std::move(artist);
  } else {
    const std::array<std::pair<std::string_view, dvo::ActionType>, 9> actions{{
        {"next", dvo::ActionType::media_next},
        {"like", dvo::ActionType::media_like},
        {"daily", dvo::ActionType::media_play_daily},
        {"order", dvo::ActionType::media_mode_order},
        {"list-loop", dvo::ActionType::media_mode_list_loop},
        {"single-loop", dvo::ActionType::media_mode_single_loop},
        {"shuffle", dvo::ActionType::media_mode_shuffle},
        {"lyrics-on", dvo::ActionType::media_lyrics_show},
        {"lyrics-off", dvo::ActionType::media_lyrics_hide},
    }};
    const auto found = std::ranges::find(
        actions, name, &decltype(actions)::value_type::first);
    if (found == actions.end()) {
      throw std::invalid_argument("unknown action for act");
    }
    action.type = found->second;
  }
  const auto backend = dvo::create_windows_action_backend(
      {config.audio.loopback_device, config.commands.action_timeout_ms,
       netease_action_config(config), catalog});
  const auto started = std::chrono::steady_clock::now();
  const auto result = backend->execute(action, std::stop_source{}.get_token());
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  nlohmann::json out{{"type", dvo::to_string(result.type)},
                     {"status", dvo::to_string(result.status)},
                     {"adapter", result.adapter},
                     {"target_id", result.target_id},
                     {"error_code", result.error_code},
                     {"message", result.message},
                     {"elapsed_ms", elapsed.count()}};
  if (result.type == dvo::ActionType::app_open) {
    out["app_id"] = result.requested_app_id
                        ? nlohmann::json(*result.requested_app_id)
                        : (action.app_id ? nlohmann::json(*action.app_id)
                                         : nlohmann::json(nullptr));
  }
  if (result.type == dvo::ActionType::media_search_play) {
    out["search_query"] = result.requested_search_query
                              ? nlohmann::json(*result.requested_search_query)
                              : (action.search_query
                                     ? nlohmann::json(*action.search_query)
                                     : nlohmann::json(nullptr));
    const auto effective_type = result.requested_search_type.has_value()
                                    ? result.requested_search_type
                                    : action.search_type;
    out["search_type"] = effective_type
                             ? nlohmann::json(dvo::to_string(*effective_type))
                             : nlohmann::json(nullptr);
    const auto effective_artist = result.requested_search_artist.has_value()
                                      ? result.requested_search_artist
                                      : action.search_artist;
    out["search_artist"] = effective_artist ? nlohmann::json(*effective_artist)
                                            : nlohmann::json(nullptr);
  }
  std::cout << out.dump(2, ' ', false, nlohmann::json::error_handler_t::replace)
            << "\n";
  return result.status == dvo::ActionStatus::failed ? 1 : 0;
}

// Runs recognized texts through the configured command parser, dry-run, so
// alternative recognizers or parser changes can be scored offline.
int run_parse(const dvo::AppConfig& config) {
  dvo::AppConfig effective = config;
  const auto wake = utf8_option_value("--wake").value_or("");
  const auto router_backend_opt = utf8_option_value("--router-backend");
  if (router_backend_opt.has_value()) {
    if (*router_backend_opt != "none" &&
        *router_backend_opt != "openai_compatible") {
      throw std::invalid_argument(
          "parse --router-backend must be none or openai_compatible");
    }
    effective.commands.router_backend = *router_backend_opt;
  }
  if (const auto mode_opt = utf8_option_value("--mode")) {
    const auto parsed_mode = dvo::parse_command_parser_mode(*mode_opt);
    if (!parsed_mode.has_value()) {
      throw std::invalid_argument(
          "parse --mode must be rules, hybrid, or router");
    }
    effective.commands.parser_mode = *mode_opt;
    if (*parsed_mode != dvo::CommandParserMode::rules &&
        !router_backend_opt.has_value() &&
        effective.commands.router_backend == "none") {
      effective.commands.router_backend = "openai_compatible";
    }
  }
  if (const auto router_url = utf8_option_value("--router-url")) {
    effective.commands.router_base_url = *router_url;
  }
  if (const auto router_model = utf8_option_value("--router-model")) {
    effective.commands.router_model = *router_model;
  }
  if (const auto router_key_env = utf8_option_value("--router-api-key-env")) {
    effective.commands.router_api_key_env = *router_key_env;
  }
  if (const auto router_timeout = utf8_option_value("--router-timeout-ms")) {
    const auto timeout = std::stoi(*router_timeout);
    if (timeout < 200 || timeout > 15000) {
      throw std::invalid_argument(
          "parse --router-timeout-ms must be in [200, 15000]");
    }
    effective.commands.router_timeout_ms = static_cast<std::uint32_t>(timeout);
  }
  const auto catalog = std::make_shared<const dvo::AppCatalog>(
      dvo::AppCatalog::from_config(effective.apps));
  const auto parser = dvo::create_command_parser(effective, catalog);
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
    const auto parsed = parser->parse(line, context);
    nlohmann::json result{{"text", line}, {"ok", parsed.ok()}};
    if (parsed.ok()) {
      auto actions = nlohmann::json::array();
      for (const auto& action : parsed.plan->actions) {
        actions.push_back({{"type", dvo::to_string(action.type)},
                           {"volume_delta_percent",
                            action.volume_delta_percent
                                ? nlohmann::json(*action.volume_delta_percent)
                                : nlohmann::json(nullptr)},
                           {"search_query",
                            action.search_query
                                ? nlohmann::json(*action.search_query)
                                : nlohmann::json(nullptr)},
                           {"search_type",
                            action.search_type
                                ? nlohmann::json(dvo::to_string(*action.search_type))
                                : nlohmann::json(nullptr)},
                           {"search_artist",
                            action.search_artist
                                ? nlohmann::json(*action.search_artist)
                                : nlohmann::json(nullptr)},
                           {"app_id",
                            action.app_id
                                ? nlohmann::json(*action.app_id)
                                : nlohmann::json(nullptr)},
                           {"app_display_name",
                            action.app_display_name
                                ? nlohmann::json(*action.app_display_name)
                                : nlohmann::json(nullptr)}});
      }
      result["parser_version"] = parsed.plan->parser_version;
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
    if (mode == "list-apps") {
      const auto catalog = dvo::AppCatalog::from_config(config.apps);
      std::cout << catalog.to_json().dump(
                       2, ' ', false, nlohmann::json::error_handler_t::replace)
                << '\n';
      return 0;
    }
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
    if (mode == "act") {
      if (argc < 3) throw std::invalid_argument("act requires an action name");
      return run_action(config, argv[2]);
    }
    if (mode == "netease-start") {
      const bool restart = argc > 2 && std::string_view(argv[2]) == "--restart";
      if (dvo::start_netease_with_control(netease_action_config(config), restart)) {
        std::cout << "NetEase Cloud Music is running with its control channel on 127.0.0.1:"
                  << config.netease.cdp_port << "\n";
        return 0;
      }
      std::cout << "NetEase Cloud Music is running without its control channel; "
                   "run netease-start --restart to restart it with the channel\n";
      return 1;
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
