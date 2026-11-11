#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <objbase.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

#include "dvo/app_catalog.h"
#include "dvo/windows_action_backend.h"

namespace {

struct ScopedTempDir {
  ScopedTempDir() {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("dvo-app-catalog-test-" + std::to_string(suffix));
    std::filesystem::create_directories(path);
  }
  ~ScopedTempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  std::filesystem::path path;
};

void touch_stub_file(const std::filesystem::path& file) {
  std::filesystem::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << "stub";
}

void create_shortcut_lnk(const std::filesystem::path& lnk_path,
                         const std::filesystem::path& target_path) {
  std::filesystem::create_directories(lnk_path.parent_path());
  const HRESULT init_hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  const bool uninit = (init_hr == S_OK || init_hr == S_FALSE);

  IShellLinkW* shell_link = nullptr;
  HRESULT hr = CoCreateInstance(
      CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER,
      IID_IShellLinkW, reinterpret_cast<void**>(&shell_link));
  REQUIRE(SUCCEEDED(hr));
  REQUIRE(SUCCEEDED(shell_link->SetPath(target_path.wstring().c_str())));

  IPersistFile* persist_file = nullptr;
  hr = shell_link->QueryInterface(
      IID_IPersistFile, reinterpret_cast<void**>(&persist_file));
  REQUIRE(SUCCEEDED(hr));
  REQUIRE(SUCCEEDED(persist_file->Save(lnk_path.wstring().c_str(), TRUE)));

  persist_file->Release();
  shell_link->Release();
  if (uninit) CoUninitialize();
}

}  // namespace

TEST_CASE("AppCatalog merges configured, QuickStart, and system apps with priority and noise filtering") {
  ScopedTempDir temp;
  const auto quickstart_dir = temp.path / "QuickStart";
  const auto system_programs_dir = temp.path / "SystemPrograms";
  const auto program_bins_dir = temp.path / "ProgramBins";
  const auto explicit_dir = temp.path / "Explicit";

  // QuickStart valid stubs + garbage entries that must be filtered out.
  touch_stub_file(quickstart_dir / "Steam.exe");
  touch_stub_file(quickstart_dir / "Clash Verge.exe");
  touch_stub_file(quickstart_dir / "MyCustomTool.bat");
  touch_stub_file(quickstart_dir / "powershell.exe");
  touch_stub_file(quickstart_dir / "Uninstall Steam.exe");
  touch_stub_file(quickstart_dir / "卸载助手.cmd");
  touch_stub_file(quickstart_dir / "README.bat");

  // Secondary Start Menu Programs (depth 1 and depth 2 .lnk shortcuts) + duplicate/garbage.
  const auto discord_bin = program_bins_dir / "Discord" / "Discord.exe";
  const auto discord_updater_bin = program_bins_dir / "Discord" / "Discord Updater.exe";
  const auto notion_bin = program_bins_dir / "Notion" / "Notion.exe";
  const auto custom_tool_sys_bin = program_bins_dir / "Tools" / "MyCustomTool.exe";
  touch_stub_file(discord_bin);
  touch_stub_file(discord_updater_bin);
  touch_stub_file(notion_bin);
  touch_stub_file(custom_tool_sys_bin);

  create_shortcut_lnk(system_programs_dir / "Discord" / "Discord.lnk", discord_bin);
  create_shortcut_lnk(system_programs_dir / "Discord" / "Discord Updater.lnk", discord_updater_bin);
  create_shortcut_lnk(system_programs_dir / "Notion.lnk", notion_bin);
  create_shortcut_lnk(system_programs_dir / "Notion" / "Notion Help.lnk", notion_bin);
  create_shortcut_lnk(system_programs_dir / "Tools" / "MyCustomTool.lnk", custom_tool_sys_bin);

  // Explicit configured target must beat QuickStart's同-named file.
  const auto explicit_ps = explicit_dir / "pwsh-custom.exe";
  touch_stub_file(explicit_ps);

  const auto qs_utf8 = quickstart_dir.u8string();
  const std::string qs_str(qs_utf8.begin(), qs_utf8.end());
  REQUIRE(_putenv_s("DVO_TEST_QUICKSTART_DIR", qs_str.c_str()) == 0);
  CHECK(dvo::expand_windows_env_vars("%DVO_TEST_QUICKSTART_DIR%") == qs_str);

  dvo::AppsConfig config;
  config.quickstart_dirs = {"%DVO_TEST_QUICKSTART_DIR%"};
  config.scan_system_apps = true;
  config.entries = {
      dvo::AppEntryConfig{
          .id = "powershell",
          .display_name = "PowerShell",
          .target = explicit_ps,
          .arguments = {"-NoLogo"},
          .aliases = {"powershell", "终端", "命令行"},
          .process_names = {"pwsh.exe", "powershell.exe"},
      },
      dvo::AppEntryConfig{
          .id = "steam",
          .display_name = "Steam",
          .target = {},  // Auto-bound from QuickStart!
          .arguments = {},
          .aliases = {"steam", "蒸汽平台", "死盯", "斯蒂姆"},
          .process_names = {"steam.exe"},
      },
      dvo::AppEntryConfig{
          .id = "discord",
          .display_name = "Discord",
          .target = {},  // Auto-bound from secondary system index!
          .arguments = {},
          .aliases = {"discord", "迪斯科", "迪斯扣"},
          .process_names = {"Discord.exe"},
      },
      dvo::AppEntryConfig{
          .id = "qq",
          .display_name = "QQ",
          .target = {},  // Remains unbound when not installed in test dirs.
          .arguments = {},
          .aliases = {"qq", "腾讯qq", "扣扣"},
          .process_names = {"QQ.exe"},
      },
  };

  dvo::AppCatalogBuildOptions options;
  options.system_program_dirs_override =
      std::vector<std::filesystem::path>{system_programs_dir};
  options.query_registry_app_paths = false;

  const auto catalog = dvo::AppCatalog::from_config(config, options);

  // 1. Explicit target is preserved even when QuickStart has powershell.exe.
  const auto* ps = catalog.find_by_id("powershell");
  REQUIRE(ps != nullptr);
  CHECK(ps->source == dvo::AppEntrySource::configured);
  CHECK(std::filesystem::equivalent(ps->target, explicit_ps));

  // 2. Empty configured targets auto-bind from QuickStart first, then system_index.
  const auto* steam = catalog.find_by_id("steam");
  REQUIRE(steam != nullptr);
  CHECK(steam->source == dvo::AppEntrySource::configured);
  REQUIRE_FALSE(steam->target.empty());
  CHECK(std::filesystem::equivalent(steam->target, quickstart_dir / "Steam.exe"));

  const auto* discord = catalog.find_by_id("discord");
  REQUIRE(discord != nullptr);
  CHECK(discord->source == dvo::AppEntrySource::configured);
  REQUIRE_FALSE(discord->target.empty());
  CHECK(std::filesystem::equivalent(discord->target, discord_bin));

  // Uninstalled preset stays registered with empty target so Activate can still
  // check running windows and Launch reports app_not_installed cleanly.
  const auto* qq = catalog.find_by_id("qq");
  REQUIRE(qq != nullptr);
  CHECK(qq->target.empty());

  // 3. QuickStart entries outrank system_index duplicates.
  const auto* custom_tool = catalog.match_alias("MyCustomTool");
  REQUIRE(custom_tool != nullptr);
  CHECK(custom_tool->source == dvo::AppEntrySource::quickstart);
  CHECK(std::filesystem::equivalent(
      custom_tool->target, quickstart_dir / "MyCustomTool.bat"));

  // 4. Secondary system_index discovers non-duplicate apps.
  const auto* notion = catalog.match_alias("notion");
  REQUIRE(notion != nullptr);
  CHECK(notion->source == dvo::AppEntrySource::system_index);
  CHECK(std::filesystem::equivalent(notion->target, notion_bin));

  // 5. Noise entries (uninstall / 卸载 / readme / help / updater) are filtered out.
  for (const auto& entry : catalog.entries()) {
    CAPTURE(entry.id, entry.display_name);
    CHECK_FALSE(dvo::is_noise_app_name(entry.id));
    CHECK_FALSE(dvo::is_noise_app_name(entry.display_name));
  }
  CHECK(catalog.match_alias("Uninstall Steam") == nullptr);
  CHECK(catalog.match_alias("卸载助手") == nullptr);
  CHECK(catalog.match_alias("README") == nullptr);
  CHECK(catalog.match_alias("Discord Updater") == nullptr);
  CHECK(catalog.match_alias("Notion Help") == nullptr);

  // 6. Case/space-insensitive and Chinese homophone match_alias queries.
  CHECK(catalog.match_alias("  S T E A M ") == steam);
  CHECK(catalog.match_alias("死盯") == steam);
  CHECK(catalog.match_alias("斯蒂姆") == steam);
  CHECK(catalog.match_alias("迪斯科") == discord);
  CHECK(catalog.match_alias("扣 扣") == qq);
  const auto* clash_verge = catalog.match_alias("clashverge");
  REQUIRE(clash_verge != nullptr);
  CHECK(clash_verge == catalog.match_alias("Clash Verge"));

  // 7. JSON export contains expected catalog metadata.
  const auto exported = catalog.to_json();
  REQUIRE(exported.is_array());
  CHECK(exported.size() >= 6);

  (void)_putenv_s("DVO_TEST_QUICKSTART_DIR", "");
}

TEST_CASE("WindowsActionBackend reports unknown_app and app_not_installed cleanly") {
  auto catalog = std::make_shared<dvo::AppCatalog>(std::vector<dvo::AppCatalogEntry>{
      dvo::AppCatalogEntry{
          .id = "nonexistent_test_app",
          .display_name = "Nonexistent Test App",
          .source = dvo::AppEntrySource::configured,
          .target = {},
          .arguments = {},
          .aliases = {"不存在的测试应用"},
          .process_names = {"definitely_not_running_dvo_test_9999.exe"},
      },
  });

  dvo::WindowsActionConfig backend_config;
  backend_config.app_catalog = catalog;
  auto backend = dvo::create_windows_action_backend(std::move(backend_config));

  dvo::PlannedAction unknown;
  unknown.action_id = "act-unknown";
  unknown.sequence = 1;
  unknown.type = dvo::ActionType::app_open;
  unknown.app_id = "unregistered_app";
  const auto unknown_res = backend->execute(unknown, std::stop_token{});
  CHECK(unknown_res.status == dvo::ActionStatus::failed);
  CHECK(unknown_res.error_code == "unknown_app");

  dvo::PlannedAction missing;
  missing.action_id = "act-missing";
  missing.sequence = 1;
  missing.type = dvo::ActionType::app_open;
  missing.app_id = "nonexistent_test_app";
  const auto missing_res = backend->execute(missing, std::stop_token{});
  CHECK(missing_res.status == dvo::ActionStatus::failed);
  CHECK(missing_res.error_code == "app_not_installed");
}
