#include "dvo/app_catalog.h"

#include <windows.h>
#include <objbase.h>
#include <shlguid.h>
#include <shobjidl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "dvo/command.h"

namespace dvo {
namespace {

std::wstring utf8_to_wide(std::string_view text) {
  if (text.empty()) return {};
  const int required = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  if (required <= 0) return {};
  std::wstring out(static_cast<std::size_t>(required), L'\0');
  MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), required);
  return out;
}

std::string wide_to_utf8(std::wstring_view text) {
  if (text.empty()) return {};
  const int required = WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) return {};
  std::string out(static_cast<std::size_t>(required), '\0');
  WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), required, nullptr, nullptr);
  return out;
}

std::string path_to_utf8_string(const std::filesystem::path& path) {
  if (path.empty()) return {};
  return wide_to_utf8(path.wstring());
}

std::filesystem::path utf8_to_path(std::string_view text) {
  if (text.empty()) return {};
  return std::filesystem::path(utf8_to_wide(text));
}

std::string ascii_lower(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const unsigned char ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      out.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else {
      out.push_back(static_cast<char>(ch));
    }
  }
  return out;
}

// Case-insensitive and whitespace-insensitive key (strips ASCII whitespace and U+3000).
std::string normalize_alias_key(std::string_view text, bool strip_separators) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if (ch <= 0x20 || ch == 0x7F) continue;
    // Full-width space U+3000: E3 80 80
    if (ch == 0xE3 && i + 2 < text.size() &&
        static_cast<unsigned char>(text[i + 1]) == 0x80 &&
        static_cast<unsigned char>(text[i + 2]) == 0x80) {
      i += 2;
      continue;
    }
    if (strip_separators && (ch == '-' || ch == '_' || ch == '.')) {
      continue;
    }
    if (ch >= 'A' && ch <= 'Z') {
      out.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else {
      out.push_back(static_cast<char>(ch));
    }
  }
  return out;
}

bool is_supported_scan_extension(std::string_view ext_lower) {
  return ext_lower == ".lnk" || ext_lower == ".exe" ||
         ext_lower == ".bat" || ext_lower == ".cmd";
}

std::string sanitize_slug(std::string_view raw) {
  std::string slug;
  slug.reserve(raw.size());
  bool prev_dash = false;
  for (const unsigned char ch : raw) {
    if (ch >= 'A' && ch <= 'Z') {
      slug.push_back(static_cast<char>(ch - 'A' + 'a'));
      prev_dash = false;
    } else if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_') {
      slug.push_back(static_cast<char>(ch));
      prev_dash = false;
    } else if (ch == ' ' || ch == '-' || ch == '.') {
      if (!slug.empty() && !prev_dash) {
        slug.push_back('-');
        prev_dash = true;
      }
    }
    if (slug.size() >= kMaxAppIdBytes) break;
  }
  while (!slug.empty() && (slug.back() == '-' || slug.back() == '.')) {
    slug.pop_back();
  }
  while (!slug.empty() && (slug.front() == '-' || slug.front() == '.')) {
    slug.erase(slug.begin());
  }
  return slug;
}

std::string fnv1a_short_id(std::string_view seed) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char ch : seed) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  std::ostringstream oss;
  oss << "app-" << std::hex << std::setw(8) << std::setfill('0')
      << static_cast<std::uint32_t>(hash & 0xFFFFFFFFULL);
  return oss.str();
}

struct ScopedComInit {
  HRESULT hr{E_FAIL};
  ScopedComInit() : hr(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)) {}
  ~ScopedComInit() {
    if (hr == S_OK || hr == S_FALSE) {
      CoUninitialize();
    }
  }
};

struct ShortcutResolution {
  std::filesystem::path target_path;
  std::wstring arguments;
};

std::optional<ShortcutResolution> resolve_lnk_shortcut(const std::filesystem::path& lnk_path) {
  IShellLinkW* shell_link = nullptr;
  HRESULT hr = CoCreateInstance(
      CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&shell_link));
  if (FAILED(hr) || !shell_link) return std::nullopt;

  IPersistFile* persist_file = nullptr;
  hr = shell_link->QueryInterface(IID_PPV_ARGS(&persist_file));
  if (FAILED(hr) || !persist_file) {
    shell_link->Release();
    return std::nullopt;
  }

  hr = persist_file->Load(lnk_path.c_str(), STGM_READ);
  if (FAILED(hr)) {
    persist_file->Release();
    shell_link->Release();
    return std::nullopt;
  }

  std::array<wchar_t, MAX_PATH * 4> target_buf{};
  WIN32_FIND_DATAW find_data{};
  hr = shell_link->GetPath(
      target_buf.data(), static_cast<int>(target_buf.size()), &find_data, SLGP_RAWPATH);

  std::array<wchar_t, 2048> args_buf{};
  const HRESULT args_hr =
      shell_link->GetArguments(args_buf.data(), static_cast<int>(args_buf.size()));

  persist_file->Release();
  shell_link->Release();

  if (FAILED(hr) || target_buf[0] == L'\0') {
    return std::nullopt;
  }

  ShortcutResolution out;
  out.target_path = expand_windows_path(std::filesystem::path(target_buf.data()));
  if (SUCCEEDED(args_hr) && args_buf[0] != L'\0') {
    out.arguments = args_buf.data();
  }
  return out;
}

struct DiscoveredApp {
  std::string raw_name;
  AppEntrySource source{AppEntrySource::quickstart};
  std::filesystem::path launch_target;
  std::filesystem::path resolved_exe;
  std::vector<std::string> process_names;
};

std::optional<DiscoveredApp> inspect_file_candidate(
    const std::filesystem::path& file_path,
    AppEntrySource source) {
  const std::string ext = ascii_lower(path_to_utf8_string(file_path.extension()));
  if (!is_supported_scan_extension(ext)) return std::nullopt;

  const std::string stem = path_to_utf8_string(file_path.stem());
  if (stem.empty() || is_noise_app_name(stem)) return std::nullopt;

  DiscoveredApp candidate;
  candidate.raw_name = stem;
  candidate.source = source;
  candidate.launch_target = file_path;

  if (ext == ".lnk") {
    if (const auto resolved = resolve_lnk_shortcut(file_path)) {
      const std::string resolved_stem = path_to_utf8_string(resolved->target_path.stem());
      if (is_noise_app_name(resolved_stem)) return std::nullopt;

      std::error_code ec;
      const bool target_exists = !resolved->target_path.empty() &&
                                 std::filesystem::exists(resolved->target_path, ec);
      if (target_exists) {
        candidate.resolved_exe = resolved->target_path;
        const std::string resolved_ext =
            ascii_lower(path_to_utf8_string(resolved->target_path.extension()));
        if (resolved_ext == ".exe") {
          candidate.process_names.push_back(
              path_to_utf8_string(resolved->target_path.filename()));
        }
        // Keep the .lnk itself when shortcut arguments are present (e.g. Squirrel
        // Update.exe --processStart Discord.exe), otherwise bind directly to the executable.
        if (resolved->arguments.empty()) {
          candidate.launch_target = resolved->target_path;
        }
      }
    }
  } else if (ext == ".exe") {
    candidate.resolved_exe = file_path;
    candidate.process_names.push_back(path_to_utf8_string(file_path.filename()));
  }

  return candidate;
}

void scan_directory_files(
    const std::filesystem::path& dir,
    std::size_t max_depth,
    bool lnk_only,
    AppEntrySource source,
    std::vector<DiscoveredApp>& out) {
  std::error_code ec;
  if (dir.empty() || !std::filesystem::exists(dir, ec) ||
      !std::filesystem::is_directory(dir, ec)) {
    return;
  }

  std::vector<std::filesystem::path> found_files;
  const auto collect_in_dir = [&](const std::filesystem::path& current,
                                  std::size_t depth,
                                  auto& self) -> void {
    std::error_code iter_ec;
    std::vector<std::filesystem::path> children;
    for (std::filesystem::directory_iterator it(
             current, std::filesystem::directory_options::skip_permission_denied, iter_ec);
         !iter_ec && it != std::filesystem::directory_iterator();
         it.increment(iter_ec)) {
      children.push_back(it->path());
    }
    std::sort(children.begin(), children.end());
    for (const auto& child : children) {
      std::error_code status_ec;
      if (std::filesystem::is_directory(child, status_ec)) {
        if (depth + 1 < max_depth) {
          self(child, depth + 1, self);
        }
      } else if (std::filesystem::is_regular_file(child, status_ec)) {
        if (lnk_only) {
          const std::string ext = ascii_lower(path_to_utf8_string(child.extension()));
          if (ext != ".lnk") continue;
        }
        found_files.push_back(child);
      }
    }
  };

  collect_in_dir(dir, 0, collect_in_dir);
  for (const auto& file : found_files) {
    if (auto item = inspect_file_candidate(file, source)) {
      out.push_back(std::move(*item));
    }
  }
}

std::wstring strip_wrapping_quotes(std::wstring_view value) {
  while (!value.empty() && (value.front() == L' ' || value.front() == L'\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == L' ' || value.back() == L'\t')) {
    value.remove_suffix(1);
  }
  if (value.size() >= 2 && value.front() == L'"' && value.back() == L'"') {
    value = value.substr(1, value.size() - 2);
  }
  return std::wstring(value);
}

void scan_registry_app_paths_root(HKEY root, std::vector<DiscoveredApp>& out) {
  constexpr const wchar_t* kAppPathsSubkey =
      L"Software\\Microsoft\\Windows\\CurrentVersion\\App Paths";
  HKEY app_paths_key = nullptr;
  if (RegOpenKeyExW(root, kAppPathsSubkey, 0, KEY_READ, &app_paths_key) != ERROR_SUCCESS) {
    return;
  }

  std::vector<std::wstring> subkeys;
  for (DWORD index = 0;; ++index) {
    std::array<wchar_t, 260> name_buf{};
    DWORD name_len = static_cast<DWORD>(name_buf.size());
    const LONG status = RegEnumKeyExW(
        app_paths_key, index, name_buf.data(), &name_len, nullptr, nullptr, nullptr, nullptr);
    if (status == ERROR_NO_MORE_ITEMS) break;
    if (status == ERROR_SUCCESS && name_len > 0) {
      subkeys.emplace_back(name_buf.data(), name_len);
    }
  }
  std::sort(subkeys.begin(), subkeys.end());

  for (const auto& subkey : subkeys) {
    HKEY entry_key = nullptr;
    if (RegOpenKeyExW(app_paths_key, subkey.c_str(), 0, KEY_READ, &entry_key) != ERROR_SUCCESS) {
      continue;
    }
    DWORD value_type = 0;
    std::array<wchar_t, MAX_PATH * 4> value_buf{};
    DWORD value_bytes = static_cast<DWORD>(value_buf.size() * sizeof(wchar_t));
    const LONG query_status = RegQueryValueExW(
        entry_key, nullptr, nullptr, &value_type,
        reinterpret_cast<LPBYTE>(value_buf.data()), &value_bytes);
    RegCloseKey(entry_key);

    if (query_status != ERROR_SUCCESS ||
        (value_type != REG_SZ && value_type != REG_EXPAND_SZ) ||
        value_buf[0] == L'\0') {
      continue;
    }

    const std::wstring cleaned = strip_wrapping_quotes(value_buf.data());
    if (cleaned.empty()) continue;
    const std::filesystem::path resolved = expand_windows_path(std::filesystem::path(cleaned));
    std::error_code ec;
    if (resolved.empty() || !std::filesystem::exists(resolved, ec) ||
        !std::filesystem::is_regular_file(resolved, ec)) {
      continue;
    }

    const std::filesystem::path key_path(subkey);
    const std::string stem = path_to_utf8_string(key_path.stem());
    const std::string target_stem = path_to_utf8_string(resolved.stem());
    if (stem.empty() || is_noise_app_name(stem) || is_noise_app_name(target_stem)) {
      continue;
    }

    DiscoveredApp item;
    item.raw_name = stem;
    item.source = AppEntrySource::system_index;
    item.launch_target = resolved;
    item.resolved_exe = resolved;
    item.process_names.push_back(path_to_utf8_string(resolved.filename()));
    const std::string subkey_utf8 = wide_to_utf8(subkey);
    if (ascii_lower(subkey_utf8) != ascii_lower(item.process_names.front()) &&
        ascii_lower(path_to_utf8_string(key_path.extension())) == ".exe") {
      item.process_names.push_back(subkey_utf8);
    }
    out.push_back(std::move(item));
  }

  RegCloseKey(app_paths_key);
}

bool candidate_matches_configured(
    const AppCatalogEntry& entry,
    const DiscoveredApp& candidate) {
  const std::string cand_compact = normalize_alias_key(candidate.raw_name, true);
  if (!cand_compact.empty()) {
    if (cand_compact == normalize_alias_key(entry.id, true) ||
        cand_compact == normalize_alias_key(entry.display_name, true)) {
      return true;
    }
    for (const auto& alias : entry.aliases) {
      if (cand_compact == normalize_alias_key(alias, true)) return true;
    }
  }

  if (!candidate.resolved_exe.empty()) {
    const std::string exe_stem_compact =
        normalize_alias_key(path_to_utf8_string(candidate.resolved_exe.stem()), true);
    if (!exe_stem_compact.empty() &&
        (exe_stem_compact == normalize_alias_key(entry.id, true) ||
         exe_stem_compact == normalize_alias_key(entry.display_name, true))) {
      return true;
    }
  }

  for (const auto& cand_proc : candidate.process_names) {
    const std::string cand_proc_lower = ascii_lower(cand_proc);
    for (const auto& entry_proc : entry.process_names) {
      if (cand_proc_lower == ascii_lower(entry_proc)) return true;
    }
  }
  return false;
}

std::string derive_candidate_id(
    const DiscoveredApp& candidate,
    const std::unordered_set<std::string>& used_ids) {
  std::string base = sanitize_slug(candidate.raw_name);
  if (!is_valid_app_id(base) && !candidate.resolved_exe.empty()) {
    base = sanitize_slug(path_to_utf8_string(candidate.resolved_exe.stem()));
  }
  if (!is_valid_app_id(base)) {
    base = fnv1a_short_id(candidate.raw_name + "|" + path_to_utf8_string(candidate.launch_target));
  }
  if (!used_ids.contains(base)) return base;

  for (std::uint32_t suffix = 2; suffix <= 999; ++suffix) {
    const std::string tag = "-" + std::to_string(suffix);
    std::string trimmed = base;
    if (trimmed.size() + tag.size() > kMaxAppIdBytes) {
      trimmed.resize(kMaxAppIdBytes - tag.size());
      while (!trimmed.empty() && (trimmed.back() == '-' || trimmed.back() == '.')) {
        trimmed.pop_back();
      }
    }
    const std::string candidate_id = trimmed + tag;
    if (is_valid_app_id(candidate_id) && !used_ids.contains(candidate_id)) {
      return candidate_id;
    }
  }
  return base;
}

}  // namespace

const char* to_string(AppEntrySource source) {
  switch (source) {
    case AppEntrySource::configured:
      return "configured";
    case AppEntrySource::quickstart:
      return "quickstart";
    case AppEntrySource::system_index:
      return "system_index";
  }
  return "configured";
}

std::string expand_windows_env_vars(std::string_view value) {
  if (value.empty()) return {};
  if (value.find('%') == std::string_view::npos) return std::string(value);
  const std::wstring wide = utf8_to_wide(value);
  if (wide.empty()) return std::string(value);
  const DWORD required = ExpandEnvironmentStringsW(wide.c_str(), nullptr, 0);
  if (required == 0) return std::string(value);
  std::wstring expanded(static_cast<std::size_t>(required), L'\0');
  const DWORD written = ExpandEnvironmentStringsW(wide.c_str(), expanded.data(), required);
  if (written == 0) return std::string(value);
  while (!expanded.empty() && expanded.back() == L'\0') {
    expanded.pop_back();
  }
  return wide_to_utf8(expanded);
}

std::filesystem::path expand_windows_path(const std::filesystem::path& value) {
  if (value.empty()) return {};
  const std::string utf8 = path_to_utf8_string(value);
  const std::string expanded = expand_windows_env_vars(utf8);
  return utf8_to_path(expanded);
}

bool is_noise_app_name(std::string_view text) {
  if (text.empty()) return true;
  const std::string lower = ascii_lower(text);
  constexpr std::array<std::string_view, 5> kNoiseTokens{
      "uninstall",
      "卸载",
      "readme",
      "help",
      "updater",
  };
  for (const auto token : kNoiseTokens) {
    if (lower.find(token) != std::string::npos) return true;
  }
  return false;
}

AppCatalog::AppCatalog(std::vector<AppCatalogEntry> entries)
    : entries_(std::move(entries)) {
  rebuild_indices();
}

AppCatalog AppCatalog::from_config(
    const AppsConfig& config,
    const AppCatalogBuildOptions& options) {
  const ScopedComInit com_guard;

  std::vector<DiscoveredApp> quickstart_candidates;
  for (const auto& raw_dir : config.quickstart_dirs) {
    const std::filesystem::path dir = utf8_to_path(expand_windows_env_vars(raw_dir));
    scan_directory_files(dir, /*max_depth=*/1, /*lnk_only=*/false,
                         AppEntrySource::quickstart, quickstart_candidates);
  }

  std::vector<DiscoveredApp> system_candidates;
  if (config.scan_system_apps) {
    std::vector<std::filesystem::path> program_dirs;
    if (options.system_program_dirs_override.has_value()) {
      program_dirs = *options.system_program_dirs_override;
    } else {
      program_dirs.push_back(utf8_to_path(
          expand_windows_env_vars("%APPDATA%/Microsoft/Windows/Start Menu/Programs")));
      program_dirs.push_back(utf8_to_path(
          expand_windows_env_vars("%PROGRAMDATA%/Microsoft/Windows/Start Menu/Programs")));
    }
    for (const auto& dir : program_dirs) {
      scan_directory_files(dir, /*max_depth=*/2, /*lnk_only=*/true,
                           AppEntrySource::system_index, system_candidates);
    }
    if (options.query_registry_app_paths) {
      scan_registry_app_paths_root(HKEY_CURRENT_USER, system_candidates);
      scan_registry_app_paths_root(HKEY_LOCAL_MACHINE, system_candidates);
    }
  }

  std::vector<AppCatalogEntry> merged;
  std::unordered_set<std::string> used_ids;
  std::unordered_set<std::string> claimed_compact_names;
  std::unordered_set<std::string> claimed_targets;

  const auto register_claims = [&](const AppCatalogEntry& item) {
    used_ids.insert(item.id);
    if (const auto k = normalize_alias_key(item.id, true); !k.empty()) {
      claimed_compact_names.insert(k);
    }
    if (const auto k = normalize_alias_key(item.display_name, true); !k.empty()) {
      claimed_compact_names.insert(k);
    }
    for (const auto& alias : item.aliases) {
      if (const auto k = normalize_alias_key(alias, true); !k.empty()) {
        claimed_compact_names.insert(k);
      }
    }
    if (!item.target.empty()) {
      claimed_targets.insert(ascii_lower(path_to_utf8_string(item.target.lexically_normal())));
    }
  };

  std::vector<bool> quickstart_consumed(quickstart_candidates.size(), false);
  std::vector<bool> system_consumed(system_candidates.size(), false);

  for (const auto& configured : config.entries) {
    if (!is_valid_app_id(configured.id)) continue;
    if (used_ids.contains(configured.id)) continue;

    AppCatalogEntry entry;
    entry.id = configured.id;
    entry.display_name = configured.display_name.empty() ? configured.id : configured.display_name;
    entry.source = AppEntrySource::configured;
    entry.target = expand_windows_path(configured.target);
    entry.arguments = configured.arguments;
    entry.aliases = configured.aliases;
    entry.process_names = configured.process_names;

    const auto bind_from_pool = [&](const std::vector<DiscoveredApp>& pool,
                                    std::vector<bool>& consumed_flags) {
      for (std::size_t i = 0; i < pool.size(); ++i) {
        if (!candidate_matches_configured(entry, pool[i])) continue;
        consumed_flags[i] = true;
        if (entry.target.empty() && !pool[i].launch_target.empty()) {
          entry.target = pool[i].launch_target;
        }
        for (const auto& proc : pool[i].process_names) {
          const bool exists = std::any_of(
              entry.process_names.begin(), entry.process_names.end(),
              [&](const std::string& existing) {
                return ascii_lower(existing) == ascii_lower(proc);
              });
          if (!exists) entry.process_names.push_back(proc);
        }
      }
    };

    bind_from_pool(quickstart_candidates, quickstart_consumed);
    bind_from_pool(system_candidates, system_consumed);

    register_claims(entry);
    merged.push_back(std::move(entry));
  }

  const auto append_discovered_pool = [&](const std::vector<DiscoveredApp>& pool,
                                          const std::vector<bool>& consumed_flags) {
    for (std::size_t i = 0; i < pool.size(); ++i) {
      if (consumed_flags[i]) continue;
      const auto& item = pool[i];
      const std::string compact_name = normalize_alias_key(item.raw_name, true);
      if (!compact_name.empty() && claimed_compact_names.contains(compact_name)) {
        continue;
      }
      const std::string norm_target =
          ascii_lower(path_to_utf8_string(item.launch_target.lexically_normal()));
      if (!norm_target.empty() && claimed_targets.contains(norm_target)) {
        continue;
      }

      AppCatalogEntry entry;
      entry.id = derive_candidate_id(item, used_ids);
      if (!is_valid_app_id(entry.id)) continue;
      entry.display_name = item.raw_name;
      entry.source = item.source;
      entry.target = item.launch_target;
      entry.aliases.push_back(item.raw_name);
      entry.process_names = item.process_names;

      register_claims(entry);
      merged.push_back(std::move(entry));
    }
  };

  append_discovered_pool(quickstart_candidates, quickstart_consumed);
  append_discovered_pool(system_candidates, system_consumed);

  return AppCatalog(std::move(merged));
}

void AppCatalog::rebuild_indices() {
  id_index_.clear();
  exact_alias_index_.clear();
  compact_alias_index_.clear();

  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const auto& entry = entries_[i];
    const std::string lower_id = ascii_lower(entry.id);
    id_index_.try_emplace(lower_id, i);

    const auto register_term = [&](std::string_view term) {
      if (const std::string exact = normalize_alias_key(term, false); !exact.empty()) {
        exact_alias_index_.try_emplace(exact, i);
      }
      if (const std::string compact = normalize_alias_key(term, true); !compact.empty()) {
        compact_alias_index_.try_emplace(compact, i);
      }
    };

    register_term(entry.id);
    register_term(entry.display_name);
    for (const auto& alias : entry.aliases) {
      register_term(alias);
    }
  }
}

const AppCatalogEntry* AppCatalog::find_by_id(std::string_view app_id) const {
  if (app_id.empty()) return nullptr;
  const auto it = id_index_.find(ascii_lower(app_id));
  if (it == id_index_.end()) return nullptr;
  return &entries_[it->second];
}

const AppCatalogEntry* AppCatalog::match_alias(std::string_view text) const {
  const std::string exact = normalize_alias_key(text, false);
  if (exact.empty()) return nullptr;
  if (const auto it = exact_alias_index_.find(exact); it != exact_alias_index_.end()) {
    return &entries_[it->second];
  }
  const std::string compact = normalize_alias_key(text, true);
  if (!compact.empty()) {
    if (const auto it = compact_alias_index_.find(compact); it != compact_alias_index_.end()) {
      return &entries_[it->second];
    }
  }
  return nullptr;
}

nlohmann::json AppCatalog::to_json() const {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& entry : entries_) {
    arr.push_back({
        {"id", entry.id},
        {"display_name", entry.display_name},
        {"source", to_string(entry.source)},
        {"target", path_to_utf8_string(entry.target)},
        {"arguments", entry.arguments},
        {"aliases", entry.aliases},
        {"process_names", entry.process_names},
    });
  }
  return arr;
}

}  // namespace dvo
