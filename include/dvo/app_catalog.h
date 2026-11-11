#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "dvo/config.h"

namespace dvo {

enum class AppEntrySource {
  configured,
  quickstart,
  system_index,
};

[[nodiscard]] const char* to_string(AppEntrySource source);

struct AppCatalogEntry {
  std::string id;
  std::string display_name;
  AppEntrySource source{AppEntrySource::configured};
  std::filesystem::path target;
  std::vector<std::string> arguments;
  std::vector<std::string> aliases;
  std::vector<std::string> process_names;
};

struct AppCatalogBuildOptions {
  // Overrides Windows Start Menu Programs directories in unit tests when set.
  std::optional<std::vector<std::filesystem::path>> system_program_dirs_override;
  bool query_registry_app_paths{true};
};

[[nodiscard]] std::string expand_windows_env_vars(std::string_view value);
[[nodiscard]] std::filesystem::path expand_windows_path(const std::filesystem::path& value);
[[nodiscard]] bool is_noise_app_name(std::string_view text);

class AppCatalog {
 public:
  AppCatalog() = default;
  explicit AppCatalog(std::vector<AppCatalogEntry> entries);

  [[nodiscard]] static AppCatalog from_config(
      const AppsConfig& config,
      const AppCatalogBuildOptions& options = {});

  [[nodiscard]] const AppCatalogEntry* find_by_id(std::string_view app_id) const;
  [[nodiscard]] const AppCatalogEntry* match_alias(std::string_view text) const;
  [[nodiscard]] const std::vector<AppCatalogEntry>& entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] nlohmann::json to_json() const;

 private:
  void rebuild_indices();

  std::vector<AppCatalogEntry> entries_;
  std::unordered_map<std::string, std::size_t> id_index_;
  std::unordered_map<std::string, std::size_t> exact_alias_index_;
  std::unordered_map<std::string, std::size_t> compact_alias_index_;
};

}  // namespace dvo
