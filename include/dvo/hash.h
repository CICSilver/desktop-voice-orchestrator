#pragma once

#include <filesystem>
#include <string>

namespace dvo {

[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

}  // namespace dvo
