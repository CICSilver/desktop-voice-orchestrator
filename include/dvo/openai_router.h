#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "dvo/app_catalog.h"
#include "dvo/command_parser.h"

namespace dvo {

struct OpenAiRouterConfig {
  std::string base_url{"https://api.openai.com/v1"};
  std::string model{"gpt-4o-mini"};
  std::string api_key_env{"OPENAI_API_KEY"};
  std::uint32_t timeout_ms{2500};
  bool structured_outputs{true};
};

// Builds the strict JSON Schema sent in response_format.json_schema.schema.
[[nodiscard]] nlohmann::json build_openai_router_response_schema();

// Recursively verifies OpenAI Structured Outputs (strict: true) invariants:
// every object schema must declare "additionalProperties": false and its
// "required" array must list every key in "properties". Returns std::nullopt
// when valid, or a human-readable violation description otherwise.
[[nodiscard]] std::optional<std::string> validate_openai_strict_json_schema(
    const nlohmann::json& schema);

// Builds the concise system prompt including action rules and the registered
// AppCatalog whitelist.
[[nodiscard]] std::string build_openai_router_system_prompt(
    const LlmRouteRequest& request,
    const AppCatalog* app_catalog);

class OpenAiCompatibleCommandRouter final : public ILlmCommandRouter {
 public:
  explicit OpenAiCompatibleCommandRouter(
      OpenAiRouterConfig config,
      std::shared_ptr<const AppCatalog> app_catalog = nullptr);

  [[nodiscard]] LlmRouteResult route(
      const LlmRouteRequest& request) const override;

  [[nodiscard]] const OpenAiRouterConfig& config() const noexcept {
    return config_;
  }

 private:
  OpenAiRouterConfig config_;
  std::shared_ptr<const AppCatalog> app_catalog_;
  nlohmann::json response_schema_;
};

}  // namespace dvo
