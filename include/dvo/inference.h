#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dvo/audio_types.h"
#include "dvo/config.h"

namespace dvo {

struct VadUpdate {
  bool speech{};
  std::vector<VadInterval> completed;
};

class IVadDetector {
 public:
  virtual ~IVadDetector() = default;
  virtual VadUpdate accept(const NormalizedFrame& frame) = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual bool available() const = 0;
  [[nodiscard]] virtual std::string status() const = 0;
};

class IKeywordSpotter {
 public:
  virtual ~IKeywordSpotter() = default;
  virtual std::optional<KwsHit> accept(const NormalizedFrame& frame) = 0;
  virtual void reset(std::uint64_t next_sample) = 0;
  [[nodiscard]] virtual bool available() const = 0;
  [[nodiscard]] virtual std::string status() const = 0;
};

std::unique_ptr<IVadDetector> create_vad(const VadConfig& config);
std::unique_ptr<IKeywordSpotter> create_keyword_spotter(const KwsConfig& config);

}  // namespace dvo
