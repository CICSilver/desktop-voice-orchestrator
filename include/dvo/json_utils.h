#pragma once

#include <nlohmann/json.hpp>

#include "dvo/audio_types.h"

namespace dvo {

inline nlohmann::json span_json(const SampleSpan& span) {
  return {{"start", span.start}, {"end", span.end}};
}

inline nlohmann::json kws_json(const KwsHit& hit) {
  return {{"keyword", hit.keyword}, {"tokens", hit.tokens},
          {"token_samples", hit.token_samples}, {"wake_span", span_json(hit.wake_span)},
          {"detected_at_sample", hit.detected_at_sample}};
}

inline nlohmann::json candidate_json(const UtteranceCandidate& candidate, bool include_pcm = false) {
  auto value = nlohmann::json{{"utterance_id", candidate.utterance_id},
                              {"position", to_string(candidate.position)},
                              {"keyword", candidate.keyword},
                              {"tokens", candidate.tokens},
                              {"wake_span", span_json(candidate.wake_span)},
                              {"sample_rate", candidate.sample_rate},
                              {"boundary_source", candidate.boundary_source},
                              {"truncated", candidate.truncated},
                              {"timed_out", candidate.timed_out},
                              {"discontinuity", candidate.discontinuity},
                              {"pcm_samples", candidate.pcm.size()}};
  value["source_spans"] = nlohmann::json::array();
  for (const auto& span : candidate.source_spans) value["source_spans"].push_back(span_json(span));
  if (include_pcm) value["pcm"] = candidate.pcm;
  return value;
}

}  // namespace dvo
