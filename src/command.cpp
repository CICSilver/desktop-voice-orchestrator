#include "dvo/command.h"

namespace dvo {

const char* to_string(ActionType type) {
  switch (type) {
    case ActionType::media_play: return "media.play";
    case ActionType::media_pause: return "media.pause";
    case ActionType::master_volume_adjust: return "audio.volume.adjust";
    case ActionType::media_next: return "media.next";
    case ActionType::media_like: return "media.like";
    case ActionType::media_play_daily: return "media.play_daily";
    case ActionType::media_mode_order: return "media.mode.order";
    case ActionType::media_mode_list_loop: return "media.mode.list_loop";
    case ActionType::media_mode_single_loop: return "media.mode.single_loop";
    case ActionType::media_mode_shuffle: return "media.mode.shuffle";
    case ActionType::media_lyrics_show: return "media.lyrics.show";
    case ActionType::media_lyrics_hide: return "media.lyrics.hide";
  }
  return "unknown";
}

const char* to_string(ExecutionSource source) {
  switch (source) {
    case ExecutionSource::live: return "live";
    case ExecutionSource::replay: return "replay";
    case ExecutionSource::benchmark: return "benchmark";
  }
  return "unknown";
}

const char* to_string(ExecutionMode mode) {
  switch (mode) {
    case ExecutionMode::live: return "live";
    case ExecutionMode::dry_run: return "dry_run";
  }
  return "unknown";
}

const char* to_string(ActionStatus status) {
  switch (status) {
    case ActionStatus::started: return "started";
    case ActionStatus::succeeded: return "succeeded";
    case ActionStatus::noop: return "noop";
    case ActionStatus::failed: return "failed";
    case ActionStatus::indeterminate: return "indeterminate";
    case ActionStatus::deduplicated: return "deduplicated";
    case ActionStatus::dry_run: return "dry_run";
    case ActionStatus::cancelled: return "cancelled";
  }
  return "unknown";
}

}  // namespace dvo
