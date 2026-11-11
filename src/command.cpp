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
    case ActionType::media_search_play: return "media.search_play";
    case ActionType::app_open: return "app.open";
  }
  return "unknown";
}

std::optional<ActionType> parse_action_type(std::string_view value) {
  if (value == "media.play") return ActionType::media_play;
  if (value == "media.pause") return ActionType::media_pause;
  if (value == "audio.volume.adjust") return ActionType::master_volume_adjust;
  if (value == "media.next") return ActionType::media_next;
  if (value == "media.like") return ActionType::media_like;
  if (value == "media.play_daily") return ActionType::media_play_daily;
  if (value == "media.mode.order") return ActionType::media_mode_order;
  if (value == "media.mode.list_loop") return ActionType::media_mode_list_loop;
  if (value == "media.mode.single_loop") return ActionType::media_mode_single_loop;
  if (value == "media.mode.shuffle") return ActionType::media_mode_shuffle;
  if (value == "media.lyrics.show") return ActionType::media_lyrics_show;
  if (value == "media.lyrics.hide") return ActionType::media_lyrics_hide;
  if (value == "media.search_play") return ActionType::media_search_play;
  if (value == "app.open") return ActionType::app_open;
  return std::nullopt;
}

bool is_valid_app_id(std::string_view app_id) {
  if (app_id.empty() || app_id.size() > kMaxAppIdBytes) return false;
  if (app_id.front() == '.' || app_id.back() == '.') return false;
  if (app_id.find("..") != std::string_view::npos) return false;
  for (const char ch : app_id) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
                    ch == '_' || ch == '-' || ch == '.';
    if (!ok) return false;
  }
  return true;
}

const char* to_string(MediaSearchType type) {
  switch (type) {
    case MediaSearchType::song: return "song";
    case MediaSearchType::playlist: return "playlist";
  }
  return "unknown";
}

std::optional<MediaSearchType> parse_media_search_type(std::string_view value) {
  if (value == "song") return MediaSearchType::song;
  if (value == "playlist") return MediaSearchType::playlist;
  return std::nullopt;
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
