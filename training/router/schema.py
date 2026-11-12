"""Desktop Voice Orchestrator — Router JSON Schema & Prompt Alignment (M4 Slice C).

Mirrors src/openai_router.cpp response schema, system prompt, and plan validation.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

ACTION_TYPES: tuple[str, ...] = (
    "media.play",
    "media.pause",
    "audio.volume.adjust",
    "media.next",
    "media.like",
    "media.play_daily",
    "media.mode.order",
    "media.mode.list_loop",
    "media.mode.single_loop",
    "media.mode.shuffle",
    "media.lyrics.show",
    "media.lyrics.hide",
    "media.search_play",
    "app.open",
)

ACTION_KEYS: tuple[str, ...] = (
    "type",
    "volume_delta_percent",
    "search_query",
    "search_type",
    "search_artist",
    "app_id",
)

SEARCH_TYPES: tuple[str, ...] = ("song", "playlist")


def build_openai_router_response_schema() -> dict[str, Any]:
    """Return verbatim OpenAI strict JSON schema matching src/openai_router.cpp."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(ACTION_TYPES),
                        },
                        "volume_delta_percent": {"type": ["integer", "null"]},
                        "search_query": {"type": ["string", "null"]},
                        "search_type": {
                            "type": ["string", "null"],
                            "enum": ["song", "playlist", None],
                        },
                        "search_artist": {"type": ["string", "null"]},
                        "app_id": {"type": ["string", "null"]},
                    },
                    "required": list(ACTION_KEYS),
                },
            }
        },
        "required": ["actions"],
    }


def build_system_prompt(
    app_entries: Sequence[Mapping[str, Any]] | None = None,
    default_step: int = 5,
    max_step: int = 20,
    max_actions: int = 8,
) -> str:
    """Construct system prompt byte-exact aligned with src/openai_router.cpp lines 463-503."""
    out = (
        "你是 Windows 桌面语音编排器的极速意图规划器。请将用户话语转换为严格的 JSON：{\"actions\":[...]}。\n"
        "核心规则：\n"
        "1. 日常闲聊、感叹句、歌词背景音、提问、或自我否定/取消意图（如“算了别切了”、“不用打开了”、“这首歌真好听”）必须返回 {\"actions\":[]}。\n"
        f"2. 多步指令必须严格按用户口述先后顺序排列，最多 {max_actions} 个动作；未使用的字段一律填 null。\n"
        "3. 受控动作与参数：\n"
        "   - media.play / media.pause / media.next / media.like / media.play_daily / media.mode.order / media.mode.list_loop / media.mode.single_loop / media.mode.shuffle / media.lyrics.show / media.lyrics.hide：全部可选参数填 null。\n"
        f"   - audio.volume.adjust：volume_delta_percent 填非零整数（范围 -{max_step}..{max_step}，未说明具体数值如“小声点/大声点”时默认填 -{default_step} 或 {default_step}）。\n"
        "   - media.search_play：search_query 填歌名或歌单关键词，search_type 填 \"song\" 或 \"playlist\"，search_artist 填歌手名或 null。\n"
        "   - app.open：app_id 只能填写下方已注册应用白名单中的 id（支持根据用途推断，如“挂代理的软件”对应 clash；若不在白名单中则返回空数组）。\n"
        "已注册应用白名单（id: 名称 [别名]）：\n"
    )

    if not app_entries:
        out += "- （当前无已注册应用）"
        return out

    max_prompt_apps = 60
    entries_to_use = list(app_entries)[:max_prompt_apps]
    total = len(entries_to_use)

    for i, app in enumerate(entries_to_use):
        app_id = str(app.get("id", "")).strip()
        name = str(app.get("display_name", app_id)).strip()
        out += f"- {app_id}: {name}"

        raw_aliases = app.get("aliases", [])
        if raw_aliases:
            aliases = [str(a).strip() for a in raw_aliases if str(a).strip()][:5]
            if aliases:
                out += f" ({', '.join(aliases)})"

        if i + 1 < total:
            out += "\n"

    return out


def strip_markdown_code_fence(text: str) -> str:
    """Strip markdown code fence blocks (```json ... ```) and leading/trailing whitespace."""
    s = text.strip()
    match = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```$", s)
    if match:
        return match.group(1).strip()
    return s


def create_action(
    action_type: str,
    *,
    volume_delta_percent: int | None = None,
    search_query: str | None = None,
    search_type: str | None = None,
    search_artist: str | None = None,
    app_id: str | None = None,
) -> dict[str, Any]:
    """Helper to create an action dict with all 6 required keys."""
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unknown action type: {action_type}")
    return {
        "type": action_type,
        "volume_delta_percent": volume_delta_percent,
        "search_query": search_query,
        "search_type": search_type,
        "search_artist": search_artist,
        "app_id": app_id,
    }


def validate_router_plan(
    obj: Any,
    allowed_app_ids: Iterable[str] | None = None,
    max_actions: int = 8,
    max_volume_delta: int = 20,
) -> tuple[bool, str]:
    """Validate router JSON response object against C++ runtime requirements."""
    if not isinstance(obj, dict):
        return False, "top-level plan must be a JSON object"

    extra_top = set(obj.keys()) - {"actions"}
    if extra_top:
        return False, f"top-level object contains disallowed extra properties: {extra_top}"

    if "actions" not in obj:
        return False, "top-level object must contain 'actions' array"

    actions = obj["actions"]
    if not isinstance(actions, list):
        return False, "'actions' field must be an array"

    if len(actions) > max_actions:
        return False, f"actions count ({len(actions)}) exceeds max_actions ({max_actions})"

    allowed_set = {str(a).strip() for a in allowed_app_ids} if allowed_app_ids is not None else None
    expected_keys = set(ACTION_KEYS)

    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            return False, f"actions[{i}] must be an object"

        actual_keys = set(action.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            details = []
            if missing:
                details.append(f"missing: {sorted(missing)}")
            if extra:
                details.append(f"extra: {sorted(extra)}")
            return False, f"actions[{i}] schema violation ({', '.join(details)})"

        atype = action["type"]
        if atype not in ACTION_TYPES:
            return False, f"actions[{i}] invalid action type '{atype}'"

        vol = action["volume_delta_percent"]
        s_query = action["search_query"]
        s_type = action["search_type"]
        s_artist = action["search_artist"]
        app_id = action["app_id"]

        if atype == "audio.volume.adjust":
            if vol is None or type(vol) is not int:
                return False, f"actions[{i}] volume_delta_percent must be a non-null integer"
            if vol == 0:
                return False, f"actions[{i}] volume_delta_percent cannot be zero"
            if abs(vol) > max_volume_delta:
                return (
                    False,
                    f"actions[{i}] volume_delta_percent {vol} exceeds maximum absolute delta {max_volume_delta}",
                )
            if s_query is not None or s_type is not None or s_artist is not None or app_id is not None:
                return False, f"actions[{i}] audio.volume.adjust must have other fields set to null"

        elif atype == "media.search_play":
            if not isinstance(s_query, str) or not s_query.strip():
                return False, f"actions[{i}] media.search_play search_query must be non-empty string"
            if s_type not in SEARCH_TYPES:
                return False, f"actions[{i}] media.search_play search_type must be 'song' or 'playlist'"
            if s_artist is not None and not isinstance(s_artist, str):
                return False, f"actions[{i}] media.search_play search_artist must be string or null"
            if vol is not None or app_id is not None:
                return False, f"actions[{i}] media.search_play volume_delta_percent and app_id must be null"

        elif atype == "app.open":
            if not isinstance(app_id, str) or not app_id.strip():
                return False, f"actions[{i}] app.open app_id must be non-empty string"
            clean_app_id = app_id.strip()
            if allowed_set is not None and clean_app_id not in allowed_set:
                return False, f"actions[{i}] app.open app_id '{clean_app_id}' not in allowed catalog"
            if vol is not None or s_query is not None or s_type is not None or s_artist is not None:
                return False, f"actions[{i}] app.open other parameter fields must be null"

        else:
            # All other 11 media actions require all 5 optional parameters to be None (null)
            if (
                vol is not None
                or s_query is not None
                or s_type is not None
                or s_artist is not None
                or app_id is not None
            ):
                return False, f"actions[{i}] {atype} all optional parameters must be null"

    return True, ""
