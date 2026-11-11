#!/usr/bin/env python3
"""Evaluate M2 natural-language command routing accuracy, safety, and latency on the 60-item benchmark.

Supports two end-to-end evaluation modes plus offline self-testing:
  1. `--backend openai` : Calls an OpenAI-compatible `/v1/chat/completions` endpoint directly
                          (reading `OPENAI_API_KEY` from the environment; never logged or persisted),
                          supporting serial or concurrent execution (`--concurrency`), and reporting
                          P50 / P95 latency (ms), positive exact-match rate (target >= 95%), and
                          negative false-execution rate (target == 0%).
  2. `--backend cpp`    : Drives the compiled `voice_frontend.exe parse --mode=hybrid` CLI over
                          line-streamed UTF-8 stdin/stdout for C++ end-to-end regression testing.
  3. `--self-test`      : Offline verification of `tools/nl_router_eval_60.jsonl` (exact 40 + 20 split,
                          all 14 C++ ActionTypes, schema validity) and scorer / C++ JSON parser logic.

Usage:
    # Offline schema & scorer self-test:
    python tools/eval_nl_router.py --self-test

    # Direct OpenAI-compatible HTTP router benchmark:
    python tools/eval_nl_router.py --backend openai --model gpt-4o-mini

    # C++ voice_frontend end-to-end hybrid parser regression:
    python tools/eval_nl_router.py --backend cpp --parser-mode hybrid
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "tools" / "nl_router_eval_60.jsonl"

VALID_ACTION_TYPES: set[str] = {
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
}

DEFAULT_APP_CATALOG: dict[str, dict[str, Any]] = {
    "steam": {"display_name": "Steam", "aliases": ["steam", "蒸汽平台", "死盯", "斯蒂姆", "游戏平台"]},
    "clash": {"display_name": "Clash", "aliases": ["clash", "代理软件", "可拉什", "克拉什", "小猫咪"]},
    "discord": {"display_name": "Discord", "aliases": ["discord", "语音软件", "迪斯科", "迪斯扣", "开黑语音"]},
    "qq": {"display_name": "QQ", "aliases": ["qq", "腾讯qq", "扣扣", "企鹅"]},
    "powershell": {"display_name": "PowerShell", "aliases": ["powershell", "终端", "命令行", "泡我秀", "帕瓦谢尔"]},
    "vscode": {"display_name": "VS Code", "aliases": ["vscode", "vs code", "代码编辑器"]},
    "wechat": {"display_name": "微信", "aliases": ["微信", "wechat", "威信"]},
    "netease": {"display_name": "网易云音乐", "aliases": ["网易云音乐", "网易云", "云音乐"]},
}

ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "name": "dvo_command_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["actions"],
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "type",
                        "volume_delta_percent",
                        "search_query",
                        "search_type",
                        "search_artist",
                        "app_id",
                    ],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": sorted(VALID_ACTION_TYPES),
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
                },
            }
        },
    },
}


def load_app_catalog(local_toml: Path | None) -> dict[str, dict[str, Any]]:
    catalog = {k: {"display_name": v["display_name"], "aliases": list(v["aliases"])} for k, v in DEFAULT_APP_CATALOG.items()}
    if local_toml and local_toml.is_file():
        try:
            parsed = tomllib.loads(local_toml.read_text("utf-8"))
            entries = parsed.get("apps", {}).get("entries", {})
            for app_id, entry in entries.items():
                aid = app_id.strip().lower()
                disp = str(entry.get("display_name", aid))
                aliases = [str(a) for a in entry.get("aliases", [])]
                if aid in catalog:
                    merged = list(dict.fromkeys(catalog[aid]["aliases"] + aliases))
                    catalog[aid] = {"display_name": disp, "aliases": merged}
                else:
                    catalog[aid] = {"display_name": disp, "aliases": aliases}
        except Exception as exc:
            print(f"Warning: failed to read [apps] from {local_toml}: {exc}", file=sys.stderr)
    return catalog


def build_router_system_prompt(app_catalog: dict[str, dict[str, Any]], default_vol_step: int = 5) -> str:
    app_lines = [
        f'  - "{aid}" ({info["display_name"]}): aliases={json.dumps(info["aliases"], ensure_ascii=False)}'
        for aid, info in sorted(app_catalog.items())
    ]
    apps_section = "\n".join(app_lines)
    return (
        "你是桌面语音编排器（Desktop Voice Orchestrator）的纯动作规划路由器（Pure Planner）。\n"
        "请将用户的中文/中英混排语音识别文本（可能带有同音错别字）解析为严格保序的动作列表 JSON `{\"actions\": [...]}`。\n\n"
        "【安全拒识铁律（最高优先级）】\n"
        "遇到以下任何情况，必须返回空数组 `{\"actions\": []}`：\n"
        "1. 自我否定、中途反悔、取消或禁止指令（即使包含唤醒词“小克”）：例如“算了还是别切歌了”、“不用打开某某了”、“先不要暂停音乐”、“别调大音量了”、“没事了叫错你了”；\n"
        "2. 日常闲聊、主观评价或陈述过去/现状：例如“这首歌旋律真好听”、“晴天真是百听不厌”、“单曲循环了一下午”、“今天 Steam 特卖打折”、“每日推荐挺合胃口”；\n"
        "3. 游戏开黑队友沟通或非桌面媒体/白名单应用控制：例如“小心后面有人快撤”、“等一下我换把枪”、“快切刀跑”、“把窗户关小一点”、“听得到我说话吗”。\n\n"
        "【动作类型与参数规范】\n"
        "- `media.play` / `media.pause` / `media.next` / `media.like` / `media.play_daily`\n"
        "- `media.mode.order` / `media.mode.list_loop` / `media.mode.single_loop` / `media.mode.shuffle`\n"
        "- `media.lyrics.show` / `media.lyrics.hide`\n"
        f"- `audio.volume.adjust`：必填整数 `volume_delta_percent`（调大为正，调小为负；若用户未口述具体百分比，默认用 `+{default_vol_step}` 或 `-{default_vol_step}`）。\n"
        "- `media.search_play`：必填 `search_type`（`\"song\"` 或 `\"playlist\"`）与 `search_query`：\n"
        "    * 若指明歌手与歌名（如“周杰伦的晴天”），设 `search_type=\"song\"`, `search_artist=\"周杰伦\"`, `search_query=\"晴天\"`；\n"
        "    * 若泛指某歌手的歌（如“随便放几首林俊杰的歌”），设 `search_type=\"song\"`, `search_artist=\"林俊杰\"`, `search_query=\"林俊杰\"`；\n"
        "    * 若明确找歌单/曲风合集（如“写代码的纯音乐歌单”、“我喜欢的音乐歌单”），设 `search_type=\"playlist\"`, `search_artist=null`。\n"
        "- `app.open`：必填小写 `app_id`（必须从下方已注册桌面应用列表中匹配，支持 ASR 谐音纠错），其余无关参数填 `null`：\n"
        f"{apps_section}\n\n"
        "复合多步指令请严格按照用户口述的执行先后顺序输出多个动作对象。"
    )


def load_eval_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark JSONL not found: {path}")
    items: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        items.append(obj)
    return items


def validate_eval_dataset(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Validates benchmark integrity: 60 items (40 positive + 20 negative), valid C++ ActionTypes & parameters."""
    if len(items) != 60:
        raise AssertionError(f"Expected exactly 60 benchmark items, got {len(items)}")

    seen_ids: set[str] = set()
    pos_items = [x for x in items if x.get("split") == "positive"]
    neg_items = [x for x in items if x.get("split") == "negative"]
    if len(pos_items) != 40 or len(neg_items) != 20:
        raise AssertionError(f"Expected 40 positive + 20 negative items, got {len(pos_items)} + {len(neg_items)}")

    seen_action_types: set[str] = set()
    seen_apps: set[str] = set()
    max_steps = 0

    for it in items:
        sid = it.get("id", "")
        if not sid or sid in seen_ids:
            raise AssertionError(f"Missing or duplicate sample id: {sid!r}")
        seen_ids.add(sid)

        if not isinstance(it.get("text"), str) or not it["text"].strip():
            raise AssertionError(f"{sid}: empty text")
        exp = it.get("expected_actions")
        if not isinstance(exp, list):
            raise AssertionError(f"{sid}: expected_actions must be a list")

        if it["split"] == "negative":
            if len(exp) != 0:
                raise AssertionError(f"{sid}: negative sample must have expected_actions == []")
        else:
            if len(exp) < 1:
                raise AssertionError(f"{sid}: positive sample must have >= 1 expected action")
            max_steps = max(max_steps, len(exp))
            for act in exp:
                atype = act.get("type")
                if atype not in VALID_ACTION_TYPES:
                    raise AssertionError(f"{sid}: unknown action type {atype!r}")
                seen_action_types.add(atype)
                if atype == "audio.volume.adjust":
                    delta = act.get("volume_delta_percent")
                    if not isinstance(delta, int) or delta == 0 or not (-100 <= delta <= 100):
                        raise AssertionError(f"{sid}: invalid volume_delta_percent {delta!r}")
                elif atype == "media.search_play":
                    if act.get("search_type") not in ("song", "playlist"):
                        raise AssertionError(f"{sid}: invalid search_type {act.get('search_type')!r}")
                    if not isinstance(act.get("search_query"), str) or not act["search_query"].strip():
                        raise AssertionError(f"{sid}: missing search_query")
                elif atype == "app.open":
                    aid = act.get("app_id")
                    if not isinstance(aid, str) or not aid.strip():
                        raise AssertionError(f"{sid}: missing app_id for app.open")
                    seen_apps.add(aid.lower())

    missing_types = VALID_ACTION_TYPES - seen_action_types
    if missing_types:
        raise AssertionError(f"Benchmark does not cover all 14 ActionTypes; missing: {sorted(missing_types)}")
    required_apps = {"steam", "clash", "discord", "qq", "powershell"}
    if not required_apps.issubset(seen_apps):
        raise AssertionError(f"Benchmark missing required desktop apps: {required_apps - seen_apps}")
    if max_steps < 3:
        raise AssertionError("Benchmark must include at least one 3-step compound instruction")

    return {
        "total": len(items),
        "positive": len(pos_items),
        "negative": len(neg_items),
        "covered_action_types": len(seen_action_types),
        "covered_apps": sorted(seen_apps),
        "max_compound_steps": max_steps,
    }


def norm_query_str(s: str | None) -> str:
    if not s:
        return ""
    cleaned = re.sub(r"[\s·\-—_《》\"'“”‘’（）()]+", "", s.strip().lower())
    # Strip trailing redundant "歌单" / "的歌" when comparing core query intent
    return cleaned


def actions_match_single(
    exp: dict[str, Any],
    act: dict[str, Any],
    allowed_volume_deltas: list[int] | None = None,
) -> tuple[bool, str]:
    etype = exp.get("type")
    atype = act.get("type")
    if etype != atype:
        return False, f"type mismatch: expected {etype!r}, got {atype!r}"

    if etype == "audio.volume.adjust":
        edelta = exp["volume_delta_percent"]
        adelta = act.get("volume_delta_percent")
        if not isinstance(adelta, int):
            return False, f"volume_delta_percent not int: {adelta!r}"
        if allowed_volume_deltas:
            if adelta not in allowed_volume_deltas:
                return False, f"volume_delta_percent {adelta} not in allowed {allowed_volume_deltas}"
        elif adelta != edelta:
            return False, f"volume_delta_percent expected {edelta}, got {adelta}"
        return True, ""

    if etype == "app.open":
        eapp = exp["app_id"].strip().lower()
        aapp = str(act.get("app_id") or "").strip().lower()
        if eapp != aapp:
            return False, f"app_id expected {eapp!r}, got {aapp!r}"
        return True, ""

    if etype == "media.search_play":
        estype = exp["search_type"]
        astype = act.get("search_type")
        if estype != astype:
            return False, f"search_type expected {estype!r}, got {astype!r}"

        eartist = norm_query_str(exp.get("search_artist"))
        aartist = norm_query_str(act.get("search_artist"))
        equery = norm_query_str(exp.get("search_query"))
        aquery = norm_query_str(act.get("search_query"))

        # Strip optional trailing "歌单" on playlist queries (e.g. "纯音乐歌单" == "纯音乐")
        if estype == "playlist" and aquery.endswith("歌单") and len(aquery) > 2:
            aquery_stripped = aquery[:-2]
        else:
            aquery_stripped = aquery

        acceptable_raw = exp.get("acceptable_queries") or [exp["search_query"]]
        acceptable_norm = set()
        for q in acceptable_raw:
            nq = norm_query_str(q)
            acceptable_norm.add(nq)
            if estype == "playlist" and nq.endswith("歌单") and len(nq) > 2:
                acceptable_norm.add(nq[:-2])

        if eartist:
            # Allow `search_query` to be either `"晴天"` or `"周杰伦晴天"` as long as artist matches
            acceptable_with_artist = set(acceptable_norm)
            for nq in list(acceptable_norm):
                acceptable_with_artist.add(f"{eartist}{nq}")
                acceptable_with_artist.add(f"{eartist}的{nq}")
            artist_ok = (aartist == eartist) or (not aartist and eartist in aquery)
            query_ok = (aquery in acceptable_with_artist) or (aquery_stripped in acceptable_with_artist)
            if not (artist_ok and query_ok):
                return (
                    False,
                    f"search mismatch: expected artist={exp.get('search_artist')!r}/query={exp.get('search_query')!r}, "
                    f"got artist={act.get('search_artist')!r}/query={act.get('search_query')!r}",
                )
        else:
            query_ok = (aquery in acceptable_norm) or (aquery_stripped in acceptable_norm)
            if not query_ok:
                return False, f"playlist/query expected one of {acceptable_raw!r}, got {act.get('search_query')!r}"

        return True, ""

    return True, ""


def score_sample(sample: dict[str, Any], actual_actions: list[dict[str, Any]]) -> dict[str, Any]:
    split = sample["split"]
    expected = sample["expected_actions"]

    if split == "negative":
        passed = len(actual_actions) == 0
        return {
            "id": sample["id"],
            "split": split,
            "category": sample.get("category", ""),
            "text": sample["text"],
            "passed": passed,
            "false_execution": not passed,
            "reason": "" if passed else f"false_execution: emitted {len(actual_actions)} action(s): {actual_actions}",
            "expected_actions": expected,
            "actual_actions": actual_actions,
        }

    if len(actual_actions) != len(expected):
        return {
            "id": sample["id"],
            "split": split,
            "category": sample.get("category", ""),
            "text": sample["text"],
            "passed": False,
            "false_execution": False,
            "reason": f"action_count mismatch: expected {len(expected)}, got {len(actual_actions)}",
            "expected_actions": expected,
            "actual_actions": actual_actions,
        }

    allowed_deltas = sample.get("allowed_volume_deltas")
    for idx, (e_act, a_act) in enumerate(zip(expected, actual_actions)):
        ok, detail = actions_match_single(e_act, a_act, allowed_volume_deltas=allowed_deltas)
        if not ok:
            return {
                "id": sample["id"],
                "split": split,
                "category": sample.get("category", ""),
                "text": sample["text"],
                "passed": False,
                "false_execution": False,
                "reason": f"step[{idx}] {detail}",
                "expected_actions": expected,
                "actual_actions": actual_actions,
            }

    return {
        "id": sample["id"],
        "split": split,
        "category": sample.get("category", ""),
        "text": sample["text"],
        "passed": True,
        "false_execution": False,
        "reason": "",
        "expected_actions": expected,
        "actual_actions": actual_actions,
    }


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 2)


def summarize_evaluation(
    scored_rows: list[dict[str, Any]],
    latencies_ms: list[float],
    backend_name: str,
    model_info: str,
) -> dict[str, Any]:
    pos_rows = [r for r in scored_rows if r["split"] == "positive"]
    neg_rows = [r for r in scored_rows if r["split"] == "negative"]

    pos_pass = sum(1 for r in pos_rows if r["passed"])
    neg_false_exec = sum(1 for r in neg_rows if r["false_execution"])

    pos_rate = 100.0 * pos_pass / max(1, len(pos_rows))
    neg_fe_rate = 100.0 * neg_false_exec / max(1, len(neg_rows))

    s_lat = sorted(latencies_ms)
    latency_summary = {
        "count": len(s_lat),
        "p50_ms": percentile(s_lat, 50.0),
        "p95_ms": percentile(s_lat, 95.0),
        "mean_ms": round(sum(s_lat) / max(1, len(s_lat)), 2),
        "min_ms": round(s_lat[0], 2) if s_lat else 0.0,
        "max_ms": round(s_lat[-1], 2) if s_lat else 0.0,
    }

    failures = [
        {
            "id": r["id"],
            "split": r["split"],
            "category": r["category"],
            "text": r["text"],
            "reason": r["reason"],
            "expected": r["expected_actions"],
            "actual": r["actual_actions"],
        }
        for r in scored_rows
        if not r["passed"]
    ]

    return {
        "backend": backend_name,
        "model_or_target": model_info,
        "total_samples": len(scored_rows),
        "positive_total": len(pos_rows),
        "positive_matched": pos_pass,
        "positive_exact_match_rate_percent": round(pos_rate, 2),
        "positive_target_met": pos_rate >= 95.0,
        "negative_total": len(neg_rows),
        "negative_false_executions": neg_false_exec,
        "negative_false_execution_rate_percent": round(neg_fe_rate, 2),
        "negative_target_met": neg_false_exec == 0,
        "latency_ms": latency_summary,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Backend 1: OpenAI-Compatible HTTP Router
# ---------------------------------------------------------------------------

class OpenAiHttpRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        timeout_s: float = 10.0,
        response_format_mode: str = "json_schema",
    ):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.response_format_mode = response_format_mode

    def _post_once(self, user_content: str, fmt_mode: str) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if fmt_mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": ROUTER_JSON_SCHEMA}
        elif fmt_mode == "json_object":
            body["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))

        content = resp_json["choices"][0]["message"]["content"]
        # Strip accidental markdown code fences if an open-weights server wraps JSON
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        parsed = json.loads(cleaned)
        actions = parsed.get("actions", [])
        return actions if isinstance(actions, list) else []

    def route(self, sample: dict[str, Any]) -> tuple[list[dict[str, Any]], float, str]:
        wake = sample.get("wake_word", "")
        user_payload = json.dumps(
            {
                "raw_text": sample["text"],
                "wake_word": wake,
                "origin": "keyword" if wake else "followup",
            },
            ensure_ascii=False,
        )
        t0 = time.perf_counter()
        try:
            actions = self._post_once(user_payload, self.response_format_mode)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return actions, elapsed_ms, ""
        except urllib.error.HTTPError as exc:
            # Graceful fallback from json_schema -> json_object for non-OpenAI compatible servers
            if exc.code == 400 and self.response_format_mode == "json_schema":
                self.response_format_mode = "json_object"
                t1 = time.perf_counter()
                try:
                    actions = self._post_once(user_payload, "json_object")
                    elapsed_ms = (time.perf_counter() - t1) * 1000.0
                    return actions, elapsed_ms, ""
                except Exception as exc2:
                    return [], (time.perf_counter() - t1) * 1000.0, f"http_fallback_error:{exc2}"
            return [], (time.perf_counter() - t0) * 1000.0, f"http_{exc.code}"
        except Exception as exc:
            return [], (time.perf_counter() - t0) * 1000.0, f"request_error:{exc}"


# ---------------------------------------------------------------------------
# Backend 2: C++ `voice_frontend.exe parse --mode=hybrid` CLI Driver
# ---------------------------------------------------------------------------

def parse_cpp_cli_json_line(raw_line: str) -> list[dict[str, Any]]:
    """Parses a single NDJSON line emitted by `voice_frontend.exe parse`."""
    obj = json.loads(raw_line)
    if not obj.get("ok"):
        return []
    raw_actions = obj.get("actions", [])
    if not isinstance(raw_actions, list):
        return []
    return raw_actions


def find_default_voice_frontend_exe() -> Path | None:
    candidates = [
        REPO_ROOT / "build" / "windows-x64" / "Release" / "voice_frontend.exe",
        REPO_ROOT / "build" / "core-validation" / "Release" / "voice_frontend.exe",
        REPO_ROOT / "build" / "Release" / "voice_frontend.exe",
        REPO_ROOT / "build" / "windows-x64" / "RelWithDebInfo" / "voice_frontend.exe",
        REPO_ROOT / "build" / "windows-x64" / "Debug" / "voice_frontend.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


class CppParseProcessPool:
    """Manages persistent `voice_frontend.exe parse` processes (keyed by wake_word) for exact per-utterance timing."""

    def __init__(self, exe_path: Path, parser_mode: str = "hybrid"):
        self.exe_path = exe_path
        self.parser_mode = parser_mode
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._supports_mode_flag: bool | None = None

    def _probe_mode_flag(self) -> bool:
        if self._supports_mode_flag is not None:
            return self._supports_mode_flag
        proc = subprocess.run(
            [str(self.exe_path), "parse", f"--mode={self.parser_mode}"],
            input="",
            text=True,
            encoding="utf-8",
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=10,
            check=False,
        )
        self._supports_mode_flag = (proc.returncode == 0)
        return self._supports_mode_flag

    def _get_proc(self, wake_word: str) -> subprocess.Popen[str]:
        if wake_word in self._procs and self._procs[wake_word].poll() is None:
            return self._procs[wake_word]

        cmd = [str(self.exe_path), "parse"]
        if self._probe_mode_flag():
            cmd.append(f"--mode={self.parser_mode}")
        if wake_word:
            cmd.append(f"--wake={wake_word}")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        self._procs[wake_word] = proc
        return proc

    def parse_one(self, sample: dict[str, Any]) -> tuple[list[dict[str, Any]], float, str]:
        wake = sample.get("wake_word", "")
        proc = self._get_proc(wake)
        assert proc.stdin is not None and proc.stdout is not None

        t0 = time.perf_counter()
        proc.stdin.write(sample["text"] + "\n")
        proc.stdin.flush()
        out_line = proc.stdout.readline()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not out_line:
            err_txt = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"voice_frontend.exe parse terminated unexpectedly: {err_txt.strip()}")

        return parse_cpp_cli_json_line(out_line), elapsed_ms, ""

    def close(self) -> None:
        for proc in self._procs.values():
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        self._procs.clear()


# ---------------------------------------------------------------------------
# Offline Self-Test
# ---------------------------------------------------------------------------

def run_self_test(benchmark_path: Path) -> int:
    items = load_eval_dataset(benchmark_path)
    meta = validate_eval_dataset(items)

    # 1. Verify golden 100% pass when predictions match expected_actions
    golden_rows = [score_sample(it, list(it["expected_actions"])) for it in items]
    golden_latencies = [float(320 + (i % 15) * 10) for i in range(len(items))]
    summary_gold = summarize_evaluation(golden_rows, golden_latencies, "mock_golden", "self_test")
    assert summary_gold["positive_matched"] == 40
    assert summary_gold["positive_exact_match_rate_percent"] == 100.0
    assert summary_gold["negative_false_executions"] == 0
    assert summary_gold["negative_false_execution_rate_percent"] == 0.0
    assert summary_gold["positive_target_met"] is True
    assert summary_gold["negative_target_met"] is True

    # 2. Verify flexible artist+song search_query compatibility ("周杰伦 晴天" with artist="周杰伦")
    pos_jay = next(it for it in items if it["id"] == "pos_15")
    res_jay = score_sample(
        pos_jay,
        [{"type": "media.search_play", "search_query": "周杰伦 晴天", "search_type": "song", "search_artist": "周杰伦"}],
    )
    assert res_jay["passed"], f"Expected artist+song query equivalence to pass: {res_jay}"

    # 3. Verify step-order violation & wrong app_id rejection
    pos_comp = next(it for it in items if it["id"] == "pos_34")
    res_reversed = score_sample(pos_comp, list(reversed(pos_comp["expected_actions"])))
    assert not res_reversed["passed"], "Reversed multi-step order must fail!"

    pos_clash = next(it for it in items if it["id"] == "pos_27")
    res_wrong_app = score_sample(pos_clash, [{"type": "app.open", "app_id": "steam"}])
    assert not res_wrong_app["passed"], "Wrong app_id must fail!"

    # 4. Verify negative false-execution penalty
    neg_cancel = next(it for it in items if it["id"] == "neg_14")
    res_fe = score_sample(neg_cancel, [{"type": "media.next"}])
    assert not res_fe["passed"] and res_fe["false_execution"] is True

    # 5. Verify C++ `voice_frontend.exe parse` NDJSON output parser
    cpp_ok_line = json.dumps({
        "text": "小克，打开死盯",
        "ok": True,
        "parser_version": "hybrid.v1",
        "actions": [{"type": "app.open", "app_id": "steam", "volume_delta_percent": None}],
    })
    cpp_rej_line = json.dumps({
        "text": "这首歌的旋律真好听",
        "ok": False,
        "error": {"code": "no_command", "message": "rejected"},
    })
    assert parse_cpp_cli_json_line(cpp_ok_line)[0]["app_id"] == "steam"
    assert parse_cpp_cli_json_line(cpp_rej_line) == []

    print(
        f"SELF-TEST PASSED! Benchmark verified ({meta['positive']} positive + {meta['negative']} negative = "
        f"{meta['total']} items, {meta['covered_action_types']}/14 ActionTypes, apps={meta['covered_apps']}), "
        f"scorer & C++ NDJSON parser verified."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BENCHMARK_PATH,
                        help=f"path to evaluation JSONL (default: {DEFAULT_BENCHMARK_PATH})")
    parser.add_argument("--backend", choices=["openai", "cpp"], default="openai",
                        help="evaluation backend: direct OpenAI-compatible HTTP (`openai`) or `voice_frontend.exe parse` (`cpp`)")
    parser.add_argument("--base-url", type=str,
                        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        help="OpenAI-compatible API base URL (default: env OPENAI_BASE_URL or https://api.openai.com/v1)")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="environment variable holding API key (default: OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="router model name for `--backend openai` (default: gpt-4o-mini)")
    parser.add_argument("--response-format", choices=["json_schema", "json_object"], default="json_schema")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="number of worker threads for `--backend openai` (1 = pure serial latency)")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request HTTP timeout in seconds")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "local.toml",
                        help="optional local.toml to load custom [apps.entries.*] aliases")
    parser.add_argument("--exe", type=Path, default=None,
                        help="path to compiled voice_frontend.exe for `--backend cpp`")
    parser.add_argument("--parser-mode", choices=["hybrid", "router", "rules"], default="hybrid",
                        help="mode passed to `voice_frontend.exe parse --mode=<mode>`")
    parser.add_argument("--out", type=Path, default=None, help="optional path to save full JSON evaluation report")
    parser.add_argument("--enforce-gates", action="store_true",
                        help="return exit code 1 if positive match < 95%% or negative false execution > 0%%")
    parser.add_argument("--self-test", action="store_true",
                        help="run offline benchmark & scorer self-test without network/API keys and exit")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.dataset)

    items = load_eval_dataset(args.dataset)
    validate_eval_dataset(items)

    scored_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    if args.backend == "openai":
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            print(
                f"Error: environment variable {args.api_key_env} is not set.\n"
                f"Set {args.api_key_env} for live HTTP evaluation, or run `--self-test` / `--backend cpp`.",
                file=sys.stderr,
            )
            return 2

        app_catalog = load_app_catalog(args.config)
        sys_prompt = build_router_system_prompt(app_catalog)
        client = OpenAiHttpRouterClient(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=sys_prompt,
            timeout_s=args.timeout,
            response_format_mode=args.response_format,
        )

        workers = max(1, args.concurrency)
        if workers == 1:
            for it in items:
                acts, lat_ms, err = client.route(it)
                row = score_sample(it, acts)
                if err:
                    row["reason"] = f"{err} | {row['reason']}".strip(" |")
                row["latency_ms"] = round(lat_ms, 2)
                scored_rows.append(row)
                latencies_ms.append(lat_ms)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(client.route, it) for it in items]
                for it, fut in zip(items, futs):
                    acts, lat_ms, err = fut.result()
                    row = score_sample(it, acts)
                    if err:
                        row["reason"] = f"{err} | {row['reason']}".strip(" |")
                    row["latency_ms"] = round(lat_ms, 2)
                    scored_rows.append(row)
                    latencies_ms.append(lat_ms)

        summary = summarize_evaluation(scored_rows, latencies_ms, "openai_compatible_http", args.model)

    else:
        exe = args.exe or find_default_voice_frontend_exe()
        if exe is None or not exe.is_file():
            print(
                "Error: could not locate compiled `voice_frontend.exe`. Pass `--exe <path/to/voice_frontend.exe>`.",
                file=sys.stderr,
            )
            return 2

        runner = CppParseProcessPool(exe_path=exe, parser_mode=args.parser_mode)
        try:
            for it in items:
                acts, lat_ms, err = runner.parse_one(it)
                row = score_sample(it, acts)
                if err:
                    row["reason"] = f"{err} | {row['reason']}".strip(" |")
                row["latency_ms"] = round(lat_ms, 2)
                scored_rows.append(row)
                latencies_ms.append(lat_ms)
        finally:
            runner.close()

        summary = summarize_evaluation(
            scored_rows,
            latencies_ms,
            "cpp_voice_frontend_parse",
            f"{exe.name} (--mode={args.parser_mode})",
        )

    print("=== M2 Natural-Language Router Benchmark Summary ===")
    print(f"Backend               : {summary['backend']} ({summary['model_or_target']})")
    print(
        f"Positive Exact Match  : {summary['positive_matched']}/{summary['positive_total']} "
        f"({summary['positive_exact_match_rate_percent']:.2f}%)  [target >= 95.0% -> "
        f"{'PASS' if summary['positive_target_met'] else 'FAIL'}]"
    )
    print(
        f"Negative False Exec   : {summary['negative_false_executions']}/{summary['negative_total']} "
        f"({summary['negative_false_execution_rate_percent']:.2f}%)  [target == 0.0%  -> "
        f"{'PASS' if summary['negative_target_met'] else 'FAIL'}]"
    )
    lat = summary["latency_ms"]
    print(
        f"Latency (ms)          : P50={lat['p50_ms']:.1f} ms | P95={lat['p95_ms']:.1f} ms | "
        f"Mean={lat['mean_ms']:.1f} ms | Max={lat['max_ms']:.1f} ms"
    )

    if summary["failures"]:
        print(f"\n--- Mismatches / Failures ({len(summary['failures'])}) ---")
        for fail in summary["failures"]:
            print(f"  [{fail['id']}] ({fail['split']}) {fail['text']!r} -> {fail['reason']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        full_report = {**summary, "samples": scored_rows}
        args.out.write_text(json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved detailed report -> {args.out}")

    if args.enforce_gates and not (summary["positive_target_met"] and summary["negative_target_met"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
