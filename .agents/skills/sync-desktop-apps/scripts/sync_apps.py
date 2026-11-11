#!/usr/bin/env python3
"""Scan Windows desktop applications, QuickStart shortcuts, and losslessly sync [apps] in config/local.toml.

Capabilities:
  1. --discover-quickstart : Auto-probe likely QuickStart / shortcut launcher folders on local drives,
                             Desktop, Taskbar Pinned, and launcher directories, listing .lnk / .exe items.
  2. --scan / --search     : Scan specified QuickStart dirs, Windows Start Menu Programs, Registry App Paths,
                             common install roots (Program Files, %LOCALAPPDATA%, etc.), and live processes;
                             safely resolve .lnk TargetPath & Arguments via PowerShell WScript.Shell while
                             filtering out uninstallers/updaters/crashpad helpers.
  3. --write-local-toml    : Losslessly create or update the `[apps]` and `[apps.entries.<id>]` sections in
                             `config/local.toml` (100% aligned with C++ AppConfig schema) without touching or
                             corrupting any other sections (`[kws]`, `[asr]`, `[preprocess]`, comments, etc.).
  4. --self-test           : Offline unit test verifying helper filtering, alias enrichment, and lossless
                             multi-pass TOML read-modify-write on a temporary config file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
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


# Auxiliary / non-primary executables that must be excluded during scanning
EXCLUDED_STEM_RE = re.compile(
    r"^(?:"
    r"unins\d*|uninstall.*|.*uninstaller.*|"
    r"updater.*|.*updater|auto_?update.*|liveupdate.*|"
    r"crashpad.*|crashreporter.*|.*crashhandler.*|errorreporter.*|bugreport.*|writeminidump.*|"
    r"setup.*|.*setup|install.*|.*installer.*|"
    r"vc_redist.*|dxsetup.*|dotnet.*|"
    r"elevation_service|notification_helper|chrome_pwa_launcher|"
    r"squirrel|unitycrashhandler.*|cefsharp\.browsersubprocess|"
    r"gameoverlayui.*|steamerrorreporter.*|steamservice.*|streaming_client.*|"
    r".*helper|.*daemon|.*-service|.*_service"
    r")$",
    re.IGNORECASE,
)

# Curated presets with colloquial names and Mandarin ASR homophones for popular desktop apps
BUILTIN_APP_PRESETS: dict[str, dict[str, Any]] = {
    "steam": {
        "display_name": "Steam",
        "exe_names": ["steam.exe"],
        "keywords": ["steam", "蒸汽平台"],
        "aliases": ["steam", "蒸汽平台", "死盯", "斯蒂姆", "游戏平台"],
        "process_names": ["steam.exe"],
    },
    "clash": {
        "display_name": "Clash",
        "exe_names": [
            "clash-verge.exe",
            "clash for windows.exe",
            "clash.for.windows.exe",
            "cfw.exe",
            "clashnyanpasu.exe",
            "clash-party.exe",
            "flclash.exe",
            "mihomo-party.exe",
        ],
        "keywords": ["clash", "verge", "nyanpasu", "flclash", "mihomo"],
        "aliases": ["clash", "代理软件", "可拉什", "克拉什", "小猫咪"],
        "process_names": [],
    },
    "discord": {
        "display_name": "Discord",
        "exe_names": ["discord.exe"],
        "keywords": ["discord"],
        "aliases": ["discord", "语音软件", "迪斯科", "迪斯扣", "开黑语音"],
        "process_names": ["Discord.exe"],
    },
    "qq": {
        "display_name": "QQ",
        "exe_names": ["qq.exe"],
        "keywords": ["qq", "tencent\\qq", "qqnt"],
        "aliases": ["qq", "腾讯qq", "扣扣", "企鹅"],
        "process_names": ["QQ.exe"],
    },
    "powershell": {
        "display_name": "PowerShell",
        "exe_names": ["pwsh.exe", "powershell.exe"],
        "keywords": ["powershell", "pwsh"],
        "aliases": ["powershell", "终端", "命令行", "泡我秀", "帕瓦谢尔"],
        "process_names": ["pwsh.exe", "powershell.exe"],
    },
    "vscode": {
        "display_name": "VS Code",
        "exe_names": ["code.exe"],
        "keywords": ["vscode", "visual studio code", "microsoft vs code"],
        "aliases": ["vscode", "vs code", "代码编辑器", "写代码软件", "维埃斯扣德"],
        "process_names": ["Code.exe"],
    },
    "wechat": {
        "display_name": "微信",
        "exe_names": ["weixin.exe", "wechat.exe"],
        "keywords": ["wechat", "weixin", "微信"],
        "aliases": ["微信", "wechat", "打开微信", "威信"],
        "process_names": ["Weixin.exe", "WeChat.exe"],
    },
    "netease": {
        "display_name": "网易云音乐",
        "exe_names": ["cloudmusic.exe"],
        "keywords": ["cloudmusic", "netease", "网易云", "网易云音乐"],
        "aliases": ["网易云音乐", "网易云", "音乐播放器", "云音乐"],
        "process_names": ["cloudmusic.exe"],
    },
    "chrome": {
        "display_name": "Google Chrome",
        "exe_names": ["chrome.exe"],
        "keywords": ["chrome", "google chrome", "谷歌浏览器"],
        "aliases": ["chrome", "浏览器", "谷歌浏览器", "克罗姆"],
        "process_names": ["chrome.exe"],
    },
    "edge": {
        "display_name": "Microsoft Edge",
        "exe_names": ["msedge.exe"],
        "keywords": ["msedge", "edge", "微软浏览器"],
        "aliases": ["edge", "微软浏览器", "浏览器", "艾奇"],
        "process_names": ["msedge.exe"],
    },
    "wt": {
        "display_name": "Windows Terminal",
        "exe_names": ["wt.exe", "windowsterminal.exe"],
        "keywords": ["windows terminal", "wt"],
        "aliases": ["terminal", "windows终端", "系统终端"],
        "process_names": ["WindowsTerminal.exe"],
    },
    "taskmgr": {
        "display_name": "任务管理器",
        "exe_names": ["taskmgr.exe"],
        "keywords": ["taskmgr", "任务管理器", "task manager"],
        "aliases": ["任务管理器", "进程管理器"],
        "process_names": ["Taskmgr.exe"],
    },
}


@dataclass
class AppCandidate:
    app_id: str
    display_name: str
    target: str
    arguments: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    process_names: list[str] = field(default_factory=list)
    source: str = ""
    shortcut_path: str = ""
    score: int = 0

    def to_entry_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "target": self.target,
            "arguments": list(self.arguments),
            "aliases": dedupe_preserve_order(self.aliases),
            "process_names": dedupe_preserve_order(self.process_names),
        }


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = item.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def sanitize_app_id(raw: str) -> str:
    """Normalizes an identifier into a clean TOML bare key `[a-z0-9_]+`."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "app"
    if cleaned[0].isdigit():
        cleaned = f"app_{cleaned}"
    return cleaned


def split_windows_arguments(arg_str: str) -> list[str]:
    arg_str = (arg_str or "").strip()
    if not arg_str:
        return []
    try:
        return shlex.split(arg_str, posix=False)
    except ValueError:
        return [p for p in arg_str.split() if p]


def is_excluded_executable(exe_path: Path, arguments: list[str] | None = None) -> bool:
    """Returns True if exe_path is an uninstaller, updater, crashpad helper, or non-exe."""
    if exe_path.suffix.lower() != ".exe":
        return True
    stem = exe_path.stem.lower()
    # Special case: Electron Squirrel Update.exe with `--processStart RealApp.exe` (e.g. Discord)
    if stem == "update":
        args = arguments or []
        for idx, tok in enumerate(args):
            if tok.lower() == "--processstart" and idx + 1 < len(args):
                inner_stem = Path(args[idx + 1].strip('"')).stem.lower()
                if inner_stem and not EXCLUDED_STEM_RE.match(inner_stem):
                    return False
        return True
    return bool(EXCLUDED_STEM_RE.match(stem))


def extract_process_names(target_path: Path, arguments: list[str]) -> list[str]:
    """Determines the actual running process image name(s) for an app entry."""
    if target_path.stem.lower() == "update":
        for idx, tok in enumerate(arguments):
            if tok.lower() == "--processstart" and idx + 1 < len(arguments):
                proc = Path(arguments[idx + 1].strip('"')).name
                if proc:
                    return [proc if proc.lower().endswith(".exe") else f"{proc}.exe"]
    return [target_path.name]


def match_preset_for_app(
    shortcut_stem: str,
    target_path: Path,
    arguments: list[str],
    from_shortcut: bool = False,
) -> tuple[str | None, dict[str, Any] | None]:
    exe_lower = target_path.name.lower()
    full_lower = str(target_path).lower()
    stem_lower = shortcut_stem.lower()
    proc_names = [p.lower() for p in extract_process_names(target_path, arguments)]

    for preset_id, preset in BUILTIN_APP_PRESETS.items():
        preset_exes = [e.lower() for e in preset["exe_names"]]
        if exe_lower in preset_exes or any(p in preset_exes for p in proc_names):
            return preset_id, preset
        if preset_id == "clash" and ("clash" in exe_lower or (from_shortcut and "clash" in stem_lower)):
            return preset_id, preset
        if from_shortcut and (stem_lower == preset_id or stem_lower == preset["display_name"].lower()):
            return preset_id, preset
    return None, None


def build_candidate(
    target_path: Path,
    arguments: list[str] | None = None,
    display_hint: str = "",
    source: str = "",
    shortcut_path: str = "",
    base_score: int = 50,
) -> AppCandidate | None:
    args = [a.strip('"') for a in (arguments or []) if a.strip()]
    if is_excluded_executable(target_path, args):
        return None

    preset_id, preset = match_preset_for_app(
        display_hint or target_path.stem,
        target_path,
        args,
        from_shortcut=bool(shortcut_path),
    )
    proc_names = extract_process_names(target_path, args)

    if preset_id and preset:
        app_id = preset_id
        display_name = preset["display_name"]
        aliases = list(preset["aliases"])
        if display_hint and display_hint.lower() not in [a.lower() for a in aliases]:
            aliases.append(display_hint)
        if preset["process_names"]:
            proc_names = dedupe_preserve_order(proc_names + list(preset["process_names"]))
        score = base_score + 20
    else:
        raw_name = (display_hint or target_path.stem).strip()
        # Clean trailing " - Shortcut" / " - 快捷方式"
        raw_name = re.sub(r"\s*-\s*(?:Shortcut|快捷方式)$", "", raw_name, flags=re.IGNORECASE).strip()
        app_id = sanitize_app_id(raw_name if re.search(r"[a-zA-Z0-9]", raw_name) else target_path.stem)
        display_name = raw_name or target_path.stem
        aliases = dedupe_preserve_order([display_name.lower(), target_path.stem.lower()])
        score = base_score

    return AppCandidate(
        app_id=app_id,
        display_name=display_name,
        target=str(target_path),
        arguments=args,
        aliases=dedupe_preserve_order(aliases),
        process_names=dedupe_preserve_order(proc_names),
        source=source,
        shortcut_path=shortcut_path,
        score=score,
    )


def resolve_lnk_batch(lnk_paths: list[Path]) -> dict[str, dict[str, str]]:
    """Safely resolves a batch of Windows .lnk files in a single PowerShell invocation."""
    if not lnk_paths or os.name != "nt":
        return {}

    ps_script = r"""
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) { Write-Output "{}"; exit 0 }
$paths = $raw | ConvertFrom-Json
$sh = New-Object -ComObject WScript.Shell
$result = @{}
foreach ($p in $paths) {
    try {
        if (Test-Path -LiteralPath $p) {
            $sc = $sh.CreateShortcut($p)
            $result[$p] = @{
                TargetPath = [string]$sc.TargetPath
                Arguments = [string]$sc.Arguments
                WorkingDirectory = [string]$sc.WorkingDirectory
                Description = [string]$sc.Description
            }
        }
    } catch {
        # Skip unreadable shortcut safely
    }
}
$result | ConvertTo-Json -Compress -Depth 4
"""
    payload = json.dumps([str(p) for p in lnk_paths], ensure_ascii=False)
    for shell_bin in ("pwsh", "powershell"):
        try:
            proc = subprocess.run(
                [shell_bin, "-NoProfile", "-NonInteractive", "-Command", ps_script],
                input=payload,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=20,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout.strip())
                if isinstance(data, dict):
                    return data
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue
    return {}


def discover_quickstart_dirs() -> list[dict[str, Any]]:
    """Probes local drives and standard user locations for QuickStart / shortcut folders."""
    candidates: list[tuple[Path, str]] = []
    seen_resolved: set[str] = set()

    folder_names = [
        "QuickStart",
        "quickstart",
        "QuickLaunch",
        "Shortcuts",
        "快捷启动",
        "快捷方式",
        "常用软件",
    ]
    parent_subdirs = ["", "Tools", "Software", "Apps", "Program", "Programs"]

    # 1. Local drive roots C: .. Z:
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:\\")
        if not drive.exists():
            continue
        for sub in parent_subdirs:
            base = drive / sub if sub else drive
            if not base.is_dir():
                continue
            for fn in folder_names:
                p = base / fn
                if p.is_dir():
                    candidates.append((p, "drive_quickstart"))

    # 2. User profile / Desktop / Taskbar pinned / third-party launcher dirs
    env_dirs: list[tuple[str, str]] = [
        (r"%USERPROFILE%\QuickStart", "user_quickstart"),
        (r"%USERPROFILE%\Desktop\QuickStart", "desktop_quickstart"),
        (r"%OneDrive%\Desktop\QuickStart", "onedrive_quickstart"),
        (r"%APPDATA%\Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar", "taskbar_pinned"),
        (r"%APPDATA%\Microsoft\Internet Explorer\Quick Launch", "quick_launch"),
        (r"%USERPROFILE%\Desktop", "user_desktop"),
        (r"%OneDrive%\Desktop", "onedrive_desktop"),
        (r"%PUBLIC%\Desktop", "public_desktop"),
        (r"%LOCALAPPDATA%\Quicker", "launcher_quicker"),
        (r"%APPDATA%\Rolan", "launcher_rolan"),
    ]
    for raw_env, kind in env_dirs:
        expanded = os.path.expandvars(raw_env)
        if "%" in expanded:
            continue
        p = Path(expanded)
        if p.is_dir():
            candidates.append((p, kind))

    results: list[dict[str, Any]] = []
    for folder, kind in candidates:
        try:
            norm_key = str(folder.resolve()).lower()
        except OSError:
            norm_key = str(folder).lower()
        if norm_key in seen_resolved:
            continue
        seen_resolved.add(norm_key)

        lnks: list[Path] = []
        exes: list[Path] = []
        try:
            for child in folder.iterdir():
                if child.is_file():
                    suf = child.suffix.lower()
                    if suf == ".lnk":
                        lnks.append(child)
                    elif suf == ".exe" and not is_excluded_executable(child):
                        exes.append(child)
        except OSError:
            continue

        if not lnks and not exes and "quickstart" not in kind:
            continue

        sample_items = [p.name for p in sorted(lnks)[:12]] + [p.name for p in sorted(exes)[:8]]
        results.append({
            "path": str(folder),
            "kind": kind,
            "lnk_count": len(lnks),
            "exe_count": len(exes),
            "sample_items": sample_items[:15],
        })

    # Prioritize explicit QuickStart folders first, then TaskBar pinned, then Desktop
    priority = {
        "drive_quickstart": 0,
        "user_quickstart": 1,
        "desktop_quickstart": 2,
        "onedrive_quickstart": 3,
        "taskbar_pinned": 4,
        "quick_launch": 5,
        "user_desktop": 6,
        "onedrive_desktop": 7,
        "public_desktop": 8,
    }
    results.sort(key=lambda r: (priority.get(r["kind"], 99), -r["lnk_count"] - r["exe_count"], r["path"]))
    return results


def scan_shortcut_directory(folder: Path, source_label: str, max_depth: int = 3, base_score: int = 90) -> list[AppCandidate]:
    if not folder.is_dir():
        return []

    lnk_files: list[Path] = []
    exe_files: list[Path] = []

    def _walk(curr: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for entry in curr.iterdir():
                if entry.is_dir():
                    _walk(entry, depth + 1)
                elif entry.is_file():
                    suf = entry.suffix.lower()
                    if suf == ".lnk":
                        lnk_files.append(entry)
                    elif suf == ".exe":
                        exe_files.append(entry)
        except OSError:
            return

    _walk(folder, 1)
    resolved_map = resolve_lnk_batch(lnk_files)
    found: list[AppCandidate] = []

    for lnk in lnk_files:
        info = resolved_map.get(str(lnk))
        if not info:
            continue
        tpath_str = (info.get("TargetPath") or "").strip()
        if not tpath_str:
            continue
        tpath = Path(os.path.expandvars(tpath_str))
        if not tpath.is_file():
            continue
        args = split_windows_arguments(info.get("Arguments", ""))
        cand = build_candidate(
            target_path=tpath,
            arguments=args,
            display_hint=lnk.stem,
            source=source_label,
            shortcut_path=str(lnk),
            base_score=base_score,
        )
        if cand:
            found.append(cand)

    for exe in exe_files:
        if not exe.is_file():
            continue
        cand = build_candidate(
            target_path=exe,
            arguments=[],
            display_hint=exe.stem,
            source=source_label,
            base_score=base_score - 5,
        )
        if cand:
            found.append(cand)

    return found


def scan_registry_apps() -> list[AppCandidate]:
    if os.name != "nt":
        return []
    import winreg

    found: list[AppCandidate] = []
    app_paths_roots = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\App Paths"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\App Paths"),
    ]
    for hive, subkey_path in app_paths_roots:
        try:
            with winreg.OpenKey(hive, subkey_path) as root_key:
                num_subkeys, _, _ = winreg.QueryInfoKey(root_key)
                for idx in range(num_subkeys):
                    try:
                        sub_name = winreg.EnumKey(root_key, idx)
                        with winreg.OpenKey(root_key, sub_name) as sk:
                            val, _ = winreg.QueryValueEx(sk, "")
                            if isinstance(val, str) and val.strip():
                                raw = val.strip().strip('"')
                                p = Path(os.path.expandvars(raw))
                                if p.is_file():
                                    cand = build_candidate(
                                        target_path=p,
                                        display_hint=Path(sub_name).stem,
                                        source="registry_app_paths",
                                        base_score=75,
                                    )
                                    if cand:
                                        found.append(cand)
                    except OSError:
                        continue
        except OSError:
            continue
    return found


def scan_common_install_dirs() -> list[AppCandidate]:
    if os.name != "nt":
        return []
    found: list[AppCandidate] = []

    # Check well-known direct paths first (instantaneous)
    direct_probes = [
        (r"%ProgramFiles(x86)%\Steam\steam.exe", "Steam"),
        (r"%ProgramFiles%\Steam\steam.exe", "Steam"),
        (r"%ProgramFiles%\PowerShell\7\pwsh.exe", "PowerShell"),
        (r"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe", "PowerShell"),
        (r"%SystemRoot%\System32\Taskmgr.exe", "任务管理器"),
        (r"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe", "Windows Terminal"),
        (r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe", "VS Code"),
        (r"%ProgramFiles%\Microsoft VS Code\Code.exe", "VS Code"),
        (r"%ProgramFiles%\Tencent\QQNT\QQ.exe", "QQ"),
        (r"%ProgramFiles(x86)%\Tencent\QQNT\QQ.exe", "QQ"),
        (r"%ProgramFiles%\Tencent\Weixin\Weixin.exe", "微信"),
        (r"%ProgramFiles%\Tencent\WeChat\WeChat.exe", "微信"),
        (r"%ProgramFiles(x86)%\NetEase\CloudMusic\cloudmusic.exe", "网易云音乐"),
        (r"%ProgramFiles%\NetEase\CloudMusic\cloudmusic.exe", "网易云音乐"),
    ]
    for raw_p, hint in direct_probes:
        expanded = os.path.expandvars(raw_p)
        if "%" in expanded:
            continue
        p = Path(expanded)
        if p.is_file():
            cand = build_candidate(p, display_hint=hint, source="known_path", base_score=82)
            if cand:
                found.append(cand)

    # Check Discord Squirrel pattern: %LOCALAPPDATA%\Discord\Update.exe --processStart Discord.exe
    discord_dir = Path(os.path.expandvars(r"%LOCALAPPDATA%\Discord"))
    if (discord_dir / "Update.exe").is_file() and list(discord_dir.glob("app-*/Discord.exe")):
        cand = build_candidate(
            discord_dir / "Update.exe",
            arguments=["--processStart", "Discord.exe"],
            display_hint="Discord",
            source="known_path",
            base_score=85,
        )
        if cand:
            found.append(cand)

    # Shallow scan (depth <= 2) of common install roots
    install_roots = [
        Path(os.path.expandvars(r"%ProgramFiles%")),
        Path(os.path.expandvars(r"%ProgramFiles(x86)%")),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs")),
        Path(os.path.expandvars(r"%LOCALAPPDATA%")),
    ]
    # Also check D:\Program Files, D:\Software, E:\Software if present
    for letter in ("D", "E", "F"):
        for sub in ("Program Files", "Program Files (x86)", "Software", "Apps", "Games", "Steam"):
            p = Path(f"{letter}:\\{sub}")
            if p.is_dir():
                install_roots.append(p)

    for root in install_roots:
        if not root.is_dir():
            continue
        try:
            for sub in root.iterdir():
                if not sub.is_dir():
                    if sub.is_file() and sub.suffix.lower() == ".exe" and root.name.lower() == "steam":
                        cand = build_candidate(sub, display_hint=sub.stem, source="install_dir", base_score=70)
                        if cand:
                            found.append(cand)
                    continue
                # Skip noisy system/cache directories in %LOCALAPPDATA%
                if sub.name.lower() in {"temp", "packages", "microsoft", "google", "cache", "pip", "npm-cache"}:
                    if sub.name.lower() == "google":
                        chrome_p = sub / r"Chrome\Application\chrome.exe"
                        if chrome_p.is_file():
                            cand = build_candidate(chrome_p, display_hint="Google Chrome", source="install_dir", base_score=78)
                            if cand:
                                found.append(cand)
                    continue
                try:
                    for child in sub.iterdir():
                        if child.is_file() and child.suffix.lower() == ".exe":
                            cand = build_candidate(child, display_hint=child.stem, source="install_dir", base_score=65)
                            if cand:
                                found.append(cand)
                except OSError:
                    continue
        except OSError:
            continue

    return found


def scan_running_processes() -> list[AppCandidate]:
    if os.name != "nt":
        return []
    ps_script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-Process | Where-Object { $_.Path } | Select-Object -Unique Name, Path | ConvertTo-Json -Compress
"""
    for shell_bin in ("pwsh", "powershell"):
        try:
            proc = subprocess.run(
                [shell_bin, "-NoProfile", "-NonInteractive", "-Command", ps_script],
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=12,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                items = json.loads(proc.stdout.strip())
                if isinstance(items, dict):
                    items = [items]
                out: list[AppCandidate] = []
                for it in items:
                    p_str = (it.get("Path") or "").strip()
                    if not p_str:
                        continue
                    p = Path(p_str)
                    # Skip Windows system background processes unless in preset
                    if "\\windows\\system32\\" in p_str.lower() and p.name.lower() not in {"powershell.exe", "taskmgr.exe", "cmd.exe", "notepad.exe", "calc.exe"}:
                        continue
                    if "\\windows\\systemapps\\" in p_str.lower():
                        continue
                    cand = build_candidate(p, display_hint=it.get("Name", p.stem), source="running_process", base_score=72)
                    if cand:
                        out.append(cand)
                return out
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue
    return []


def collect_all_candidates(quickstart_dirs: list[Path] | None = None, include_system: bool = True) -> list[AppCandidate]:
    raw_list: list[AppCandidate] = []

    for qdir in (quickstart_dirs or []):
        raw_list.extend(scan_shortcut_directory(qdir, source_label=f"quickstart:{qdir}", max_depth=2, base_score=100))

    if include_system:
        start_menus = [
            Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs")),
            Path(os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs")),
            Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
            Path(os.path.expandvars(r"%PUBLIC%\Desktop")),
        ]
        for sm in start_menus:
            if sm.is_dir():
                raw_list.extend(scan_shortcut_directory(sm, source_label="start_menu_or_desktop", max_depth=3, base_score=85))
        raw_list.extend(scan_common_install_dirs())
        raw_list.extend(scan_registry_apps())
        raw_list.extend(scan_running_processes())

    # Deduplicate by normalized target + arguments, keeping highest score and merging aliases
    by_target: dict[tuple[str, tuple[str, ...]], AppCandidate] = {}
    for cand in raw_list:
        key = (os.path.normcase(os.path.normpath(cand.target)), tuple(cand.arguments))
        existing = by_target.get(key)
        if existing is None:
            by_target[key] = cand
        else:
            merged_aliases = dedupe_preserve_order(
                (cand.aliases + existing.aliases) if cand.score > existing.score else (existing.aliases + cand.aliases)
            )
            merged_procs = dedupe_preserve_order(existing.process_names + cand.process_names)
            if cand.score > existing.score:
                cand.aliases = merged_aliases
                cand.process_names = merged_procs
                by_target[key] = cand
            else:
                existing.aliases = merged_aliases
                existing.process_names = merged_procs

    results = list(by_target.values())
    results.sort(key=lambda c: (-c.score, c.app_id, c.target))
    return results


def search_candidates(candidates: list[AppCandidate], keyword: str) -> list[AppCandidate]:
    kw = keyword.strip().lower()
    if not kw:
        return candidates

    matched: list[tuple[int, AppCandidate]] = []
    for cand in candidates:
        haystacks_exact = [
            cand.app_id.lower(),
            cand.display_name.lower(),
            Path(cand.target).stem.lower(),
            Path(cand.target).name.lower(),
            *[a.lower() for a in cand.aliases],
        ]
        if kw in haystacks_exact:
            matched.append((cand.score + 100, cand))
            continue
        if any(kw in h for h in haystacks_exact) or kw in cand.target.lower():
            matched.append((cand.score + 40, cand))

    matched.sort(key=lambda pair: (-pair[0], pair[1].app_id))
    return [c for _, c in matched]


# ---------------------------------------------------------------------------
# Lossless TOML [apps] Section Formatter & Merger
# ---------------------------------------------------------------------------

TOML_SECTION_HEADER_RE = re.compile(r"^\s*\[(\[?)([^\[\]]+)\]\1\s*(?:#.*)?$")


def toml_quote_str(value: str) -> str:
    """Formats a Python string as a valid TOML basic double-quoted string."""
    return json.dumps(value, ensure_ascii=False)


def toml_format_str_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_quote_str(v) for v in values) + "]"


def render_apps_toml_block(apps_data: dict[str, Any]) -> str:
    """Renders `[apps]` and `[apps.entries.<id>]` in the exact C++ AppConfig TOML format."""
    quickstart_dirs = [str(d) for d in apps_data.get("quickstart_dirs", [])]
    scan_system_apps = bool(apps_data.get("scan_system_apps", True))
    entries: dict[str, dict[str, Any]] = dict(apps_data.get("entries", {}))

    lines: list[str] = [
        "[apps]",
        f"quickstart_dirs = {toml_format_str_list(quickstart_dirs)}",
        f"scan_system_apps = {'true' if scan_system_apps else 'false'}",
    ]

    for entry_id in entries:
        entry = entries[entry_id]
        safe_id = sanitize_app_id(entry_id) if not re.fullmatch(r"[A-Za-z0-9_-]+", entry_id) else entry_id
        display_name = str(entry.get("display_name", entry_id))
        target = str(entry.get("target", ""))
        arguments = [str(x) for x in entry.get("arguments", [])]
        aliases = [str(x) for x in entry.get("aliases", [])]
        process_names = [str(x) for x in entry.get("process_names", [])]

        lines.extend([
            "",
            f"[apps.entries.{safe_id}]",
            f"display_name = {toml_quote_str(display_name)}",
            f"target = {toml_quote_str(target)}",
            f"arguments = {toml_format_str_list(arguments)}",
            f"aliases = {toml_format_str_list(aliases)}",
            f"process_names = {toml_format_str_list(process_names)}",
        ])

    return "\n".join(lines) + "\n"


def parse_toml_section_root_key(header_inner: str) -> str:
    """Extracts the first path segment of a TOML `[section.sub]` header."""
    first = header_inner.strip().split(".", 1)[0].strip()
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1]
    return first


def update_local_toml_apps_lossless(
    config_path: Path,
    new_entries: dict[str, dict[str, Any]] | None = None,
    quickstart_dirs: list[str] | None = None,
    scan_system_apps: bool | None = None,
    replace_entries: bool = False,
) -> dict[str, Any]:
    """Reads `config_path` (if present), losslessly updates `[apps]`, verifies AST equality of all other sections, and writes atomically."""
    if config_path.is_file():
        orig_raw = config_path.read_text(encoding="utf-8")
        orig_parsed = tomllib.loads(orig_raw)
    else:
        orig_raw = ""
        orig_parsed = {}

    existing_apps = dict(orig_parsed.get("apps", {}))
    merged_quickstart = list(existing_apps.get("quickstart_dirs", []))
    if quickstart_dirs is not None:
        merged_quickstart = dedupe_preserve_order(merged_quickstart + list(quickstart_dirs))

    merged_scan = existing_apps.get("scan_system_apps", True) if scan_system_apps is None else bool(scan_system_apps)

    merged_entries: dict[str, dict[str, Any]] = {} if replace_entries else dict(existing_apps.get("entries", {}))
    for app_id, entry_payload in (new_entries or {}).items():
        clean_id = sanitize_app_id(app_id)
        prev = merged_entries.get(clean_id, {})
        merged_aliases = dedupe_preserve_order(
            list(entry_payload.get("aliases", [])) + list(prev.get("aliases", []))
        )
        merged_procs = dedupe_preserve_order(
            list(entry_payload.get("process_names", [])) + list(prev.get("process_names", []))
        )
        merged_entries[clean_id] = {
            "display_name": entry_payload.get("display_name") or prev.get("display_name") or clean_id,
            "target": entry_payload.get("target") or prev.get("target") or "",
            "arguments": list(entry_payload.get("arguments", prev.get("arguments", []))),
            "aliases": merged_aliases,
            "process_names": merged_procs,
        }

    merged_apps_obj = {
        "quickstart_dirs": merged_quickstart,
        "scan_system_apps": merged_scan,
        "entries": merged_entries,
    }
    rendered_apps = render_apps_toml_block(merged_apps_obj)

    # Strip existing [apps] and [apps.*] blocks line-by-line while keeping every other line verbatim
    kept_before: list[str] = []
    kept_after: list[str] = []
    in_apps_section = False
    saw_apps_section = False

    for line in orig_raw.splitlines(keepends=True):
        m = TOML_SECTION_HEADER_RE.match(line)
        if m:
            root_key = parse_toml_section_root_key(m.group(2))
            if root_key == "apps":
                in_apps_section = True
                saw_apps_section = True
                continue
            else:
                in_apps_section = False

        if not in_apps_section:
            if not saw_apps_section:
                kept_before.append(line)
            else:
                kept_after.append(line)

    prefix_txt = "".join(kept_before).rstrip("\r\n")
    suffix_txt = "".join(kept_after).lstrip("\r\n")

    parts: list[str] = []
    if prefix_txt:
        parts.append(prefix_txt)
    parts.append(rendered_apps.rstrip("\r\n"))
    if suffix_txt:
        parts.append(suffix_txt.rstrip("\r\n"))
    final_text = "\n\n".join(parts) + "\n"

    # Strict AST verification before touching disk: every non-`apps` top-level key must match 100%
    verified_parsed = tomllib.loads(final_text)
    orig_non_apps = {k: v for k, v in orig_parsed.items() if k != "apps"}
    new_non_apps = {k: v for k, v in verified_parsed.items() if k != "apps"}
    if orig_non_apps != new_non_apps:
        raise RuntimeError("Safety abort: non-[apps] TOML configuration sections changed unexpectedly!")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_path.write_text(final_text, encoding="utf-8", newline="\n")
        os.replace(tmp_path, config_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return verified_parsed["apps"]


def run_self_test() -> int:
    """Runs offline unit tests for helper filtering, Discord Squirrel handling, and lossless TOML updates."""
    # 1. Helper / uninstaller exclusion checks
    assert is_excluded_executable(Path(r"C:\Apps\Foo\unins000.exe"))
    assert is_excluded_executable(Path(r"C:\Apps\Foo\Uninstall Bar.exe"))
    assert is_excluded_executable(Path(r"C:\Apps\Foo\crashpad_handler.exe"))
    assert is_excluded_executable(Path(r"C:\Apps\Foo\updater.exe"))
    assert is_excluded_executable(Path(r"C:\Users\me\AppData\Local\Discord\Update.exe"), [])
    # Discord Squirrel Update.exe WITH --processStart Discord.exe must be accepted!
    assert not is_excluded_executable(
        Path(r"C:\Users\me\AppData\Local\Discord\Update.exe"),
        ["--processStart", "Discord.exe"],
    )
    discord_cand = build_candidate(
        Path(r"C:\Users\me\AppData\Local\Discord\Update.exe"),
        arguments=["--processStart", "Discord.exe"],
        display_hint="Discord",
    )
    assert discord_cand is not None
    assert discord_cand.app_id == "discord"
    assert "Discord.exe" in discord_cand.process_names
    assert "迪斯科" in discord_cand.aliases

    # 2. Lossless multi-pass TOML read-modify-write test on temporary local.toml
    sample_existing_toml = """# 用户本地覆盖配置（严禁丢失注释与已有段）
[preprocess]
implementation = "webrtc_aec3"
aec_enabled = true

[kws]
threshold = 0.28
asr_probe = true

[asr]
final_model = "models/asr-personal-20260926/model.int8.onnx"
fallback_decoder = "none"

[commands]
parser_mode = "hybrid"
router_backend = "openai_compatible"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "local.toml"
        cfg_file.write_text(sample_existing_toml, encoding="utf-8")

        # Pass 1: Write Steam & Clash
        update_local_toml_apps_lossless(
            cfg_file,
            new_entries={
                "steam": {
                    "display_name": "Steam",
                    "target": r"C:\Program Files (x86)\Steam\steam.exe",
                    "arguments": [],
                    "aliases": ["steam", "蒸汽平台", "死盯", "斯蒂姆"],
                    "process_names": ["steam.exe"],
                },
                "clash": {
                    "display_name": "Clash",
                    "target": r"C:\Program Files\Clash Verge\clash-verge.exe",
                    "arguments": [],
                    "aliases": ["clash", "代理软件", "可拉什", "克拉什"],
                    "process_names": ["clash-verge.exe"],
                },
            },
            quickstart_dirs=[r"D:\QuickStart"],
            scan_system_apps=True,
        )

        # Pass 2: Incrementally add Discord & PowerShell without losing Steam/Clash or [kws]/[asr]
        update_local_toml_apps_lossless(
            cfg_file,
            new_entries={
                "discord": discord_cand.to_entry_dict(),
                "powershell": {
                    "display_name": "PowerShell",
                    "target": r"C:\Program Files\PowerShell\7\pwsh.exe",
                    "arguments": [],
                    "aliases": ["powershell", "终端", "泡我秀"],
                    "process_names": ["pwsh.exe"],
                },
            },
        )

        final_txt = cfg_file.read_text(encoding="utf-8")
        parsed = tomllib.loads(final_txt)

        assert "# 用户本地覆盖配置（严禁丢失注释与已有段）" in final_txt
        assert parsed["kws"]["threshold"] == 0.28
        assert parsed["asr"]["final_model"] == "models/asr-personal-20260926/model.int8.onnx"
        assert parsed["preprocess"]["implementation"] == "webrtc_aec3"
        assert parsed["commands"]["parser_mode"] == "hybrid"
        assert parsed["apps"]["quickstart_dirs"] == [r"D:\QuickStart"]
        assert parsed["apps"]["scan_system_apps"] is True
        assert set(parsed["apps"]["entries"].keys()) == {"steam", "clash", "discord", "powershell"}
        assert parsed["apps"]["entries"]["steam"]["target"] == r"C:\Program Files (x86)\Steam\steam.exe"
        assert "死盯" in parsed["apps"]["entries"]["steam"]["aliases"]
        assert parsed["apps"]["entries"]["discord"]["arguments"] == ["--processStart", "Discord.exe"]

    print("SELF-TEST PASSED! Shortcut filtering, Discord Squirrel handling, and lossless TOML sync verified.")
    return 0


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    default_cfg = repo_root / "config" / "local.toml"

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--discover-quickstart", action="store_true",
                        help="auto-probe local drives and user folders for QuickStart / shortcut directories")
    parser.add_argument("--scan", action="store_true",
                        help="scan QuickStart folders and/or Windows system apps for launchable executables")
    parser.add_argument("--search", type=str, default="",
                        help="search for a specific desktop application by name, exe, or alias")
    parser.add_argument("--quickstart-dir", dest="quickstart_dirs", action="append", default=[],
                        help="QuickStart directory path (repeatable)")
    parser.add_argument("--no-system-scan", action="store_true",
                        help="only scan explicitly provided --quickstart-dir paths")

    # TOML write / entry upsert flags
    parser.add_argument("--write-local-toml", action="store_true",
                        help="losslessly write selected/explicit app entries into [apps] of --config")
    parser.add_argument("--config", type=Path, default=default_cfg,
                        help=f"target TOML file path (default: {default_cfg})")
    parser.add_argument("--auto-select", action="store_true",
                        help="when used with --search and --write-local-toml, write the top-ranked search match")
    parser.add_argument("--add-id", type=str, default="",
                        help="explicit app entry id (e.g. steam, clash, discord) to upsert into [apps.entries.<id>]")
    parser.add_argument("--display-name", type=str, default="", help="display_name for --add-id")
    parser.add_argument("--target", type=str, default="", help="executable target path for --add-id")
    parser.add_argument("--arguments", type=str, default="",
                        help="comma-separated or shell-quoted launch arguments for --add-id")
    parser.add_argument("--aliases", type=str, default="",
                        help="comma-separated spoken aliases & ASR homophones (e.g. 'steam,蒸汽平台,死盯,斯蒂姆')")
    parser.add_argument("--process-names", type=str, default="",
                        help="comma-separated process image names (e.g. 'steam.exe')")
    parser.add_argument("--scan-system-apps", choices=["true", "false"], default=None,
                        help="explicitly set [apps].scan_system_apps in TOML")
    parser.add_argument("--replace-entries", action="store_true",
                        help="replace existing [apps.entries.*] instead of incremental merge")
    parser.add_argument("--limit", type=int, default=30, help="max results to print in --scan / --search")
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON")
    parser.add_argument("--self-test", action="store_true", help="run offline unit tests and exit")

    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.discover_quickstart:
        found_dirs = discover_quickstart_dirs()
        if args.json:
            print(json.dumps(found_dirs, ensure_ascii=False, indent=2))
        else:
            print(f"=== Discovered {len(found_dirs)} QuickStart / Shortcut Candidate Directories ===")
            for idx, d in enumerate(found_dirs, 1):
                samples = ", ".join(d["sample_items"][:6])
                print(f"  [{idx}] {d['path']}  ({d['kind']}: {d['lnk_count']} .lnk, {d['exe_count']} .exe)")
                if samples:
                    print(f"      e.g.: {samples}")
        return 0

    qdirs = [Path(os.path.expandvars(p)) for p in args.quickstart_dirs]
    candidates: list[AppCandidate] = []
    if args.scan or args.search:
        candidates = collect_all_candidates(
            quickstart_dirs=qdirs,
            include_system=not args.no_system_scan,
        )
        if args.search:
            candidates = search_candidates(candidates, args.search)

        if not args.write_local_toml:
            shown = candidates[: max(1, args.limit)]
            if args.json:
                print(json.dumps([
                    {"app_id": c.app_id, "source": c.source, "score": c.score, **c.to_entry_dict()}
                    for c in shown
                ], ensure_ascii=False, indent=2))
            else:
                header = f"Search '{args.search}'" if args.search else "Scanned Desktop Apps"
                print(f"=== {header}: {len(candidates)} match(es) (showing {len(shown)}) ===")
                for c in shown:
                    print(f"  [{c.app_id}] {c.display_name}  (source={c.source})")
                    print(f"    target        : {c.target}")
                    if c.arguments:
                        print(f"    arguments     : {c.arguments}")
                    print(f"    aliases       : {c.aliases}")
                    print(f"    process_names : {c.process_names}")
            return 0

    if args.write_local_toml:
        entries_to_write: dict[str, dict[str, Any]] = {}

        if args.add_id:
            if not args.target:
                parser.error("--add-id requires --target <path_to_exe>")
            app_id = sanitize_app_id(args.add_id)
            target_p = Path(os.path.expandvars(args.target))
            arg_list = split_windows_arguments(args.arguments)
            alias_list = parse_csv_list(args.aliases)
            proc_list = parse_csv_list(args.process_names) or extract_process_names(target_p, arg_list)

            # Merge built-in preset aliases if matching and user didn't supply any
            if not alias_list and app_id in BUILTIN_APP_PRESETS:
                alias_list = list(BUILTIN_APP_PRESETS[app_id]["aliases"])
            elif not alias_list:
                alias_list = dedupe_preserve_order([(args.display_name or app_id).lower(), target_p.stem.lower()])

            disp = args.display_name or BUILTIN_APP_PRESETS.get(app_id, {}).get("display_name") or app_id
            entries_to_write[app_id] = {
                "display_name": disp,
                "target": str(target_p),
                "arguments": arg_list,
                "aliases": dedupe_preserve_order(alias_list),
                "process_names": dedupe_preserve_order(proc_list),
            }
        elif args.search and args.auto_select:
            if not candidates:
                print(f"Error: no desktop app found matching '{args.search}'.", file=sys.stderr)
                return 1
            best = candidates[0]
            extra_aliases = parse_csv_list(args.aliases)
            entry_dict = best.to_entry_dict()
            if extra_aliases:
                entry_dict["aliases"] = dedupe_preserve_order(entry_dict["aliases"] + extra_aliases)
            entries_to_write[best.app_id] = entry_dict
        elif args.scan and qdirs:
            # Sync all apps discovered inside explicitly provided QuickStart directories
            q_only = collect_all_candidates(quickstart_dirs=qdirs, include_system=False)
            for c in q_only:
                entries_to_write[c.app_id] = c.to_entry_dict()

        scan_flag = None if args.scan_system_apps is None else (args.scan_system_apps == "true")
        updated_apps = update_local_toml_apps_lossless(
            config_path=args.config,
            new_entries=entries_to_write,
            quickstart_dirs=[str(p) for p in qdirs] if qdirs else None,
            scan_system_apps=scan_flag,
            replace_entries=args.replace_entries,
        )
        print(f"Updated [apps] in {args.config} (total entries: {len(updated_apps.get('entries', {}))})")
        if entries_to_write:
            print(render_apps_toml_block({
                "quickstart_dirs": updated_apps.get("quickstart_dirs", []),
                "scan_system_apps": updated_apps.get("scan_system_apps", True),
                "entries": entries_to_write,
            }).rstrip())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
