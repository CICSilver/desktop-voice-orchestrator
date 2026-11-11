---
name: sync-desktop-apps
description: >-
  Use this skill when the user asks to add, locate, or configure desktop application executable
  paths for voice launching, synchronize a QuickStart shortcut folder, or update spoken app aliases
  and Mandarin ASR homophones in `config/local.toml`.
---

# Sync Desktop Apps (`[apps]` Voice Launch Runbook)

Follow this workflow whenever the user asks to register a desktop program (e.g. "帮我把某某软件加进语音控制") or sync a shortcut launcher folder (e.g. "同步我的 QuickStart 应用").

## Workflow Steps

### 1. Locate the Real Executable & Process Name

Run [sync_apps.py](./scripts/sync_apps.py) to probe QuickStart folders or search across Start Menu shortcuts, Registry `App Paths`, common install directories, and running processes:

```powershell
# Auto-discover candidate QuickStart / shortcut folders on this machine:
python .agents/skills/sync-desktop-apps/scripts/sync_apps.py --discover-quickstart

# Search for a specific app (resolves .lnk TargetPath & Arguments, excludes uninstallers/updaters):
python .agents/skills/sync-desktop-apps/scripts/sync_apps.py --search "<app_name>"

# Or scan an entire QuickStart directory confirmed by the user:
python .agents/skills/sync-desktop-apps/scripts/sync_apps.py --scan --quickstart-dir "D:\QuickStart" --no-system-scan
```

> [!TIP]
> Electron Squirrel apps such as Discord launch via `Update.exe --processStart Discord.exe` while running as `Discord.exe`. `sync_apps.py` preserves `arguments = ["--processStart", "Discord.exe"]` and sets `process_names = ["Discord.exe"]` automatically.

### 2. Curate 3–5 Spoken Names & Mandarin ASR Homophones

Before writing `aliases`, ensure every English or uncommon app name includes **the lowercase original name**, **1–2 natural Chinese descriptions**, and **2–3 Mandarin ASR phonetic homophones** so both local rules and the LLM router recognize noisy transcripts:

| App | Recommended `aliases` |
| :--- | :--- |
| `Steam` | `["steam", "蒸汽平台", "死盯", "斯蒂姆"]` |
| `Clash` | `["clash", "代理软件", "可拉什", "克拉什", "小猫咪"]` |
| `Discord` | `["discord", "语音软件", "迪斯科", "迪斯扣", "开黑语音"]` |
| `QQ` | `["qq", "腾讯qq", "扣扣", "企鹅"]` |
| `PowerShell` | `["powershell", "终端", "命令行", "泡我秀", "帕瓦谢尔"]` |

### 3. Losslessly Upsert `[apps]` in `config/local.toml`

Use `--write-local-toml` to merge the entry into `config/local.toml`. The script verifies via `tomllib` that existing `[kws]`, `[asr]`, `[preprocess]`, `[commands]`, and comments remain 100% untouched:

```powershell
# Add or update a single app entry with curated aliases:
python .agents/skills/sync-desktop-apps/scripts/sync_apps.py --write-local-toml `
  --add-id steam `
  --display-name "Steam" `
  --target "C:\Program Files (x86)\Steam\steam.exe" `
  --aliases "steam,蒸汽平台,死盯,斯蒂姆" `
  --process-names "steam.exe"

# Or sync all valid shortcuts from a confirmed QuickStart directory:
python .agents/skills/sync-desktop-apps/scripts/sync_apps.py --scan `
  --quickstart-dir "D:\QuickStart" --write-local-toml
```

Expected TOML schema written to `config/local.toml`:

```toml
[apps]
quickstart_dirs = ["..."]
scan_system_apps = true

[apps.entries.steam]
display_name = "Steam"
target = "C:\\Program Files (x86)\\Steam\\steam.exe"
arguments = []
aliases = ["steam", "蒸汽平台", "死盯", "斯蒂姆"]
process_names = ["steam.exe"]
```

### 4. Verify C++ Catalog Recognition

Run the compiled frontend binary's `list-apps` command to confirm the C++ `[apps]` loader recognizes the new entry, aliases, and target executable:

```powershell
./build/windows-x64/Release/voice_frontend.exe list-apps
```
