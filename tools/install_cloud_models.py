#!/usr/bin/env python3
"""
install_cloud_models.py - Home PC One-Click Installer for Colab M3 & M4 Outputs.

Copies trained M3 (SenseVoice INT8 ASR) and M4 (Qwen3-4B Q4_K_M Router GGUF)
artifacts from Google Drive for Desktop using buffered streaming I/O (safe with
Windows Drive File Stream), registers/smoke-tests `dvo-router` in local Ollama,
and losslessly merges required settings into `config/local.toml`.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

CHUNK_SIZE_BYTES = 8 * 1024 * 1024  # 8 MiB buffered copy

COMMON_DRIVE_LETTERS = ["H:", "G:", "I:", "F:", "D:", "C:"]
COMMON_DRIVE_SUBPATHS = [
    "我的云端硬盘/desktop-voice-orchestrator-data",
    "My Drive/desktop-voice-orchestrator-data",
    "desktop-voice-orchestrator-data",
]

M3_CONFIG_UPDATES: Dict[str, Dict[str, Any]] = {
    "asr": {
        "final_decoder": "sense_voice",
        "final_model": "models/asr-personal-v2/model.int8.onnx",
        "final_tokens": "models/asr-personal-v2/tokens.txt",
    }
}

M4_CONFIG_UPDATES: Dict[str, Dict[str, Any]] = {
    "commands": {
        "parser_mode": "hybrid",
        "router_backend": "openai_compatible",
        "router_base_url": "http://127.0.0.1:11434/v1",
        "router_model": "dvo-router",
        "router_profile": "reading",
        "local_router_base_url": "http://127.0.0.1:11434/v1",
        "local_router_model": "dvo-router",
    }
}


def resolve_drive_root(explicit_root: Optional[str] = None) -> Path:
    """Resolve the Google Drive data root directory."""
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    env_root = os.environ.get("DVO_DRIVE_DATA_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    for letter in COMMON_DRIVE_LETTERS:
        for sub in COMMON_DRIVE_SUBPATHS:
            candidate = Path(letter) / sub
            if candidate.exists():
                return candidate.resolve()

    return Path("H:/我的云端硬盘/desktop-voice-orchestrator-data").resolve()


def stream_copy_file(
    src: Path,
    dst: Path,
    *,
    chunk_size: int = CHUNK_SIZE_BYTES,
    dry_run: bool = False,
) -> int:
    """
    Copy a file using standard buffered binary I/O and atomic rename.

    Avoids Windows CopyFileEx/mmap quirks on Google Drive virtual drives.
    """
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    expected_size = src.stat().st_size
    if dry_run:
        print(f"[DRY-RUN] stream copy {src} -> {dst} ({expected_size} bytes)")
        return expected_size

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{dst.name}.tmp")
    copied = 0
    t0 = time.monotonic()
    try:
        with open(src, "rb") as fin, open(tmp_dst, "wb") as fout:
            while True:
                buf = fin.read(chunk_size)
                if not buf:
                    break
                fout.write(buf)
                copied += len(buf)
            fout.flush()
            os.fsync(fout.fileno())
        if copied != expected_size or tmp_dst.stat().st_size != expected_size:
            raise RuntimeError(
                f"Byte count mismatch copying {src}: copied={copied}, expected={expected_size}"
            )
        os.replace(tmp_dst, dst)
    except Exception:
        tmp_dst.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - t0
    print(f"Copied {dst.name} ({copied:,} bytes) in {elapsed:.1f}s -> {dst}")
    return copied


def _copy_artifact_group(
    src_dir: Path,
    dst_dir: Path,
    required: Iterable[str],
    optional: Iterable[str],
    *,
    dry_run: bool = False,
) -> List[Path]:
    """Copy required and optional artifact files from Drive to local dir."""
    copied_paths: List[Path] = []
    for name in required:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(
                f"Required cloud artifact missing: {src}\n"
                "Please check that Colab training finished and Google Drive finished syncing."
            )
        dst = dst_dir / name
        stream_copy_file(src, dst, dry_run=dry_run)
        copied_paths.append(dst)

    for name in optional:
        src = src_dir / name
        if src.is_file():
            dst = dst_dir / name
            stream_copy_file(src, dst, dry_run=dry_run)
            copied_paths.append(dst)
    return copied_paths


def install_m3_files(drive_root: Path, repo_root: Path, *, dry_run: bool = False) -> List[Path]:
    """Install M3 personalized SenseVoice INT8 artifacts into models/asr-personal-v2/."""
    src_dir = drive_root / "models" / "asr-personal-v2"
    dst_dir = repo_root / "models" / "asr-personal-v2"
    return _copy_artifact_group(
        src_dir,
        dst_dir,
        required=("model.int8.onnx", "tokens.txt"),
        optional=("eval_report.json", "train_summary.json"),
        dry_run=dry_run,
    )


def install_m4_files(drive_root: Path, repo_root: Path, *, dry_run: bool = False) -> List[Path]:
    """Install M4 router GGUF, Modelfile, and eval report into models/router/."""
    src_dir = drive_root / "router_m4" / "export"
    dst_dir = repo_root / "models" / "router"
    copied = _copy_artifact_group(
        src_dir,
        dst_dir,
        required=("dvo-qwen3-4b-router-q4_k_m.gguf",),
        optional=("eval_report.json",),
        dry_run=dry_run,
    )
    drive_modelfile = src_dir / "Modelfile.dvo-router"
    repo_modelfile = repo_root / "training" / "router" / "Modelfile.dvo-router"
    modelfile_src = drive_modelfile if drive_modelfile.is_file() else repo_modelfile
    if not modelfile_src.is_file():
        raise FileNotFoundError(
            f"Modelfile.dvo-router not found in {drive_modelfile} or {repo_modelfile}"
        )
    modelfile_dst = dst_dir / "Modelfile.dvo-router"
    stream_copy_file(modelfile_src, modelfile_dst, dry_run=dry_run)
    copied.append(modelfile_dst)
    return copied


def find_ollama_executable() -> Optional[str]:
    """Locate the Ollama CLI binary on Windows/macOS/Linux."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
        Path("C:/Program Files (x86)/Ollama/ollama.exe"),
        Path("D:/Ollama/ollama.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Extract JSON payload from router completion text, stripping think/code tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    return json.loads(cleaned)


def _call_ollama_chat(base_url: str, model: str, user_text: str, timeout_sec: float = 30.0) -> Dict[str, Any]:
    """Call Ollama OpenAI-compatible /chat/completions endpoint and parse JSON actions."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": user_text}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to connect to Ollama OpenAI endpoint at {url}: {exc}. "
            "Ensure Ollama Desktop or `ollama serve` is running."
        ) from exc

    content = body["choices"][0]["message"]["content"]
    parsed = _parse_json_response(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
        raise ValueError(f"Invalid router response schema for {user_text!r}: {content!r}")
    return parsed


def register_and_smoke_test_ollama(
    router_dir: Path,
    *,
    model_name: str = "dvo-router",
    base_url: str = "http://127.0.0.1:11434/v1",
    dry_run: bool = False,
) -> None:
    """Register dvo-router via Ollama Modelfile and run 1 positive + 1 negative smoke test."""
    ollama_bin = find_ollama_executable()
    if not ollama_bin:
        if dry_run:
            print("[DRY-RUN] ollama CLI not found on PATH; skipping `ollama create` preview.")
            return
        raise FileNotFoundError(
            "Ollama CLI executable not found. Please install Ollama on the host PC "
            "(https://ollama.com) or re-run with --skip-ollama."
        )

    modelfile_path = router_dir / "Modelfile.dvo-router"
    if dry_run:
        print(f"[DRY-RUN] {ollama_bin} create {model_name} -f {modelfile_path}")
        print(f"[DRY-RUN] smoke test POST {base_url}/chat/completions + {ollama_bin} stop {model_name}")
        return

    print(f"Registering Ollama model '{model_name}' from {modelfile_path} ...")
    subprocess.run(
        [ollama_bin, "create", model_name, "-f", "Modelfile.dvo-router"],
        cwd=str(router_dir),
        check=True,
    )

    try:
        pos = _call_ollama_chat(base_url, model_name, "把声音调小一点再切歌")
        if not pos["actions"]:
            raise AssertionError(f"Positive smoke sample returned empty actions: {pos}")
        neg = _call_ollama_chat(base_url, model_name, "今天天气真不错")
        if neg["actions"] != []:
            raise AssertionError(f"Negative smoke sample expected actions==[], got: {neg}")
        print(
            "Ollama smoke test passed: "
            f"pos_actions={[a.get('action') for a in pos['actions']]}, neg_actions=[]"
        )
    finally:
        subprocess.run([ollama_bin, "stop", model_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _format_toml_value(value: Any) -> str:
    """Format a Python scalar into a TOML literal string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _extract_section_header(line: str) -> Optional[str]:
    """Return exact TOML section name if line is a `[section]` header, else None."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("[["):
        return None
    # Remove trailing comment outside quotes (section headers don't hold strings normally)
    code_part = stripped.split("#", 1)[0].strip()
    if "=" in code_part:
        return None
    match = re.match(r"^\[([^\[\]]+)\]$", code_part)
    if not match:
        return None
    return match.group(1).strip()


def _extract_key_name(line: str) -> Optional[Tuple[str, str]]:
    """Return (leading_indent, key_name) if line assigns a bare TOML key."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#") or stripped.startswith("["):
        return None
    match = re.match(r"^(\s*)([A-Za-z0-9_-]+)\s*=", line)
    if not match:
        return None
    return match.group(1), match.group(2)


def patch_local_toml(existing_text: str, updates: Dict[str, Dict[str, Any]]) -> str:
    """
    Losslessly update targeted TOML sections/keys while preserving all other sections and comments.

    Matches exact section headers (e.g. `[commands]` without touching `[commands.phrases]`).
    """
    newline = "\r\n" if "\r\n" in existing_text else "\n"
    lines = existing_text.splitlines() if existing_text else []

    # Group lines into (section_name_or_None, section_lines)
    blocks: List[Tuple[Optional[str], List[str]]] = []
    current_section: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        header = _extract_section_header(line)
        if header is not None:
            if current_lines or current_section is not None:
                blocks.append((current_section, current_lines))
            current_section = header
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines or current_section is not None:
        blocks.append((current_section, current_lines))

    updated_sections = set()
    new_blocks: List[Tuple[Optional[str], List[str]]] = []

    for sec_name, sec_lines in blocks:
        if sec_name not in updates:
            new_blocks.append((sec_name, sec_lines))
            continue

        updated_sections.add(sec_name)
        kv_targets = updates[sec_name]
        seen_keys = set()
        patched_lines: List[str] = []

        for line in sec_lines:
            parsed_key = _extract_key_name(line)
            if parsed_key and parsed_key[1] in kv_targets:
                indent, key = parsed_key
                patched_lines.append(f"{indent}{key} = {_format_toml_value(kv_targets[key])}")
                seen_keys.add(key)
            else:
                patched_lines.append(line)

        missing_lines = [
            f"{key} = {_format_toml_value(val)}"
            for key, val in kv_targets.items()
            if key not in seen_keys
        ]
        if missing_lines:
            # Insert new keys before trailing blank lines inside this section
            trailing_blanks = 0
            while len(patched_lines) > 1 and patched_lines[-1].strip() == "":
                patched_lines.pop()
                trailing_blanks += 1
            patched_lines.extend(missing_lines)
            patched_lines.extend([""] * trailing_blanks)

        new_blocks.append((sec_name, patched_lines))

    for sec_name, kv_targets in updates.items():
        if sec_name in updated_sections:
            continue
        sec_lines = [f"[{sec_name}]"] + [
            f"{key} = {_format_toml_value(val)}" for key, val in kv_targets.items()
        ]
        if new_blocks and new_blocks[-1][1] and new_blocks[-1][1][-1].strip() != "":
            new_blocks[-1][1].append("")
        new_blocks.append((sec_name, sec_lines))

    flat_lines: List[str] = []
    for _, sec_lines in new_blocks:
        flat_lines.extend(sec_lines)

    result = newline.join(flat_lines)
    if result and not result.endswith(newline):
        result += newline

    if tomllib is not None:
        tomllib.loads(result)
    return result


def update_local_toml_file(
    local_toml_path: Path,
    only: str,
    *,
    dry_run: bool = False,
) -> Path:
    """Merge M3 and/or M4 settings losslessly into config/local.toml."""
    updates: Dict[str, Dict[str, Any]] = {}
    if only in ("all", "m3"):
        updates.update(M3_CONFIG_UPDATES)
    if only in ("all", "m4"):
        updates.update(M4_CONFIG_UPDATES)

    existing_text = local_toml_path.read_text("utf-8") if local_toml_path.is_file() else ""
    merged_text = patch_local_toml(existing_text, updates)

    if dry_run:
        print(f"[DRY-RUN] update {local_toml_path} sections={list(updates.keys())}")
        return local_toml_path

    local_toml_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_toml_path.with_name(f".{local_toml_path.name}.tmp")
    tmp_path.write_text(merged_text, encoding="utf-8")
    os.replace(tmp_path, local_toml_path)
    print(f"Updated {local_toml_path} (sections: {', '.join(updates.keys())})")
    return local_toml_path


def install_models(
    *,
    drive_root: Path,
    repo_root: Path,
    only: str = "all",
    skip_ollama: bool = False,
    skip_config: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full local deployment flow for M3 and/or M4 models."""
    if not drive_root.is_dir():
        raise FileNotFoundError(
            f"Google Drive data directory not found: {drive_root}. "
            "Pass --drive-root or set DVO_DRIVE_DATA_ROOT."
        )

    if only in ("all", "m3"):
        install_m3_files(drive_root, repo_root, dry_run=dry_run)

    if only in ("all", "m4"):
        install_m4_files(drive_root, repo_root, dry_run=dry_run)
        if not skip_ollama:
            register_and_smoke_test_ollama(repo_root / "models" / "router", dry_run=dry_run)

    if not skip_config:
        update_local_toml_file(repo_root / "config" / "local.toml", only, dry_run=dry_run)


def run_self_test() -> None:
    """Run offline self-test covering streaming copy, Modelfile fallback, and lossless TOML merge."""
    print("Running install_cloud_models.py self-test ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        drive_root = root / "drive_data"
        repo_root = root / "repo"

        # Seed simulated Drive M3 & M4 artifacts
        m3_src = drive_root / "models" / "asr-personal-v2"
        m3_src.mkdir(parents=True)
        (m3_src / "model.int8.onnx").write_bytes(b"ONNX_FAKE_BYTES_" * 1024)
        (m3_src / "tokens.txt").write_text("<blk> 0\n你 1\n好 2\n", encoding="utf-8")
        (m3_src / "eval_report.json").write_text('{"cer": 0.021}', encoding="utf-8")
        (m3_src / "train_summary.json").write_text('{"best_epoch": 12}', encoding="utf-8")

        m4_src = drive_root / "router_m4" / "export"
        m4_src.mkdir(parents=True)
        (m4_src / "dvo-qwen3-4b-router-q4_k_m.gguf").write_bytes(b"GGUF_DATA_" * 2048)
        (m4_src / "eval_report.json").write_text('{"exact_plan_match_rate": 0.99}', encoding="utf-8")

        # Provide Modelfile in repo fallback path
        repo_mf = repo_root / "training" / "router" / "Modelfile.dvo-router"
        repo_mf.parent.mkdir(parents=True)
        repo_mf.write_text('FROM ./dvo-qwen3-4b-router-q4_k_m.gguf\nSYSTEM """test"""\n', encoding="utf-8")

        # Seed pre-existing home config/local.toml with custom settings and comments
        local_toml = repo_root / "config" / "local.toml"
        local_toml.parent.mkdir(parents=True)
        local_toml.write_text(
            "# Home PC local config\n"
            "[audio]\n"
            "microphone_device = \"USB-Mic-Home\"\n\n"
            "[commands]\n"
            "parser_mode = \"rules\"\n"
            "default_volume_step_percent = 8\n\n"
            "[commands.phrases]\n"
            "play = [\"放首歌\"]\n\n"
            "[apps.entries.steam]\n"
            "target = \"D:/Steam/steam.exe\"\n",
            encoding="utf-8",
        )

        install_models(
            drive_root=drive_root,
            repo_root=repo_root,
            only="all",
            skip_ollama=True,
            skip_config=False,
            dry_run=False,
        )

        # Idempotent second pass check
        install_models(
            drive_root=drive_root,
            repo_root=repo_root,
            only="all",
            skip_ollama=True,
            skip_config=False,
            dry_run=False,
        )

        assert (repo_root / "models" / "asr-personal-v2" / "model.int8.onnx").read_bytes() == (
            m3_src / "model.int8.onnx"
        ).read_bytes()
        assert (repo_root / "models" / "asr-personal-v2" / "tokens.txt").exists()
        assert (repo_root / "models" / "router" / "dvo-qwen3-4b-router-q4_k_m.gguf").read_bytes() == (
            m4_src / "dvo-qwen3-4b-router-q4_k_m.gguf"
        ).read_bytes()
        assert (repo_root / "models" / "router" / "Modelfile.dvo-router").read_text("utf-8") == repo_mf.read_text("utf-8")

        merged_text = local_toml.read_text("utf-8")
        merged_lines = merged_text.splitlines()
        assert "# Home PC local config" in merged_text
        assert sum(1 for ln in merged_lines if ln.startswith("final_model =")) == 1
        assert sum(1 for ln in merged_lines if ln.startswith("router_model =")) == 1
        assert sum(1 for ln in merged_lines if ln.startswith("local_router_model =")) == 1
        assert tomllib is not None, "tomllib must be available on Python 3.11+ for self-test"
        parsed = tomllib.loads(merged_text)
        assert parsed["audio"]["microphone_device"] == "USB-Mic-Home"
        assert parsed["asr"]["final_decoder"] == "sense_voice"
        assert parsed["asr"]["final_model"] == "models/asr-personal-v2/model.int8.onnx"
        assert parsed["asr"]["final_tokens"] == "models/asr-personal-v2/tokens.txt"
        assert parsed["commands"]["parser_mode"] == "hybrid"
        assert parsed["commands"]["default_volume_step_percent"] == 8
        assert parsed["commands"]["local_router_model"] == "dvo-router"
        assert parsed["commands"]["phrases"]["play"] == ["放首歌"]
        assert parsed["apps"]["entries"]["steam"]["target"] == "D:/Steam/steam.exe"

    print("SELF-TEST PASSED!")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install Colab-trained M3 (ASR) and M4 (Router) models from Google Drive."
    )
    parser.add_argument("--drive-root", help="Path to desktop-voice-orchestrator-data on Google Drive.")
    parser.add_argument(
        "--only",
        choices=["all", "m3", "m4"],
        default="all",
        help="Install M3 ASR, M4 Router, or both (default: all).",
    )
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama create & smoke test.")
    parser.add_argument("--skip-config", action="store_true", help="Skip updating config/local.toml.")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without modifying files.")
    parser.add_argument("--self-test", action="store_true", help="Run offline unit self-test and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    drive_root = resolve_drive_root(args.drive_root)
    install_models(
        drive_root=drive_root,
        repo_root=repo_root,
        only=args.only,
        skip_ollama=args.skip_ollama,
        skip_config=args.skip_config,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
