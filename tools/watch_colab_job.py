#!/usr/bin/env python3
"""
watch_colab_job.py - Local PC Side: Zero-Token Blocking Colab Sentinel & Rescue Dispatcher.

Monitors Colab training jobs through Google Drive for Desktop file synchronization,
providing zero-token blocking waits for Antigravity agents, automatic heartbeat
staleness detection, remote rescue command dispatching, and repo syncing.
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_POLL_SEC = 20.0
DEFAULT_STALE_TIMEOUT_SEC = 300.0
DEFAULT_MAX_WAIT_SEC = 21600.0  # 6 hours

COMMON_DRIVE_LETTERS = ["H:", "G:", "I:", "F:", "D:", "C:"]
COMMON_DRIVE_SUBPATHS = [
    "我的云端硬盘/desktop-voice-orchestrator-data",
    "My Drive/desktop-voice-orchestrator-data",
    "desktop-voice-orchestrator-data",
]


def resolve_drive_root(explicit_root: Optional[str] = None) -> pathlib.Path:
    """Resolve the Google Drive data root directory."""
    if explicit_root:
        return pathlib.Path(explicit_root).expanduser().resolve()

    env_root = os.environ.get("DVO_DRIVE_DATA_ROOT")
    if env_root:
        p = pathlib.Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    for letter in COMMON_DRIVE_LETTERS:
        for sub in COMMON_DRIVE_SUBPATHS:
            candidate = pathlib.Path(letter) / sub
            if candidate.exists():
                return candidate.resolve()

    # Fallback to standard default path
    return pathlib.Path("H:/我的云端硬盘/desktop-voice-orchestrator-data").resolve()


def read_text_safe(path: pathlib.Path, max_retries: int = 3, retry_delay: float = 0.4) -> str:
    """
    Read text safely from Google Drive file stream without mmap,
    retrying transient OS/permission locks.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    if last_err:
        raise last_err
    return ""


def read_json_safe(path: pathlib.Path, max_retries: int = 3, retry_delay: float = 0.4) -> Optional[Dict[str, Any]]:
    """
    Read and parse JSON safely from Google Drive file stream.
    Retries transient read or mid-sync decoding collisions.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with open(path, "rb") as f:
                data = f.read().decode("utf-8", errors="replace")
            if not data.strip():
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            return json.loads(data)
        except (PermissionError, OSError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    if last_err:
        raise last_err
    return None


def atomic_write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON using a temporary file and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".command.{os.getpid()}_{uuid.uuid4().hex[:6]}.tmp"
    content = json.dumps(data, indent=2, ensure_ascii=False)
    with open(tmp_path, "wb") as f:
        f.write(content.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def get_tail_lines(log_path: pathlib.Path, n: int = 20) -> List[str]:
    """Read the last n lines of a log file safely."""
    if not log_path.exists():
        return []
    try:
        text = read_text_safe(log_path)
        lines = [line.rstrip("\r\n") for line in text.splitlines()]
        return lines[-n:]
    except Exception:
        return []


def print_status_summary(status: Dict[str, Any], tail_path: Optional[pathlib.Path] = None, show_full_error: bool = False) -> None:
    """Print structured status summary to stdout."""
    print("=" * 64)
    print(f"Job Name:       {status.get('job_name', 'N/A')}")
    print(f"Run ID:         {status.get('run_id', 'N/A')}")
    print(f"State:          {status.get('state', 'N/A')}")
    print(f"Stage:          {status.get('stage', 'N/A')} ({status.get('stage_index', 0)}/{status.get('total_stages', 0)})")
    
    prog = status.get("progress") or {}
    step = prog.get("step")
    total_steps = prog.get("total_steps")
    loss = prog.get("loss")
    epoch = prog.get("epoch")
    eta = prog.get("eta_sec")
    msg = prog.get("message")
    prog_str = f"step={step}/{total_steps}" if step is not None else ""
    if epoch is not None:
        prog_str += f", epoch={epoch}"
    if loss is not None:
        prog_str += f", loss={loss:.4f}" if isinstance(loss, (int, float)) else f", loss={loss}"
    if eta is not None:
        prog_str += f", eta={eta:.1f}s"
    if msg:
        prog_str += f" ({msg})"
    if prog_str:
        print(f"Progress:       {prog_str}")

    gpu = status.get("gpu") or {}
    if gpu:
        vram_u = gpu.get("vram_used_mb", 0)
        vram_t = gpu.get("vram_total_mb", 0)
        util = gpu.get("utilization_pct", 0)
        name = gpu.get("name", "Unknown GPU")
        print(f"GPU:            {name} | VRAM: {vram_u}MB / {vram_t}MB | Util: {util}%")

    updated_ts = status.get("updated_at_ts")
    if updated_ts:
        age = time.time() - float(updated_ts)
        print(f"Heartbeat Age:  {age:.1f}s (seq={status.get('heartbeat_seq', 'N/A')})")

    deadline = status.get("rescue_deadline_ts")
    if deadline:
        rem = deadline - time.time()
        print(f"Rescue Window:  {rem:.1f}s remaining (deadline={deadline})")

    metrics = status.get("summary_metrics")
    if metrics:
        print(f"Metrics:        {json.dumps(metrics, ensure_ascii=False)}")

    artifacts = status.get("artifacts")
    if artifacts:
        print("Artifacts:")
        for art in artifacts:
            p = art.get("path") if isinstance(art, dict) else str(art)
            sz = art.get("size_bytes") if isinstance(art, dict) else ""
            sz_str = f" ({sz} bytes)" if sz else ""
            print(f"  - {p}{sz_str}")

    err = status.get("error")
    if err and isinstance(err, dict):
        print("-" * 64)
        print(f"Error Type:     {err.get('exception_type', 'N/A')}")
        print(f"Error Message:  {err.get('message', 'N/A')}")
        print(f"Failed Stage:   {err.get('failed_stage', 'N/A')}")
        if show_full_error:
            tb = err.get("traceback")
            if tb:
                print("Traceback:")
                print(tb)
            log_tail = err.get("log_tail")
            if log_tail:
                print("Error Log Tail:")
                print(log_tail)

    if tail_path and tail_path.exists():
        recent_logs = get_tail_lines(tail_path, 20)
        if recent_logs:
            print("-" * 64)
            print(f"Recent tail.log ({tail_path.name}):")
            for line in recent_logs:
                print(f"  {line}")
    print("=" * 64)


def file_sha256(path: pathlib.Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sync_repo(repo_root: pathlib.Path, drive_root: pathlib.Path) -> int:
    """
    Incrementally sync training/ and tools/ scripts, notebooks, Modelfiles,
    and configs to <drive_root>/repo_sync/.
    Also syncs Colab notebooks to <drive_root>/../Colab Notebooks/ if present.
    """
    sync_dirs = ["training", "tools"]
    ignore_extensions = {
        ".pyc", ".onnx", ".gguf", ".pt", ".bin", ".wav", ".mp3", ".flac",
        ".tar", ".bz2", ".gz", ".zip", ".tmp"
    }
    ignore_dirs = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode", "build", "build-tts"}

    copied_count = 0
    repo_sync_root = drive_root / "repo_sync"

    for d in sync_dirs:
        src_base = repo_root / d
        if not src_base.exists():
            continue
        dst_base = repo_sync_root / d

        for root, dirs, files in os.walk(src_base):
            # Prune ignored directories in-place
            dirs[:] = [dir_name for dir_name in dirs if dir_name not in ignore_dirs and not dir_name.startswith(".")]

            rel_path = pathlib.Path(root).relative_to(src_base)
            target_dir = dst_base / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                ext = pathlib.Path(f).suffix.lower()
                if ext in ignore_extensions or f.startswith("."):
                    continue

                src_file = pathlib.Path(root) / f
                dst_file = target_dir / f

                should_copy = False
                if not dst_file.exists():
                    should_copy = True
                else:
                    src_stat = src_file.stat()
                    dst_stat = dst_file.stat()
                    if src_stat.st_size != dst_stat.st_size:
                        should_copy = True
                    elif src_stat.st_mtime > dst_stat.st_mtime:
                        if file_sha256(src_file) != file_sha256(dst_file):
                            should_copy = True

                if should_copy:
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1

    # Optional sync for Colab Notebooks folder in Drive
    notebook_target_dir = drive_root.parent / "Colab Notebooks"
    if notebook_target_dir.exists():
        notebooks_to_copy = [
            repo_root / "training" / "colab_m3_pipeline.ipynb",
            repo_root / "training" / "router" / "colab_m4_router_a100.ipynb"
        ]
        for nb_path in notebooks_to_copy:
            if nb_path.exists():
                dst_nb = notebook_target_dir / nb_path.name
                should_copy = False
                if not dst_nb.exists():
                    should_copy = True
                else:
                    if nb_path.stat().st_size != dst_nb.stat().st_size or file_sha256(nb_path) != file_sha256(dst_nb):
                        should_copy = True
                if should_copy:
                    shutil.copy2(nb_path, dst_nb)
                    copied_count += 1

    print(f"[REPO_SYNC] Synced {copied_count} changed files from '{repo_root}' to '{repo_sync_root}'")
    return copied_count


def send_command(
    drive_root: pathlib.Path,
    job_name: str,
    action: str,
    target_run_id: Optional[str] = None,
    note: Optional[str] = None,
    repo_root: Optional[pathlib.Path] = None,
) -> None:
    """Send a remote control command to Colab runtime."""
    valid_actions = {"sync_and_retry", "git_pull_and_retry", "abort"}
    if action not in valid_actions:
        raise ValueError(f"Invalid action '{action}'. Must be one of {valid_actions}")

    if action == "sync_and_retry":
        resolved_repo_root = repo_root or pathlib.Path(__file__).resolve().parent.parent
        print("[SEND_COMMAND] sync_and_retry requested: incrementally pushing repo changes first...")
        sync_repo(resolved_repo_root, drive_root)

    run_dir = drive_root / "colab_runs" / job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"

    resolved_run_id = target_run_id
    if not resolved_run_id:
        if status_path.exists():
            status = read_json_safe(status_path)
            if status and status.get("run_id"):
                resolved_run_id = status["run_id"]
        if not resolved_run_id:
            resolved_run_id = f"manual-{int(time.time())}"

    cmd_data = {
        "cmd_id": f"cmd-{int(time.time())}-{uuid.uuid4().hex[:6]}",
        "target_run_id": resolved_run_id,
        "issued_at_ts": time.time(),
        "action": action,
        "note": note or ""
    }

    cmd_path = run_dir / "command.json"
    atomic_write_json(cmd_path, cmd_data)
    print(f"[SEND_COMMAND] Successfully dispatched command to: {cmd_path}")
    print(f"Payload: {json.dumps(cmd_data, indent=2, ensure_ascii=False)}")


def do_status(drive_root: pathlib.Path, job_name: str) -> int:
    """Perform one-shot inspection of a Colab job."""
    run_dir = drive_root / "colab_runs" / job_name
    status_path = run_dir / "status.json"
    tail_path = run_dir / "tail.log"

    if not status_path.exists():
        print(f"[COLAB_STATUS] No status.json found for job '{job_name}' at: {run_dir}")
        return 1

    status = read_json_safe(status_path)
    if not status:
        print(f"[COLAB_STATUS] Failed to parse status.json at: {status_path}")
        return 1

    print_status_summary(status, tail_path, show_full_error=True)
    return 0


def do_watch(
    drive_root: pathlib.Path,
    job_name: str,
    poll_sec: float,
    stale_timeout_sec: float,
    max_wait_sec: float,
    wait_for_new_run: bool,
) -> int:
    """
    Zero-token blocking sentinel monitoring loop.
    Exits with:
      0: succeeded
      2: failed
      3: heartbeat stale
      4: max wait timeout
     10: waiting_for_fix
    """
    run_dir = drive_root / "colab_runs" / job_name
    status_path = run_dir / "status.json"
    tail_path = run_dir / "tail.log"

    start_time = time.time()
    last_stage: Optional[str] = None
    last_progress_print_ts: float = 0.0

    initial_run_id: Optional[str] = None
    initial_updated_ts: float = 0.0
    initial_state: Optional[str] = None
    waiting_for_fresh = False

    if wait_for_new_run and status_path.exists():
        curr = read_json_safe(status_path)
        if curr:
            initial_run_id = curr.get("run_id")
            initial_updated_ts = float(curr.get("updated_at_ts", 0.0))
            initial_state = curr.get("state")
            now = time.time()
            age = now - initial_updated_ts
            if initial_state in ("succeeded", "failed") or age > stale_timeout_sec:
                waiting_for_fresh = True
                print(
                    f"[COLAB_SENTINEL] Previous run '{initial_run_id}' is in terminal/stale state ('{initial_state}', age={age:.1f}s). "
                    f"Waiting for fresh run to start...",
                    flush=True,
                )

    print(f"[COLAB_SENTINEL] Monitoring job '{job_name}' at: {run_dir} (poll={poll_sec}s, stale_limit={stale_timeout_sec}s)", flush=True)

    while True:
        now = time.time()
        if now - start_time > max_wait_sec:
            print(f"[COLAB_MAX_WAIT_TIMEOUT] Hard safety ceiling {max_wait_sec}s exceeded for job '{job_name}'!", flush=True)
            if status_path.exists():
                curr = read_json_safe(status_path)
                if curr:
                    print_status_summary(curr, tail_path, show_full_error=True)
            return 4

        if not status_path.exists():
            time.sleep(poll_sec)
            continue

        status = read_json_safe(status_path)
        if not status:
            time.sleep(poll_sec)
            continue

        curr_run_id = status.get("run_id")
        curr_updated_ts = float(status.get("updated_at_ts", 0.0))
        curr_state = status.get("state", "unknown")

        if waiting_for_fresh:
            if curr_run_id == initial_run_id and curr_updated_ts <= initial_updated_ts:
                time.sleep(poll_sec)
                continue
            if curr_run_id == initial_run_id and curr_state not in ("starting", "running"):
                time.sleep(poll_sec)
                continue
            waiting_for_fresh = False
            print(f"[COLAB_SENTINEL] Fresh run detected: run_id='{curr_run_id}', state='{curr_state}'", flush=True)

        age = now - curr_updated_ts

        # Check terminal and rescue states
        if curr_state == "succeeded":
            print("[COLAB_SUCCEEDED]", flush=True)
            print_status_summary(status, tail_path, show_full_error=False)
            return 0

        if curr_state == "waiting_for_fix":
            print("[COLAB_WAITING_FOR_FIX]", flush=True)
            print_status_summary(status, tail_path, show_full_error=True)
            return 10

        if curr_state == "failed":
            print("[COLAB_FAILED]", flush=True)
            print_status_summary(status, tail_path, show_full_error=True)
            return 2

        if curr_state in ("starting", "running"):
            if age > stale_timeout_sec:
                print(f"[COLAB_HEARTBEAT_STALE] Heartbeat age {age:.1f}s exceeded timeout {stale_timeout_sec}s!", flush=True)
                print_status_summary(status, tail_path, show_full_error=True)
                return 3

            stage = status.get("stage", "starting")
            stage_idx = status.get("stage_index", 0)
            total_stages = status.get("total_stages", 0)
            prog = status.get("progress") or {}
            step = prog.get("step")
            total_steps = prog.get("total_steps")
            loss = prog.get("loss")
            gpu = status.get("gpu") or {}
            vram = gpu.get("vram_used_mb")

            time_since_print = now - last_progress_print_ts
            if stage != last_stage or time_since_print >= 300.0:
                last_stage = stage
                last_progress_print_ts = now
                now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                step_str = f" step={step}/{total_steps}" if step is not None else ""
                loss_str = f" loss={loss:.4f}" if isinstance(loss, (int, float)) else ""
                gpu_str = f" gpu={vram}MB" if vram is not None else ""
                print(f"[{now_str}] [{job_name}] stage={stage} ({stage_idx}/{total_stages}){step_str}{loss_str}{gpu_str}", flush=True)

        time.sleep(poll_sec)


def run_self_tests() -> int:
    """
    Execute self-test suite covering the 4 mandatory requirements:
    1) --wait-for-new-run ignores old succeeded run, detects new run running -> succeeded, exits 0.
    2) running -> waiting_for_fix exits 10 with raw traceback.
    3) --send-command sync_and_retry syncs training/ + tools/ and writes valid command.json.
    4) stale updated_at_ts triggers [COLAB_HEARTBEAT_STALE] with exit code 3.
    """
    import tempfile

    print("=" * 64)
    print("RUNNING SELF-TEST SUITE: tools/watch_colab_job.py")
    print("=" * 64)

    this_script = pathlib.Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="dvo_selftest_") as tmp_dir_str:
        tmp_dir = pathlib.Path(tmp_dir_str)
        drive_root = tmp_dir / "drive"
        job_dir = drive_root / "colab_runs" / "test_job"
        job_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Scenario 1: --wait-for-new-run ignores old succeeded run, detects new running -> succeeded (exit 0)
        # -------------------------------------------------------------
        print("\n[Scenario 1] Testing --wait-for-new-run with transition to succeeded...")
        old_status = {
            "schema_version": 1,
            "job_name": "test_job",
            "run_id": "run-old-111",
            "state": "succeeded",
            "stage": "done",
            "stage_index": 6,
            "total_stages": 6,
            "updated_at_ts": time.time() - 1000.0,
            "artifacts": [{"path": "model.onnx", "size_bytes": 1024}]
        }
        with open(job_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(old_status, f)

        proc1 = subprocess.Popen(
            [
                sys.executable,
                str(this_script),
                "--watch",
                "--job", "test_job",
                "--drive-root", str(drive_root),
                "--poll-sec", "0.05",
                "--stale-timeout-sec", "5.0",
                "--wait-for-new-run"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(0.3)
        # Verify proc is still running and did NOT exit immediately on old succeeded run
        if proc1.poll() is not None:
            stdout, stderr = proc1.communicate()
            print("FAILED Scenario 1: Proc exited prematurely on old run!")
            print(f"Stdout:\n{stdout}\nStderr:\n{stderr}")
            return 1

        # Now simulate new run arriving
        new_status = {
            "schema_version": 1,
            "job_name": "test_job",
            "run_id": "run-new-222",
            "state": "succeeded",
            "stage": "done",
            "stage_index": 6,
            "total_stages": 6,
            "updated_at_ts": time.time(),
            "artifacts": [{"path": "model.onnx", "size_bytes": 2048}]
        }
        with open(job_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(new_status, f)

        stdout1, stderr1 = proc1.communicate(timeout=10)
        if proc1.returncode != 0:
            print(f"FAILED Scenario 1: Expected exit code 0, got {proc1.returncode}")
            print(f"Stdout:\n{stdout1}\nStderr:\n{stderr1}")
            return 1
        if "[COLAB_SUCCEEDED]" not in stdout1:
            print("FAILED Scenario 1: '[COLAB_SUCCEEDED]' missing from output!")
            print(f"Stdout:\n{stdout1}")
            return 1
        print("  -> Scenario 1 PASSED: Correctly waited for fresh run and exited 0 on succeeded.")

        # -------------------------------------------------------------
        # Scenario 2: running -> waiting_for_fix exits 10 with raw traceback
        # -------------------------------------------------------------
        print("\n[Scenario 2] Testing waiting_for_fix exit code 10 and traceback output...")
        fix_status = {
            "schema_version": 1,
            "job_name": "test_job",
            "run_id": "run-fix-333",
            "state": "waiting_for_fix",
            "stage": "cell5_qlora_sft",
            "stage_index": 5,
            "total_stages": 6,
            "updated_at_ts": time.time(),
            "rescue_deadline_ts": time.time() + 1800.0,
            "error": {
                "exception_type": "RuntimeError",
                "message": "CUDA out of memory in batch allocation",
                "failed_stage": "cell5_qlora_sft",
                "traceback": "Traceback (most recent call last):\n  File 'train.py', line 42, in sft\n    loss.backward()\nRuntimeError: CUDA out of memory",
                "log_tail": "Last log before OOM"
            }
        }
        with open(job_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(fix_status, f)

        proc2 = subprocess.run(
            [
                sys.executable,
                str(this_script),
                "--watch",
                "--job", "test_job",
                "--drive-root", str(drive_root),
                "--poll-sec", "0.05",
                "--stale-timeout-sec", "5.0"
            ],
            capture_output=True,
            text=True
        )
        if proc2.returncode != 10:
            print(f"FAILED Scenario 2: Expected exit code 10, got {proc2.returncode}")
            print(f"Stdout:\n{proc2.stdout}\nStderr:\n{proc2.stderr}")
            return 1
        if "[COLAB_WAITING_FOR_FIX]" not in proc2.stdout or "CUDA out of memory" not in proc2.stdout:
            print("FAILED Scenario 2: Waiting for fix header or traceback missing!")
            print(f"Stdout:\n{proc2.stdout}")
            return 1
        print("  -> Scenario 2 PASSED: Exited with code 10 and printed raw traceback.")

        # -------------------------------------------------------------
        # Scenario 3: --send-command sync_and_retry syncs repo and writes command.json
        # -------------------------------------------------------------
        print("\n[Scenario 3] Testing --send-command sync_and_retry...")
        mock_repo = tmp_dir / "mock_repo"
        (mock_repo / "training").mkdir(parents=True, exist_ok=True)
        (mock_repo / "tools").mkdir(parents=True, exist_ok=True)
        (mock_repo / "training" / "finetune.py").write_text("print('hotfixed code')\n", encoding="utf-8")
        (mock_repo / "tools" / "helper.py").write_text("# tool code\n", encoding="utf-8")

        proc3 = subprocess.run(
            [
                sys.executable,
                str(this_script),
                "--send-command", "sync_and_retry",
                "--job", "test_job",
                "--drive-root", str(drive_root),
                "--repo-root", str(mock_repo),
                "--note", "Hotfixed OOM batch size"
            ],
            capture_output=True,
            text=True
        )
        if proc3.returncode != 0:
            print(f"FAILED Scenario 3: send_command failed with code {proc3.returncode}")
            print(f"Stdout:\n{proc3.stdout}\nStderr:\n{proc3.stderr}")
            return 1

        synced_file = drive_root / "repo_sync" / "training" / "finetune.py"
        if not synced_file.exists() or "hotfixed code" not in synced_file.read_text(encoding="utf-8"):
            print("FAILED Scenario 3: Synced file missing or content incorrect in repo_sync!")
            return 1

        cmd_file = job_dir / "command.json"
        if not cmd_file.exists():
            print("FAILED Scenario 3: command.json not created!")
            return 1

        with open(cmd_file, "r", encoding="utf-8") as f:
            cmd_json = json.load(f)

        assert cmd_json.get("action") == "sync_and_retry"
        assert cmd_json.get("target_run_id") == "run-fix-333"
        assert cmd_json.get("note") == "Hotfixed OOM batch size"
        assert "cmd_id" in cmd_json
        assert "issued_at_ts" in cmd_json
        print("  -> Scenario 3 PASSED: Sync completed and command.json schema fully verified.")

        # -------------------------------------------------------------
        # Scenario 4: Stale updated_at_ts triggers [COLAB_HEARTBEAT_STALE] (exit 3)
        # -------------------------------------------------------------
        print("\n[Scenario 4] Testing stale heartbeat detection (exit 3)...")
        stale_status = {
            "schema_version": 1,
            "job_name": "test_job",
            "run_id": "run-stale-444",
            "state": "running",
            "stage": "cell4_train",
            "stage_index": 4,
            "total_stages": 6,
            "updated_at_ts": time.time() - 20.0,  # 20 seconds ago, timeout is 0.5s
            "progress": {"step": 50, "total_steps": 100}
        }
        with open(job_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(stale_status, f)

        proc4 = subprocess.run(
            [
                sys.executable,
                str(this_script),
                "--watch",
                "--job", "test_job",
                "--drive-root", str(drive_root),
                "--poll-sec", "0.05",
                "--stale-timeout-sec", "0.5"
            ],
            capture_output=True,
            text=True
        )
        if proc4.returncode != 3:
            print(f"FAILED Scenario 4: Expected exit code 3, got {proc4.returncode}")
            print(f"Stdout:\n{proc4.stdout}\nStderr:\n{proc4.stderr}")
            return 1
        if "[COLAB_HEARTBEAT_STALE]" not in proc4.stdout:
            print("FAILED Scenario 4: '[COLAB_HEARTBEAT_STALE]' missing from output!")
            print(f"Stdout:\n{proc4.stdout}")
            return 1
        print("  -> Scenario 4 PASSED: Correctly caught stale heartbeat and exited with code 3.")

    print("\n" + "=" * 64)
    print("ALL 4 SELF-TEST SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 64)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local PC Side: Zero-Token Blocking Colab Sentinel & Rescue Dispatcher."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--watch", action="store_true", help="Blocking watch mode (default).")
    group.add_argument("--status", action="store_true", help="One-shot non-blocking status check.")
    group.add_argument("--sync-repo", action="store_true", help="Sync training/ and tools/ to Drive repo_sync/.")
    group.add_argument("--send-command", choices=["sync_and_retry", "git_pull_and_retry", "abort"], help="Send command to Colab.")
    group.add_argument("--self-test", action="store_true", help="Run self-test suite covering all 4 verification scenarios.")

    parser.add_argument("--job", help="Target job name (e.g. 'm4_router', 'm3_asr').")
    parser.add_argument("--drive-root", help="Explicit Google Drive data root path.")
    parser.add_argument("--repo-root", help="Repository root path for --sync-repo.")
    parser.add_argument("--poll-sec", type=float, default=DEFAULT_POLL_SEC, help="Poll interval in seconds (default: 20s).")
    parser.add_argument("--stale-timeout-sec", type=float, default=DEFAULT_STALE_TIMEOUT_SEC, help="Heartbeat stale threshold in seconds (default: 300s).")
    parser.add_argument("--max-wait-sec", type=float, default=DEFAULT_MAX_WAIT_SEC, help="Hard maximum wait time in seconds (default: 21600s).")
    parser.add_argument("--wait-for-new-run", action="store_true", help="Wait for a new run if current status is stale or terminal.")
    parser.add_argument("--run-id", help="Explicit target run_id for --send-command.")
    parser.add_argument("--note", help="Optional note string for --send-command.")

    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_tests())

    drive_root = resolve_drive_root(args.drive_root)
    repo_root = pathlib.Path(args.repo_root).resolve() if args.repo_root else pathlib.Path(__file__).resolve().parent.parent

    if args.sync_repo:
        sync_repo(repo_root, drive_root)
        sys.exit(0)

    if args.send_command:
        if not args.job:
            parser.error("--job is required when using --send-command")
        send_command(
            drive_root=drive_root,
            job_name=args.job,
            action=args.send_command,
            target_run_id=args.run_id,
            note=args.note,
            repo_root=repo_root
        )
        sys.exit(0)

    if args.status:
        if not args.job:
            parser.error("--job is required when using --status")
        sys.exit(do_status(drive_root, args.job))

    # Default mode is watch
    if not args.job:
        parser.error("--job is required for monitoring")

    exit_code = do_watch(
        drive_root=drive_root,
        job_name=args.job,
        poll_sec=args.poll_sec,
        stale_timeout_sec=args.stale_timeout_sec,
        max_wait_sec=args.max_wait_sec,
        wait_for_new_run=args.wait_for_new_run
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
