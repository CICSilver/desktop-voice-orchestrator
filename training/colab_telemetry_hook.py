#!/usr/bin/env python3
"""
Desktop Voice Orchestrator — Colab Telemetry & Remote Rescue Hook
================================================================

Pure Python standard-library implementation of the Colab-side telemetry,
progress reporting, heartbeat monitoring, and remote rescue hook.
Syncs atomically to Google Drive under `<drive_data_root>/colab_runs/<job_name>/`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union


class ColabJobHook:
    """Colab job lifecycle, telemetry, heartbeat, and remote rescue hook."""

    def __init__(
        self,
        job_name: str,
        drive_data_root: Union[str, Path] = "/content/drive/MyDrive/desktop-voice-orchestrator-data",
        repo_dir: Union[str, Path] = "/content/desktop-voice-orchestrator",
        total_stages: int = 6,
        heartbeat_interval_sec: float = 30.0,
        rescue_wait_sec: float = 900.0,
        rescue_poll_sec: float = 10.0,
        auto_disconnect: bool = True,
        disconnect_grace_sec: float = 8.0,
    ) -> None:
        self.job_name = str(job_name).strip()
        self.drive_data_root = Path(drive_data_root)
        self.repo_dir = Path(repo_dir)
        self.total_stages = int(total_stages)
        self.heartbeat_interval_sec = float(heartbeat_interval_sec)
        self.rescue_wait_sec = float(rescue_wait_sec)
        self.rescue_poll_sec = float(rescue_poll_sec)
        self.auto_disconnect = bool(auto_disconnect)
        self.disconnect_grace_sec = float(disconnect_grace_sec)

        self.run_dir = self.drive_data_root / "colab_runs" / self.job_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        token = secrets.token_hex(3)
        now_ts = time.time()
        time_prefix = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_ts))
        self.run_id = f"{time_prefix}-{token}"

        self.started_at_ts = now_ts
        self.updated_at_ts = now_ts
        self.heartbeat_seq = 0
        self.state = "starting"
        self.stage = "init"
        self.stage_index = 0
        self.progress: Dict[str, Any] = {
            "step": None,
            "total_steps": None,
            "epoch": None,
            "loss": None,
            "eta_sec": None,
            "message": "starting",
        }
        self.gpu: Dict[str, Any] = self._query_gpu()
        self.git_commit: str = self._query_git_commit()
        self.rescue_deadline_ts: Optional[float] = None
        self.artifacts: List[Dict[str, Any]] = []
        self.summary_metrics: Dict[str, Any] = {}
        self.error: Optional[Dict[str, Any]] = None

        self._consumed_cmd_ids: Set[str] = set()
        self._tail_deque: collections.deque[str] = collections.deque(maxlen=120)
        self._lock = threading.Lock()
        self._last_progress_flush_ts = 0.0
        self._stop_heartbeat = threading.Event()
        self._mock_disconnect_callback: Optional[Callable[[str], None]] = None

        # Write initial events and status
        self._log_event("job_started", {
            "job_name": self.job_name,
            "git_commit": self.git_commit,
            "gpu": self.gpu,
            "total_stages": self.total_stages,
        })
        self._write_status_atomic()

        # Start daemon heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"ColabHook-Heartbeat-{self.job_name}",
        )
        self._heartbeat_thread.start()

    # -------------------------------------------------------------------------
    # GPU and Git Info Gathering
    # -------------------------------------------------------------------------
    def _query_gpu(self) -> Dict[str, Any]:
        """Query GPU name, memory, and utilization via nvidia-smi if available."""
        fallback = {
            "name": "N/A (CPU)",
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "utilization_pct": 0,
        }
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    return {
                        "name": parts[0],
                        "vram_used_mb": int(float(parts[1])),
                        "vram_total_mb": int(float(parts[2])),
                        "utilization_pct": int(float(parts[3])),
                    }
        except Exception:
            pass
        return fallback

    def _query_git_commit(self) -> str:
        """Query short git commit hash of repo_dir if available."""
        try:
            res = subprocess.run(
                ["git", "-C", str(self.repo_dir), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    # -------------------------------------------------------------------------
    # File I/O (Atomic Status & Tail, Append-only Events)
    # -------------------------------------------------------------------------
    def _build_status_dict(self) -> Dict[str, Any]:
        """Build status.json dictionary matching wire schema."""
        return {
            "schema_version": 1,
            "job_name": self.job_name,
            "run_id": self.run_id,
            "state": self.state,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "progress": dict(self.progress),
            "gpu": dict(self.gpu),
            "started_at_ts": self.started_at_ts,
            "updated_at_ts": self.updated_at_ts,
            "heartbeat_seq": self.heartbeat_seq,
            "rescue_deadline_ts": self.rescue_deadline_ts,
            "git_commit": self.git_commit,
            "artifacts": list(self.artifacts),
            "summary_metrics": dict(self.summary_metrics),
            "error": dict(self.error) if self.error is not None else None,
        }

    def _write_status_atomic(self) -> None:
        """Atomically write status.json via .status.<pid>.tmp + fsync + replace."""
        target_path = self.run_dir / "status.json"
        pid = os.getpid()
        tmp_path = self.run_dir / f".status.{pid}.tmp"
        with self._lock:
            data = self._build_status_dict()
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
        except Exception as e:
            # Temporary Drive FUSE write glitch
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

    def _write_tail_atomic(self) -> None:
        """Atomically write rolling last 120 lines to tail.log."""
        target_path = self.run_dir / "tail.log"
        pid = os.getpid()
        tmp_path = self.run_dir / f".tail.{pid}.tmp"
        with self._lock:
            lines = list(self._tail_deque)
        content = "\n".join(lines) + ("\n" if lines else "")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _log_event(self, event_name: str, detail: Optional[Dict[str, Any]] = None) -> None:
        """Append-only record in events.jsonl with fsync."""
        events_path = self.run_dir / "events.jsonl"
        rec = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event_name,
            "detail": detail or {},
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        try:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[ColabJobHook] Warning: failed to append event {event_name}: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Heartbeat Loop
    # -------------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        """Periodically refresh GPU info, heartbeat seq, and updated_at_ts."""
        while not self._stop_heartbeat.wait(self.heartbeat_interval_sec):
            with self._lock:
                self.heartbeat_seq += 1
                self.updated_at_ts = time.time()
                self.gpu = self._query_gpu()
            try:
                self._write_status_atomic()
            except Exception as e:
                print(f"[ColabJobHook] Heartbeat sync notice: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Progress Updating
    # -------------------------------------------------------------------------
    def update_progress(
        self,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        epoch: Optional[float] = None,
        loss: Optional[float] = None,
        eta_sec: Optional[float] = None,
        message: Optional[str] = None,
        force_flush: bool = False,
    ) -> None:
        """Thread-safe progress update throttled to at most once per 5s unless force_flush=True."""
        now = time.time()
        with self._lock:
            if step is not None:
                self.progress["step"] = int(step)
            if total_steps is not None:
                self.progress["total_steps"] = int(total_steps)
            if epoch is not None:
                self.progress["epoch"] = round(float(epoch), 4)
            if loss is not None:
                self.progress["loss"] = round(float(loss), 5)
            if eta_sec is not None:
                self.progress["eta_sec"] = round(float(eta_sec), 1)
            if message is not None:
                self.progress["message"] = str(message)

        if force_flush or (now - self._last_progress_flush_ts >= 5.0):
            with self._lock:
                self.updated_at_ts = now
                self._last_progress_flush_ts = now
            try:
                self._write_status_atomic()
            except Exception as e:
                print(f"[ColabJobHook] Progress flush notice: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Command Stream Execution (Subprocess + Live Console + Tail Deque)
    # -------------------------------------------------------------------------
    def run_cmd(
        self,
        cmd: List[str],
        stage_name: Optional[str] = None,
        cwd: Optional[Union[Path, str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Stream merged stdout/stderr line-by-line, echoing to console and tail.log."""
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env=run_env,
        )

        # Regex patterns for progress hints
        # e.g., "epoch 1/4", "Epoch 2", "step 120/506", "loss: 0.184"
        re_epoch_slash = re.compile(r"epoch\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        re_step_slash = re.compile(r"step\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        re_loss = re.compile(r"(?:loss|train_loss)[:=]\s*([0-9\.]+)", re.IGNORECASE)

        line_count = 0
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            sys.stdout.write(raw_line)
            sys.stdout.flush()

            clean_line = raw_line.rstrip("\r\n")
            with self._lock:
                self._tail_deque.append(clean_line)

            line_count += 1
            if line_count % 20 == 0:
                self._write_tail_atomic()

            # Optional heuristic progress extraction from output stream
            m_ep = re_epoch_slash.search(clean_line)
            m_st = re_step_slash.search(clean_line)
            m_loss = re_loss.search(clean_line)
            if m_ep or m_st or m_loss:
                kwargs: Dict[str, Any] = {}
                if m_ep:
                    kwargs["epoch"] = float(m_ep.group(1))
                if m_st:
                    kwargs["step"] = int(m_st.group(1))
                    kwargs["total_steps"] = int(m_st.group(2))
                if m_loss:
                    try:
                        kwargs["loss"] = float(m_loss.group(1))
                    except ValueError:
                        pass
                self.update_progress(**kwargs)

        proc.wait()
        self._write_tail_atomic()

        if proc.returncode != 0:
            with self._lock:
                tail_str = "\n".join(list(self._tail_deque))
            err = subprocess.CalledProcessError(proc.returncode, cmd, output=tail_str)
            raise err

        return proc.returncode

    # -------------------------------------------------------------------------
    # Stage Runner & Remote Rescue Loop
    # -------------------------------------------------------------------------
    def _poll_command(self) -> Optional[Dict[str, Any]]:
        """Read and validate command.json using standard open(..., 'rb') without mmap."""
        cmd_path = self.run_dir / "command.json"
        if not cmd_path.is_file():
            return None

        # Transient retry in case Drive is syncing write
        for _ in range(3):
            try:
                with open(cmd_path, "rb") as f:
                    content = f.read().decode("utf-8", errors="replace").strip()
                if not content:
                    time.sleep(0.2)
                    continue
                cmd_data = json.loads(content)
                cmd_id = cmd_data.get("cmd_id")
                target_run_id = cmd_data.get("target_run_id")
                action = cmd_data.get("action")

                if not cmd_id or not action:
                    return None

                if target_run_id in ("*", self.run_id) and cmd_id not in self._consumed_cmd_ids:
                    return cmd_data
                return None
            except Exception:
                time.sleep(0.2)
        return None

    def _do_sync(self) -> None:
        """Sync updated files from repo_sync to repo_dir."""
        repo_sync_dir = self.drive_data_root / "repo_sync"
        if repo_sync_dir.is_dir():
            has_rsync = shutil.which("rsync") is not None
            if has_rsync:
                subprocess.run(
                    ["rsync", "-a", f"{repo_sync_dir}/", str(self.repo_dir)],
                    check=True,
                )
            else:
                shutil.copytree(repo_sync_dir, self.repo_dir, dirs_exist_ok=True)
        else:
            subprocess.run(
                ["git", "-C", str(self.repo_dir), "pull", "--ff-only"],
                check=True,
            )

    def _do_git_pull(self) -> None:
        """Run git fetch origin and git pull --ff-only."""
        subprocess.run(["git", "-C", str(self.repo_dir), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(self.repo_dir), "pull", "--ff-only"], check=True)

    def run_stage(
        self,
        stage_name: str,
        stage_index: int,
        fn: Callable[[], Any],
        allow_rescue: bool = True,
    ) -> Any:
        """Execute a pipeline stage with waiting_for_fix remote rescue loop."""
        with self._lock:
            self.state = "running"
            self.stage = stage_name
            self.stage_index = int(stage_index)
            self.updated_at_ts = time.time()
        self._write_status_atomic()
        self._log_event("stage_started", {"stage": stage_name, "stage_index": stage_index})

        while True:
            try:
                result = fn()
                self._log_event("stage_completed", {"stage": stage_name, "stage_index": stage_index})
                return result
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                tb_str = traceback.format_exc()
                with self._lock:
                    tail_lines = list(self._tail_deque)[-80:]
                tail_str = "\n".join(tail_lines)
                err_obj = {
                    "exception_type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": tb_str,
                    "failed_stage": stage_name,
                    "log_tail": tail_str,
                }

                if allow_rescue and self.rescue_wait_sec > 0:
                    with self._lock:
                        self.state = "waiting_for_fix"
                        self.rescue_deadline_ts = time.time() + self.rescue_wait_sec
                        self.error = err_obj
                        self.updated_at_ts = time.time()
                    self._write_status_atomic()
                    self._log_event("waiting_for_fix", {
                        "stage": stage_name,
                        "error": str(exc),
                        "deadline_ts": self.rescue_deadline_ts,
                    })

                    print(
                        f"\n⚠️ Stage failed: {stage_name} ({exc.__class__.__name__}: {exc})\n"
                        f"⚠️ Waiting up to {self.rescue_wait_sec:.0f}s for local agent rescue via Drive command.json..."
                    )

                    rescued = False
                    while True:
                        now = time.time()
                        if self.rescue_deadline_ts is not None and now >= self.rescue_deadline_ts:
                            print("⚠️ Rescue deadline expired.")
                            break
                        if self._stop_heartbeat.is_set():
                            break

                        cmd = self._poll_command()
                        if cmd:
                            action = cmd.get("action")
                            cmd_id = cmd.get("cmd_id")
                            if cmd_id:
                                self._consumed_cmd_ids.add(cmd_id)
                                try:
                                    (self.run_dir / "command.json").unlink(missing_ok=True)
                                except Exception:
                                    pass
                            self._log_event("command_received", {
                                "cmd_id": cmd_id,
                                "action": action,
                                "note": cmd.get("note"),
                            })

                            if action == "sync_and_retry":
                                print(f"🔄 Received rescue command: sync_and_retry ({cmd_id})")
                                self._do_sync()
                                with self._lock:
                                    self.error = None
                                    self.rescue_deadline_ts = None
                                    self.state = "running"
                                    self.updated_at_ts = time.time()
                                self._write_status_atomic()
                                rescued = True
                                break
                            elif action == "git_pull_and_retry":
                                print(f"🔄 Received rescue command: git_pull_and_retry ({cmd_id})")
                                self._do_git_pull()
                                with self._lock:
                                    self.error = None
                                    self.rescue_deadline_ts = None
                                    self.state = "running"
                                    self.updated_at_ts = time.time()
                                self._write_status_atomic()
                                rescued = True
                                break
                            elif action == "abort":
                                print(f"🛑 Received abort command ({cmd_id})")
                                break
                            else:
                                print(f"⚠️ Unknown command action: {action}")

                        time.sleep(self.rescue_poll_sec)

                    if rescued:
                        print(f"🔄 Retrying stage {stage_name}...")
                        continue

                # Stage failed without rescue
                with self._lock:
                    self.state = "failed"
                    self.rescue_deadline_ts = None
                    self.error = err_obj
                    self.updated_at_ts = time.time()
                self._write_status_atomic()
                self._log_event("job_failed", {"stage": stage_name, "error": str(exc)})
                self._stop_heartbeat.set()
                self._maybe_disconnect(reason="failed")
                raise

    # -------------------------------------------------------------------------
    # Job Completion & Disconnect
    # -------------------------------------------------------------------------
    def finish_job(
        self,
        artifacts: Optional[List[Union[str, Path, Dict[str, Any]]]] = None,
        summary_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark job succeeded, resolve artifact sizes, stop heartbeat, and disconnect."""
        resolved_artifacts: List[Dict[str, Any]] = []
        if artifacts:
            for art in artifacts:
                if isinstance(art, dict):
                    resolved_artifacts.append(dict(art))
                else:
                    p = Path(art)
                    size = p.stat().st_size if p.is_file() else 0
                    resolved_artifacts.append({"path": str(p), "size_bytes": size})

        with self._lock:
            self.state = "succeeded"
            self.error = None
            self.artifacts = resolved_artifacts
            self.summary_metrics = summary_metrics or {}
            self.updated_at_ts = time.time()

        self._write_status_atomic()
        self._log_event("job_succeeded", {
            "artifacts_count": len(resolved_artifacts),
            "summary_metrics": self.summary_metrics,
        })
        self._stop_heartbeat.set()
        self._maybe_disconnect(reason="succeeded")

    def _maybe_disconnect(self, reason: str) -> None:
        """Safely unassign Colab compute runtime if auto_disconnect is active."""
        if not self.auto_disconnect:
            return

        if self.disconnect_grace_sec > 0:
            time.sleep(self.disconnect_grace_sec)

        self._log_event("runtime_unassigned", {"reason": reason})

        if self._mock_disconnect_callback is not None:
            self._mock_disconnect_callback(reason)

        try:
            from google.colab import runtime  # type: ignore
            runtime.unassign()
        except (ImportError, AttributeError):
            pass


# -----------------------------------------------------------------------------
# HuggingFace / Transformers Trainer Callback Factory
# -----------------------------------------------------------------------------
def make_hf_trainer_callback(hook: ColabJobHook) -> Any:
    """Create a transformers.TrainerCallback that pushes step/epoch/loss to ColabJobHook."""
    from transformers import TrainerCallback  # type: ignore

    class ColabTelemetryCallback(TrainerCallback):
        def __init__(self, hook_ref: ColabJobHook):
            super().__init__()
            self.hook_ref = hook_ref

        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            if not logs:
                return
            step = state.global_step
            max_steps = state.max_steps
            epoch = logs.get("epoch", state.epoch)
            loss = logs.get("loss")
            self.hook_ref.update_progress(
                step=step,
                total_steps=max_steps,
                epoch=epoch,
                loss=loss,
                message="training_log",
            )

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.hook_ref.update_progress(
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=state.epoch,
            )

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            eval_loss = (metrics or {}).get("eval_loss")
            self.hook_ref.update_progress(
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=state.epoch,
                loss=eval_loss,
                message="evaluate",
                force_flush=True,
            )

    return ColabTelemetryCallback(hook)


# -----------------------------------------------------------------------------
# CLI Self-Test
# -----------------------------------------------------------------------------
def _run_self_tests() -> None:
    """Run full self-test suite covering:

    1) Normal multi-stage job -> succeeded + status.json schema check;
    2) Failing stage -> waiting_for_fix -> remote command.json sync_and_retry -> succeeded;
    3) Rescue timeout -> failed + mock disconnect.
    """
    import tempfile

    print("=== [ColabJobHook] Starting Self-Test Suite ===")

    # Test 1: Normal Multi-Stage Pipeline
    print("\n--- Test 1: Normal Multi-Stage Execution ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        drive_root = tmp_path / "drive_root"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)

        hook = ColabJobHook(
            job_name="test_normal",
            drive_data_root=drive_root,
            repo_dir=repo_dir,
            total_stages=3,
            heartbeat_interval_sec=0.5,
            rescue_wait_sec=2.0,
            rescue_poll_sec=0.2,
            auto_disconnect=True,
            disconnect_grace_sec=0.0,
        )

        def stage1():
            hook.update_progress(step=10, total_steps=100, epoch=0.1, loss=0.5, force_flush=True)
            return "stage1_ok"

        def stage2():
            # Test run_cmd
            hook.run_cmd([sys.executable, "-c", "print('hello from cmd'); print('step 50/100 loss: 0.25')"])
            return "stage2_ok"

        def stage3():
            out_file = repo_dir / "artifact.bin"
            out_file.write_bytes(b"dummy_model_bytes_12345")
            preformed = {"path": "preformed_dict_artifact.bin", "size_bytes": 999}
            hook.finish_job(artifacts=[out_file, preformed], summary_metrics={"accuracy": 0.99})
            return "stage3_ok"

        res1 = hook.run_stage("stage1", 1, stage1)
        res2 = hook.run_stage("stage2", 2, stage2)
        res3 = hook.run_stage("stage3", 3, stage3)
        assert res1 == "stage1_ok" and res2 == "stage2_ok" and res3 == "stage3_ok"

        # Verify status.json schema
        status_file = drive_root / "colab_runs" / "test_normal" / "status.json"
        assert status_file.is_file(), "status.json must exist!"
        status_data = json.loads(status_file.read_text("utf-8"))

        required_keys = [
            "schema_version", "job_name", "run_id", "state", "stage", "stage_index",
            "total_stages", "progress", "gpu", "started_at_ts", "updated_at_ts",
            "heartbeat_seq", "rescue_deadline_ts", "git_commit", "artifacts",
            "summary_metrics", "error",
        ]
        for k in required_keys:
            assert k in status_data, f"Missing key in status.json: {k}"

        assert status_data["schema_version"] == 1
        assert status_data["state"] == "succeeded"
        assert status_data["error"] is None
        assert len(status_data["artifacts"]) == 2
        assert status_data["artifacts"][0]["size_bytes"] == len(b"dummy_model_bytes_12345")
        assert status_data["artifacts"][1]["path"] == "preformed_dict_artifact.bin"
        assert status_data["artifacts"][1]["size_bytes"] == 999
        assert status_data["summary_metrics"]["accuracy"] == 0.99
        print("✅ Test 1 Passed: Normal run reached succeeded, schema matches, and dict-tolerant artifacts verified.")

    # Test 2: Rescue via command.json (sync_and_retry)
    print("\n--- Test 2: Failing Stage Rescue via sync_and_retry ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        drive_root = tmp_path / "drive_root"
        repo_dir = tmp_path / "repo"
        repo_sync = drive_root / "repo_sync"
        repo_dir.mkdir(parents=True)
        repo_sync.mkdir(parents=True)

        hook = ColabJobHook(
            job_name="test_rescue",
            drive_data_root=drive_root,
            repo_dir=repo_dir,
            total_stages=2,
            heartbeat_interval_sec=0.5,
            rescue_wait_sec=5.0,
            rescue_poll_sec=0.2,
            auto_disconnect=False,
            disconnect_grace_sec=0.0,
        )

        code_file = repo_dir / "worker.py"
        code_file.write_text("raise RuntimeError('Original broken code')\n", "utf-8")

        def background_rescuer():
            # Wait until state is waiting_for_fix
            status_file = drive_root / "colab_runs" / "test_rescue" / "status.json"
            for _ in range(50):
                if status_file.is_file():
                    try:
                        st = json.loads(status_file.read_text("utf-8"))
                        if st.get("state") == "waiting_for_fix":
                            break
                    except Exception:
                        pass
                time.sleep(0.1)

            # Prepare fix in repo_sync
            fixed_file = repo_sync / "worker.py"
            fixed_file.write_text("print('Fixed code executed successfully')\n", "utf-8")

            # Write command.json
            cmd_file = drive_root / "colab_runs" / "test_rescue" / "command.json"
            cmd_content = {
                "cmd_id": "cmd-test-rescue-001",
                "target_run_id": hook.run_id,
                "issued_at_ts": time.time(),
                "action": "sync_and_retry",
                "note": "hotfix pushed to repo_sync",
            }
            cmd_file.write_text(json.dumps(cmd_content), "utf-8")

        rescuer_thread = threading.Thread(target=background_rescuer, daemon=True)
        rescuer_thread.start()

        def flaky_stage():
            # Run the worker script
            hook.run_cmd([sys.executable, str(repo_dir / "worker.py")])
            return "flaky_stage_ok"

        res = hook.run_stage("flaky_stage", 1, flaky_stage, allow_rescue=True)
        assert res == "flaky_stage_ok"
        hook.finish_job()

        status_file = drive_root / "colab_runs" / "test_rescue" / "status.json"
        final_status = json.loads(status_file.read_text("utf-8"))
        assert final_status["state"] == "succeeded"
        assert final_status["error"] is None
        cmd_file_check = drive_root / "colab_runs" / "test_rescue" / "command.json"
        assert not cmd_file_check.exists(), "command.json must be unlinked after consumption!"
        print("✅ Test 2 Passed: Waiting_for_fix -> command sync_and_retry -> command.json unlinked -> succeeded.")

    # Test 3: Rescue Timeout -> Failed + Disconnect
    print("\n--- Test 3: Rescue Timeout -> Failed State & Disconnect ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        drive_root = tmp_path / "drive_root"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)

        disconnect_called = []

        hook = ColabJobHook(
            job_name="test_timeout",
            drive_data_root=drive_root,
            repo_dir=repo_dir,
            total_stages=1,
            heartbeat_interval_sec=0.5,
            rescue_wait_sec=0.3,
            rescue_poll_sec=0.1,
            auto_disconnect=True,
            disconnect_grace_sec=0.0,
        )
        hook._mock_disconnect_callback = lambda reason: disconnect_called.append(reason)

        def failing_stage():
            raise ValueError("Unrecoverable error in test_timeout")

        failed_caught = False
        try:
            hook.run_stage("timeout_stage", 1, failing_stage, allow_rescue=True)
        except ValueError:
            failed_caught = True

        assert failed_caught, "Must re-raise the exception after rescue deadline expires!"
        assert disconnect_called == ["failed"], f"Expected disconnect callback with 'failed', got {disconnect_called}"

        status_file = drive_root / "colab_runs" / "test_timeout" / "status.json"
        final_status = json.loads(status_file.read_text("utf-8"))
        assert final_status["state"] == "failed"
        assert final_status["error"] is not None
        assert final_status["error"]["exception_type"] == "ValueError"
        assert "Unrecoverable error" in final_status["error"]["message"]
        print("✅ Test 3 Passed: Rescue timeout cleanly transitioned to failed, error logged, and disconnect triggered.")

    print("\n🎉 ALL SELF-TESTS PASSED SUCCESSFULLY! (COLAB_HOOK_SELF_TEST_OK)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Telemetry & Remote Rescue Hook")
    parser.add_argument("--self-test", action="store_true", help="Run self-test suite")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
    else:
        parser.print_help()
