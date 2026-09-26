"""Export recorded sessions into a training dataset of per-take clips.

Reads data/sessions/<session>/{labels.json, mic.wav, loopback.wav, manifest.json}
and writes, without touching the sessions:

    <out>/clips/<session>/<clip>.wav       microphone, 16 kHz mono int16
    <out>/reference/<session>/<clip>.wav   speaker loopback, same span and format
    <out>/manifest.jsonl                   one JSON object per clip
    <out>/summary.json                     counts and hours per split/kind/condition

Splits:
    test       evaluation sessions recorded with the current wake word, and the second
               half of each daily background recording; never train on these
    train      training-mode sessions (labels "purpose": "train"), evaluation sessions
               recorded with an earlier wake word, and the first half of each daily
               background recording
    dev        a fixed 10% of training-mode takes, chosen by a hash of the clip id
    unlabeled  sessions without labels, cut into fixed-length chunks

Re-running regenerates everything, so new sessions are picked up by running it again.

    python tools/export_dataset.py [--sessions data/sessions] [--out data/dataset]
                                   [--wake 小克]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

RATE = 16000
PAD_BEFORE_S = 0.2
# The take ends when the space bar is pressed; its click lands 0.1-0.2 s later.
PAD_AFTER_S = 0.05
CHUNK_S = 10.0
DEV_PERCENT = 10
# Same threshold as the evaluator: loopback louder than this means music played.
MUSIC_DBFS = -45.0


def read_float_wav(path: Path) -> tuple[int, np.ndarray]:
    """Returns (rate, samples[frames, channels]) of a 32-bit float WAV."""
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, rate, channels, bits, tag = 12, 0, 0, 0, 0
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", data[pos + 8:pos + 16])
            bits = struct.unpack("<H", data[pos + 22:pos + 24])[0]
        elif chunk == b"data":
            if tag not in (3, 0xFFFE) or bits != 32:
                raise ValueError(f"{path}: expected 32-bit float samples")
            available = len(data) - pos - 8
            # A recording that was never finalized still has a zero-size header.
            if size == 0 or size > available:
                size = available
            size = size // (4 * channels) * (4 * channels)
            samples = np.frombuffer(data, dtype="<f4", count=size // 4, offset=pos + 8)
            return rate, samples.reshape(-1, channels)
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path} has no data chunk")


def resample(signal: np.ndarray, rate: int) -> np.ndarray:
    """Band-limited resampling to 16 kHz for integer ratios (48 kHz loopback)."""
    if rate == RATE:
        return signal
    if rate % RATE:
        raise ValueError(f"unsupported rate {rate}")
    factor = rate // RATE
    n = len(signal) // factor * factor
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    spectrum = np.fft.rfft(signal[:n])
    spectrum[len(spectrum) // factor:] = 0
    return np.fft.irfft(spectrum, n)[::factor].astype(np.float32)


def write_pcm16(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())


def level_dbfs(signal: np.ndarray) -> float | None:
    if len(signal) == 0:
        return None
    rms = math.sqrt(float(np.mean(np.square(signal, dtype=np.float64))))
    return round(20 * math.log10(rms), 1) if rms > 1e-6 else -120.0


def frame_db(signal: np.ndarray, frame: int = 320) -> np.ndarray:
    n = len(signal) // frame
    if n == 0:
        return np.zeros(0)
    frames = signal[:n * frame].astype(np.float64).reshape(n, frame)
    return 10 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)


def speech_bounds(signal: np.ndarray) -> list[float] | None:
    """Energy-based speech span in seconds, for quiet recordings only.

    Frames 12 dB over the clip's noise floor and within 20 dB of its loudest
    frame form segments (the second rule keeps a fluctuating room-noise floor
    out); gaps under 0.3 s are joined and segments under 0.12 s (key clicks)
    dropped. Advisory only: clips are never cut to these bounds.
    """
    db = frame_db(signal)
    if len(db) < 5:
        return None
    threshold = max(float(np.percentile(db, 10)) + 12.0, float(np.percentile(db, 99)) - 20.0, -55.0)
    active = np.flatnonzero(db > threshold)
    if len(active) == 0:
        return None
    frame_s = 0.02
    segments, start, last = [], active[0], active[0]
    for index in active[1:]:
        if (index - last) * frame_s > 0.3:
            segments.append((start, last))
            start = index
        last = index
    segments.append((start, last))
    segments = [(a, b) for a, b in segments if (b - a + 1) * frame_s >= 0.12]
    if not segments:
        return None
    return [round(segments[0][0] * frame_s, 2), round((segments[-1][1] + 1) * frame_s, 2)]


def normalize_text(text: str) -> str:
    """Characters a recognizer should output: CJK, letters, digits and %."""
    return "".join(re.findall(r"[㐀-䶿一-鿿A-Za-z0-9%]", text))


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        rate, mic = read_float_wav(path / "mic.wav")
        # Early sessions captured the microphone at 48 kHz.
        self.mic = resample(mic[:, 0].astype(np.float32), rate)
        self.loopback = None
        self.loopback_rate = 0
        self.loopback_offset_s = 0.0
        if (path / "loopback.wav").is_file():
            self.loopback_rate, loopback = read_float_wav(path / "loopback.wav")
            self.loopback = loopback.mean(axis=1).astype(np.float32)
            streams = {}
            if (path / "manifest.json").is_file():
                streams = json.loads((path / "manifest.json").read_text("utf-8")).get("streams", {})
            first_mic = streams.get("microphone", {}).get("first_qpc_100ns")
            first_loop = streams.get("loopback", {}).get("first_qpc_100ns")
            if first_mic and first_loop:
                # Loopback sample 0 was captured this much later than mic sample 0.
                self.loopback_offset_s = (first_loop - first_mic) / 1e7

    @property
    def duration_s(self) -> float:
        return len(self.mic) / RATE

    def microphone(self, start_s: float, end_s: float) -> np.ndarray:
        return self.mic[max(0, int(start_s * RATE)):max(0, int(end_s * RATE))]

    def reference(self, start_s: float, end_s: float) -> np.ndarray | None:
        if self.loopback is None:
            return None
        rate = self.loopback_rate
        begin = int((start_s - self.loopback_offset_s) * rate)
        end = int((end_s - self.loopback_offset_s) * rate)
        piece = self.loopback[max(0, begin):max(0, end)]
        if begin < 0:
            piece = np.concatenate([np.zeros(-begin, dtype=np.float32), piece])
        want = end - begin
        if len(piece) < want:
            piece = np.concatenate([piece, np.zeros(want - len(piece), dtype=np.float32)])
        return resample(piece, rate)


def split_for(labels: dict | None, clip_id: str, wake: str,
              position: float | None = None) -> str:
    """position is a background chunk's place in its recording, 0..1."""
    if labels is None:
        return "unlabeled"
    if labels.get("purpose") == "train":
        digest = int(hashlib.sha1(clip_id.encode("utf-8")).hexdigest(), 16)
        return "dev" if digest % 100 < DEV_PERCENT else "train"
    if position is not None:
        # Daily background serves both as augmentation noise and wake negatives
        # (first half) and as the false-execution regression set (second half);
        # splitting by time keeps the two apart.
        return "train" if position < 0.5 else "test"
    # Evaluation sessions with the current wake word are the held-out test set;
    # those recorded with an earlier wake word cannot score it and train instead.
    return "test" if labels.get("wake_word") == wake else "train"


def export_clip(session: Session, out: Path, clip: str, start_s: float, end_s: float,
                record: dict, bounds: bool) -> dict | None:
    start_s = max(0.0, start_s)
    end_s = min(session.duration_s, end_s)
    if end_s - start_s < 0.3:
        return None
    audio = session.microphone(start_s, end_s)
    reference = session.reference(start_s, end_s)
    relative = f"{session.name}/{clip}.wav"
    write_pcm16(out / "clips" / relative, audio)
    music_dbfs = level_dbfs(reference) if reference is not None else None
    music = music_dbfs is not None and music_dbfs > MUSIC_DBFS
    if reference is not None:
        write_pcm16(out / "reference" / relative, reference)
    record.update({
        "id": f"{session.name}/{clip}",
        "audio": f"clips/{relative}",
        "reference": f"reference/{relative}" if reference is not None else None,
        "session": session.name,
        "start_s": round(start_s, 3),
        "duration_s": round(len(audio) / RATE, 3),
        "mic_dbfs": level_dbfs(audio),
        "music_dbfs": music_dbfs,
        "music": music,
        # Energy bounds are meaningless under music; they are left empty there.
        "speech_s": speech_bounds(audio) if bounds and not music else None,
    })
    return record


def export_labeled(session: Session, labels: dict, out: Path, wake: str) -> list[dict]:
    group = labels.get("group", {})
    records = []
    for take in labels.get("takes", []):
        if take.get("status", "ok") != "ok":
            continue  # re-recorded or skipped: what was said is unknown
        kind = take.get("kind", "")
        start_s = take["start_ms"] / 1000.0
        end_s = take["end_ms"] / 1000.0
        base = {
            "collection_id": labels.get("collection_id"),
            "purpose": labels.get("purpose", "evaluate"),
            "group": group.get("id"),
            "condition": group.get("condition"),
            "distance_m": group.get("distance_m"),
            "speaker": labels.get("speaker"),
            "wake_word": labels.get("wake_word"),
            "kind": kind,
            "wake_position": take.get("wake_position", "none"),
            "prompt": take.get("prompt", ""),
            "text": normalize_text(take.get("text", "")),
            "expected_actions": take.get("expected_actions", []),
            "attempt": take.get("attempt", 1),
        }
        if kind in ("silence", "background"):
            # No speech expected: long spans become fixed-length chunks.
            pieces = max(1, math.ceil((end_s - start_s) / CHUNK_S - 1e-9))
            for piece in range(pieces):
                a = start_s + piece * CHUNK_S
                b = min(end_s, a + CHUNK_S)
                clip = take["id"] if pieces == 1 else f"{take['id']}-{piece + 1:03d}"
                position = piece / pieces if kind == "background" else None
                record = {**base, "split": split_for(labels, f"{session.name}/{clip}", wake, position)}
                exported = export_clip(session, out, clip, a, b, record, bounds=False)
                if exported:
                    records.append(exported)
            continue
        clip = take["id"]
        record = {**base, "split": split_for(labels, f"{session.name}/{clip}", wake)}
        exported = export_clip(session, out, clip, start_s - PAD_BEFORE_S, end_s + PAD_AFTER_S,
                               record, bounds=True)
        if exported:
            records.append(exported)
    return records


def export_unlabeled(session: Session, out: Path) -> list[dict]:
    records = []
    pieces = math.ceil(session.duration_s / CHUNK_S)
    for piece in range(pieces):
        a = piece * CHUNK_S
        record = {"purpose": "unlabeled", "kind": "unlabeled", "split": "unlabeled",
                  "text": None, "wake_word": None}
        exported = export_clip(session, out, f"chunk-{piece + 1:04d}", a, a + CHUNK_S, record,
                               bounds=False)
        if exported:
            records.append(exported)
    return records


def summarize(records: list[dict]) -> dict:
    table: dict = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
    for record in records:
        condition = "music" if record["music"] else "quiet"
        key = f"{record['split']}/{record['kind']}/{condition}"
        table[key]["clips"] += 1
        table[key]["seconds"] += record["duration_s"]
    totals: dict = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
    for key, value in table.items():
        split = key.split("/")[0]
        totals[split]["clips"] += value["clips"]
        totals[split]["seconds"] += value["seconds"]
    round_seconds = lambda d: {k: {"clips": v["clips"], "seconds": round(v["seconds"], 1)}
                               for k, v in sorted(d.items())}
    return {"splits": round_seconds(totals), "detail": round_seconds(table)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sessions", type=Path, default=Path("data/sessions"))
    parser.add_argument("--out", type=Path, default=Path("data/dataset"))
    parser.add_argument("--wake", default="小克", help="current wake word; its evaluation sessions become test")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    records, skipped = [], []
    for path in sorted(p for p in args.sessions.iterdir() if p.is_dir()):
        if not (path / "mic.wav").is_file():
            skipped.append(f"{path.name}: no mic.wav")
            continue
        labels = None
        if (path / "labels.json").is_file():
            labels = json.loads((path / "labels.json").read_text("utf-8"))
        try:
            session = Session(path)
        except ValueError as error:
            skipped.append(f"{path.name}: {error}")
            continue
        exported = export_labeled(session, labels, args.out, args.wake) if labels \
            else export_unlabeled(session, args.out)
        records.extend(exported)
        print(f"{path.name}: {len(exported)} clips ({'labeled' if labels else 'unlabeled'})")

    with open(args.out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    summary["wake_word"] = args.wake
    summary["skipped_sessions"] = skipped
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           "utf-8")
    print(json.dumps(summary["splits"], ensure_ascii=False))
    for line in skipped:
        print("skipped", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
