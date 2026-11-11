"""Mine natural speech from headset gaming/background sessions with dual-ASR cross-verification.

Designed for 1-2 hour background sessions recorded while wearing headphones:
  * Strictly reads ONLY `mic.wav` (never `loopback.wav`, which contains teammates/game audio);
  * Supports 32-bit float or 16-bit PCM WAV at any integer multiple of 16 kHz (e.g. 48 kHz)
    with band-limited anti-aliased resampling to 16 kHz mono;
  * Strictly honors `status == "discarded"` (and any non-ok) exclusion spans in `labels.json`;
  * Segments natural utterances using `models/silero_vad.int8.onnx` (when `onnxruntime` is
    installed) or an adaptive dual-threshold energy VAD fallback, padding 0.15 s before and
    0.20 s after each utterance;
  * Acoustic gating: drops shouted/clipped segments (`peak_dbfs > -1.5 dBFS`), short filler
    exclamations (`< 1.5 s`), and overly long glued spans (`> 12.0 s`);
  * Flags clean medium-length steady clips (`3.0 s ~ 7.5 s`, high SNR, steady level) as
    `prompt_candidate = True` for optional inclusion in the zero-shot TTS reference pool;
  * Optional `--transcribe` mode (for Colab GPU): runs SenseVoice + faster-whisper `large-v3`
    cross-verification. Only when both heterogeneous models produce 100% character-identical
    normalized Chinese text with >= 3 characters is a clip emitted as `kind="game_verified"`
    positive ASR training data; disagreed clips are deterministically split 50/50 into
    `train` unlabeled negative pool and `test` false-wake / false-execution stress test set.

Usage:
    # Local VAD mining + acoustic filtering:
    python tools/mine_game_sessions.py --sessions data/sessions --out data/game_mined

    # Colab GPU dual-model pseudo-labeling:
    python tools/mine_game_sessions.py --sessions /path/to/game_sessions --out /path/to/out \\
        --transcribe --sensevoice-dir /path/to/sherpa-sense-voice --whisper-model large-v3

    # Run built-in synthetic waveform unit test:
    python tools/mine_game_sessions.py --self-test
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
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

RATE = 16000
VAD_WINDOW = 512  # 32 ms @ 16 kHz
PAD_BEFORE_S = 0.15
PAD_AFTER_S = 0.20
MIN_DURATION_S = 1.5
MAX_DURATION_S = 12.0
MAX_PEAK_DBFS = -1.5
MIN_PEAK_DBFS = -30.0

# Common unambiguous Traditional -> Simplified character map for Whisper output normalization
TRAD_TO_SIMP = str.maketrans(
    "後發開關聽說語音樂聲點換選單隨機循環愛喜歡電腦網際線視訊畫面時間問題長短開頭結尾"
    "這裏裡為甚麼麼個們來時會過對進還沒有現實際應該讓給從當與將並車東馬鳥魚風雲龍門間"
    "體國學區書寫讀見覺親觀買賣價錢數號塊條雙隻張頁節課級隊員場賽輸贏殺敵槍彈藥護頭盔",
    "后发开关听说语音乐声点换选单随机循环爱喜欢电脑网际线视讯画面时间问题长短开头结尾"
    "这里里为什么么个们来时会过对进还没有现实际应该让给从当与将并车东马鸟鱼风云龙门间"
    "体国学区书写读见觉亲观买卖价钱数号块条双只张页节课级队员场赛输赢杀敌枪弹药护头盔",
)


def read_mic_wav(path: Path) -> tuple[int, np.ndarray]:
    """Reads `mic.wav` (32-bit float or 16/32-bit PCM) and returns (rate, mono_float32)."""
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a valid WAV file")
    pos, rate, channels, bits, tag = 12, 0, 0, 0, 0
    while pos + 8 <= len(data):
        chunk = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", data[pos + 8:pos + 16])
            bits = struct.unpack("<H", data[pos + 22:pos + 24])[0]
            if tag == 0xFFFE and size >= 40:
                tag = struct.unpack("<H", data[pos + 32:pos + 34])[0]
        elif chunk == b"data":
            avail = len(data) - pos - 8
            if size == 0 or size > avail:
                size = avail
            bytes_per_sample = max(1, bits // 8)
            frame_bytes = bytes_per_sample * max(1, channels)
            size = (size // frame_bytes) * frame_bytes
            buf = data[pos + 8:pos + 8 + size]
            if tag == 3 and bits == 32:
                arr = np.frombuffer(buf, dtype="<f4").reshape(-1, channels)[:, 0].astype(np.float32)
            elif tag == 1 and bits == 16:
                arr = np.frombuffer(buf, dtype="<i2").reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0
            elif tag == 1 and bits == 32:
                arr = np.frombuffer(buf, dtype="<i4").reshape(-1, channels)[:, 0].astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"{path}: unsupported WAV encoding tag={tag} bits={bits}")
            return rate, arr
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path} has no data chunk")


def resample_bandlimited(signal: np.ndarray, rate: int) -> np.ndarray:
    """Band-limited anti-aliased resampling to 16 kHz."""
    if rate == RATE:
        return signal
    if len(signal) == 0:
        return np.zeros(0, dtype=np.float32)
    if rate % RATE == 0:
        factor = rate // RATE
        n = (len(signal) // factor) * factor
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # Process in 60-second blocks so multi-hour recordings stay fast and low-memory
        block_in = 60 * rate
        out_parts: list[np.ndarray] = []
        for start in range(0, n, block_in):
            piece = signal[start:min(n, start + block_in)]
            m = (len(piece) // factor) * factor
            if m == 0:
                continue
            spec = np.fft.rfft(piece[:m])
            spec[len(spec) // factor:] = 0
            out_parts.append(np.fft.irfft(spec, m)[::factor].astype(np.float32))
        return np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.float32)

    target_len = int(round(len(signal) * RATE / rate))
    if target_len == 0:
        return np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(signal)
    out_spec = np.zeros(target_len // 2 + 1, dtype=np.complex128)
    bins = min(len(spec), len(out_spec))
    out_spec[:bins] = spec[:bins]
    return (np.fft.irfft(out_spec, target_len) * (target_len / len(signal))).astype(np.float32)


def write_pcm16(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())


def write_float32_wav(path: Path, signal: np.ndarray, rate: int) -> None:
    """Helper for synthetic self-tests: writes IEEE 32-bit float mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_samples = np.asarray(signal, dtype="<f4").tobytes()
    data_size = len(raw_samples)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, 1, rate, rate * 4, 4, 32)
    data_hdr = struct.pack("<4sI", b"data", data_size)
    riff_hdr = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_hdr) + data_size, b"WAVE")
    path.write_bytes(riff_hdr + fmt_chunk + data_hdr + raw_samples)


def load_discarded_spans(labels: dict | None) -> list[tuple[float, float]]:
    """Extracts excluded time intervals (in seconds) where status == 'discarded' (or != 'ok')."""
    if not labels:
        return []
    spans: list[tuple[float, float]] = []
    for take in labels.get("takes", []):
        status = take.get("status", "ok")
        if status == "discarded" or status != "ok":
            if "start_ms" in take and "end_ms" in take:
                s0 = max(0.0, float(take["start_ms"]) / 1000.0 - 0.05)
                s1 = float(take["end_ms"]) / 1000.0 + 0.05
                if s1 > s0:
                    spans.append((s0, s1))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for s0, s1 in spans:
        if merged and s0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
        else:
            merged.append((s0, s1))
    return merged


def overlaps_any(start_s: float, end_s: float, excluded: list[tuple[float, float]]) -> bool:
    for ex0, ex1 in excluded:
        if start_s < ex1 and end_s > ex0:
            return True
    return False


def clamp_away_from_excluded(
    start_s: float, end_s: float, core_start_s: float, core_end_s: float,
    excluded: list[tuple[float, float]], total_s: float
) -> tuple[float, float] | None:
    """Ensures neither core speech nor padding crosses into a discarded span."""
    if overlaps_any(core_start_s, core_end_s, excluded):
        return None
    s0 = max(0.0, start_s)
    s1 = min(total_s, end_s)
    for ex0, ex1 in excluded:
        if ex1 <= core_start_s and s0 < ex1:
            s0 = ex1
        if ex0 >= core_end_s and s1 > ex0:
            s1 = ex0
    return (s0, s1) if s1 > s0 else None


def segment_with_silero_vad(
    signal: np.ndarray,
    model_path: Path,
    onset: float = 0.50,
    offset: float = 0.35,
    min_silence_s: float = 0.45,
) -> list[tuple[float, float]]:
    """Runs Silero VAD v4 ONNX with dual-threshold hysteresis."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    n_frames = len(signal) // VAD_WINDOW
    frame_s = VAD_WINDOW / RATE  # 0.032 s
    min_silence_frames = max(1, int(round(min_silence_s / frame_s)))

    spans: list[tuple[float, float]] = []
    in_speech = False
    speech_start = 0
    last_voice = 0

    for i in range(n_frames):
        chunk = signal[i * VAD_WINDOW:(i + 1) * VAD_WINDOW].reshape(1, VAD_WINDOW)
        prob_out, h, c = sess.run(None, {"x": chunk, "h": h, "c": c})
        p = float(prob_out[0, 0])
        if not in_speech:
            if p >= onset:
                in_speech = True
                speech_start = i
                last_voice = i
        else:
            if p >= offset:
                last_voice = i
            elif (i - last_voice) >= min_silence_frames:
                spans.append((speech_start * frame_s, (last_voice + 1) * frame_s))
                in_speech = False
                # Reset recurrent state softly on long silences so drift never accumulates
                h.fill(0.0)
                c.fill(0.0)

    if in_speech:
        spans.append((speech_start * frame_s, (last_voice + 1) * frame_s))
    return spans


def segment_with_energy_vad(
    signal: np.ndarray,
    min_silence_s: float = 0.45,
) -> list[tuple[float, float]]:
    """Adaptive dual-threshold energy VAD fallback when onnxruntime is unavailable."""
    hop = 320  # 20 ms @ 16 kHz
    frame_s = hop / RATE
    n_frames = len(signal) // hop
    if n_frames == 0:
        return []
    frames = signal[:n_frames * hop].astype(np.float64).reshape(n_frames, hop)
    db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)

    noise_floor = min(float(np.percentile(db, 20)), -40.0)
    high_th = max(noise_floor + 13.0, -44.0)
    low_th = max(noise_floor + 8.0, -49.0)
    min_silence_frames = max(1, int(round(min_silence_s / frame_s)))

    spans: list[tuple[float, float]] = []
    in_speech = False
    speech_start = 0
    last_voice = 0

    for i, val in enumerate(db):
        if not in_speech:
            if val >= high_th:
                in_speech = True
                speech_start = i
                last_voice = i
        else:
            if val >= low_th:
                last_voice = i
            elif (i - last_voice) >= min_silence_frames:
                spans.append((speech_start * frame_s, (last_voice + 1) * frame_s))
                in_speech = False

    if in_speech:
        spans.append((speech_start * frame_s, (last_voice + 1) * frame_s))
    return spans


def analyze_clip_acoustics(audio: np.ndarray) -> dict:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    peak_dbfs = 20.0 * math.log10(peak) if peak > 1e-6 else -120.0
    rms = math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    rms_db = 20.0 * math.log10(rms) if rms > 1e-6 else -120.0

    hop = 320
    n_f = len(audio) // hop
    if n_f >= 4:
        frames = audio[:n_f * hop].astype(np.float64).reshape(n_f, hop)
        f_db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
        noise_floor_db = min(float(np.percentile(f_db, 15)), -38.0)
        voiced = f_db[f_db > noise_floor_db + 8.0]
        if len(voiced) >= 3:
            speech_db = float(np.percentile(voiced, 85))
            voiced_std_db = float(np.std(voiced))
        else:
            speech_db = float(np.max(f_db))
            voiced_std_db = 12.0
        snr_db = speech_db - noise_floor_db
    else:
        snr_db = 0.0
        voiced_std_db = 12.0

    dur_s = len(audio) / RATE
    is_prompt_cand = (
        3.0 <= dur_s <= 7.5
        and -20.0 <= peak_dbfs <= -2.0
        and snr_db >= 20.0
        and voiced_std_db <= 8.5
    )
    return {
        "duration_s": round(dur_s, 3),
        "peak_dbfs": round(peak_dbfs, 2),
        "mic_dbfs": round(rms_db, 1),
        "snr_db": round(snr_db, 1),
        "level_std_db": round(voiced_std_db, 2),
        "prompt_candidate": bool(is_prompt_cand),
    }


def normalize_zh(text: str) -> str:
    """Strips model tags/punctuation, unifies unambiguous Traditional->Simplified, keeps CJK."""
    cleaned = re.sub(r"<\|[^|]*\|>", "", text)
    cleaned = cleaned.translate(TRAD_TO_SIMP)
    return "".join(re.findall(r"[\u4e00-\u9fff]", cleaned))


class DualAsrVerifier:
    """Heterogeneous dual-ASR verifier: SenseVoice (CTC) + faster-whisper large-v3 (AED)."""

    def __init__(self, sensevoice_dir: Path, whisper_model: str, device: str = "cuda"):
        import sherpa_onnx
        from faster_whisper import WhisperModel

        model_onnx = sensevoice_dir / "model.int8.onnx"
        if not model_onnx.is_file():
            model_onnx = sensevoice_dir / "model.onnx"
        tokens_txt = sensevoice_dir / "tokens.txt"
        if not model_onnx.is_file() or not tokens_txt.is_file():
            raise FileNotFoundError(
                f"SenseVoice model/tokens not found under {sensevoice_dir}"
            )
        self.sensevoice = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_onnx),
            tokens=str(tokens_txt),
            language="zh",
            use_itn=False,
            num_threads=4,
        )
        compute_type = "float16" if device == "cuda" else "int8"
        self.whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    def transcribe_pair(self, wav_path: Path, audio_16k: np.ndarray) -> tuple[str, str, str, str]:
        stream = self.sensevoice.create_stream()
        stream.accept_waveform(RATE, audio_16k.tolist())
        self.sensevoice.decode_stream(stream)
        raw_sv = stream.result.text.strip()

        segments, _ = self.whisper.transcribe(
            str(wav_path),
            language="zh",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt="以下是普通话简体中文日常语音。",
        )
        raw_wh = "".join(seg.text for seg in segments).strip()
        return raw_sv, raw_wh, normalize_zh(raw_sv), normalize_zh(raw_wh)


def disagree_split(session_name: str, clip_id: str, total_sessions: int) -> str:
    """Splits disagreed clips 50/50 into train (negatives) and test (false-wake stress test).

    Uses session-name hashing when >= 4 sessions exist so an entire session stays on one side;
    falls back to clip-id hashing when only 1-3 long sessions were recorded so both splits
    still receive a balanced ~50/50 share.
    """
    key = session_name if total_sessions >= 4 else clip_id
    digest = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return "train" if (digest % 2 == 0) else "test"


def mine_sessions(
    sessions_root: Path,
    out_dir: Path,
    vad_model: Path | None = None,
    force_energy_vad: bool = False,
    include_labeled_non_bg: bool = False,
    transcribe: bool = False,
    sensevoice_dir: Path | None = None,
    whisper_model: str = "large-v3",
    device: str = "cuda",
) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_silero = False
    if not force_energy_vad and vad_model is not None and vad_model.is_file():
        try:
            import onnxruntime  # noqa: F401
            use_silero = True
        except ImportError:
            use_silero = False

    verifier = None
    if transcribe:
        if sensevoice_dir is None:
            raise ValueError("--sensevoice-dir is required when --transcribe is enabled")
        verifier = DualAsrVerifier(sensevoice_dir, whisper_model, device=device)

    # Support passing either a single session directory (containing mic.wav) or a parent folder
    if (sessions_root / "mic.wav").is_file():
        session_dirs = [sessions_root]
    else:
        session_dirs = sorted(p for p in sessions_root.iterdir() if p.is_dir())

    eligible_sessions: list[tuple[Path, dict | None]] = []
    skipped_sessions: list[str] = []

    for sdir in session_dirs:
        if not (sdir / "mic.wav").is_file():
            skipped_sessions.append(f"{sdir.name}: no mic.wav")
            continue
        labels = None
        if (sdir / "labels.json").is_file():
            labels = json.loads((sdir / "labels.json").read_text("utf-8"))
            if not include_labeled_non_bg:
                cond = labels.get("group", {}).get("condition")
                if cond not in (None, "background"):
                    skipped_sessions.append(f"{sdir.name}: skipped structured {cond} session")
                    continue
        eligible_sessions.append((sdir, labels))

    stats: dict[str, int] = defaultdict(int)
    stats["vad_backend_silero"] = int(use_silero)
    stats["sessions_processed"] = len(eligible_sessions)

    manifest_records: list[dict] = []
    prompt_candidates: list[dict] = []

    for sdir, labels in eligible_sessions:
        rate, raw_mic = read_mic_wav(sdir / "mic.wav")
        mic_16k = resample_bandlimited(raw_mic, rate)
        total_s = len(mic_16k) / RATE
        excluded = load_discarded_spans(labels)

        raw_spans = (
            segment_with_silero_vad(mic_16k, vad_model)
            if use_silero and vad_model is not None
            else segment_with_energy_vad(mic_16k)
        )
        stats["raw_vad_segments"] += len(raw_spans)

        clip_idx = 0
        for core_s0, core_s1 in raw_spans:
            clamped = clamp_away_from_excluded(
                core_s0 - PAD_BEFORE_S,
                core_s1 + PAD_AFTER_S,
                core_s0,
                core_s1,
                excluded,
                total_s,
            )
            if clamped is None:
                stats["rejected_discarded_span"] += 1
                continue
            s0, s1 = clamped
            dur = s1 - s0
            if dur < MIN_DURATION_S:
                stats["rejected_too_short"] += 1
                continue
            if dur > MAX_DURATION_S:
                stats["rejected_too_long"] += 1
                continue

            audio = mic_16k[int(round(s0 * RATE)):int(round(s1 * RATE))]
            acoustics = analyze_clip_acoustics(audio)
            if acoustics["peak_dbfs"] > MAX_PEAK_DBFS:
                stats["rejected_clipping"] += 1
                continue
            if acoustics["peak_dbfs"] < MIN_PEAK_DBFS:
                stats["rejected_too_quiet"] += 1
                continue

            clip_idx += 1
            clip_name = f"game-{clip_idx:04d}"
            rel_wav = f"clips/{sdir.name}/{clip_name}.wav"
            wav_out = out_dir / rel_wav
            write_pcm16(wav_out, audio)
            clip_id = f"{sdir.name}/{clip_name}"

            rec = {
                "id": clip_id,
                "audio": rel_wav,
                "reference": None,
                "session": sdir.name,
                "start_s": round(s0, 3),
                **acoustics,
                "music": False,
                "condition": "quiet",
            }

            if verifier is not None:
                raw_sv, raw_wh, norm_sv, norm_wh = verifier.transcribe_pair(wav_out, audio)
                rec["sensevoice_raw"] = raw_sv
                rec["whisper_raw"] = raw_wh
                if norm_sv and norm_sv == norm_wh and len(norm_sv) >= 3:
                    rec["kind"] = "game_verified"
                    rec["split"] = "train"
                    rec["text"] = norm_sv
                    rec["prompt"] = norm_sv
                    stats["verified_exact_match"] += 1
                else:
                    assigned_split = disagree_split(sdir.name, clip_id, len(eligible_sessions))
                    rec["kind"] = "background"
                    rec["split"] = assigned_split
                    rec["text"] = ""
                    rec["prompt"] = ""
                    stats[f"disagreed_to_{assigned_split}"] += 1
            else:
                rec["kind"] = "unlabeled"
                rec["split"] = "unlabeled"
                rec["text"] = None
                rec["prompt"] = ""

            stats["kept_clips"] += 1
            manifest_records.append(rec)
            if rec["prompt_candidate"]:
                stats["prompt_candidates"] += 1
                prompt_candidates.append({
                    "id": clip_id.replace("/", "__"),
                    "wav_path": rel_wav,
                    "prompt_text": rec.get("text") or "",
                    "duration_s": rec["duration_s"],
                    "peak_dbfs": rec["peak_dbfs"],
                    "snr_db": rec["snr_db"],
                    "verified": rec["kind"] == "game_verified",
                })

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "prompt_candidates.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in prompt_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = {
        "stats": dict(stats),
        "skipped_sessions": skipped_sessions,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return summary


def synth_speech_like_burst(rng: np.random.Generator, rate: int, duration_s: float, peak_amp: float) -> np.ndarray:
    """Creates harmonic speech-like voiced formant bursts so both Silero VAD and energy VAD respond."""
    t = np.arange(int(round(duration_s * rate)), dtype=np.float64) / rate
    f0 = 135.0 + 15.0 * np.sin(2.0 * math.pi * 2.2 * t)
    phase = 2.0 * math.pi * np.cumsum(f0) / rate
    sig = np.zeros_like(t)
    for k, weight in [(1, 1.0), (2, 0.7), (3, 0.45), (4, 0.3), (6, 0.2), (9, 0.12), (14, 0.06)]:
        sig += weight * np.sin(k * phase)
    # Syllabic amplitude modulation (~4 Hz syllable rate)
    env = 0.65 + 0.35 * np.sin(2.0 * math.pi * 4.1 * t)
    # Smooth fade-in/out
    fade = min(int(0.04 * rate), len(t) // 4)
    if fade > 0:
        env[:fade] *= np.linspace(0.0, 1.0, fade)
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
    sig = sig * env
    sig = sig / max(float(np.max(np.abs(sig))), 1e-9) * peak_amp
    return sig.astype(np.float32)


def run_self_test(vad_model: Path | None) -> int:
    """Builds a synthetic 48 kHz float WAV session in a temp dir and verifies all filters."""
    tmp_root = Path(tempfile.mkdtemp(prefix="dvo_mine_game_test_"))
    try:
        sr = 48000
        rng = np.random.default_rng(20261111)
        total_s = 28.0
        # Quiet natural room floor around -56 dBFS
        mic = rng.normal(0.0, 0.0012, int(total_s * sr)).astype(np.float32)

        def place(start_s: float, dur_s: float, peak_amp: float) -> None:
            seg = synth_speech_like_burst(rng, sr, dur_s, peak_amp)
            i0 = int(start_s * sr)
            mic[i0:i0 + len(seg)] += seg

        # 1) [1.5s .. 2.2s] (0.7s): short exclamation -> MUST be rejected as too_short (< 1.5s)
        place(1.5, 0.7, 0.25)
        # 2) [4.5s .. 8.7s] (4.2s, peak ~ -8 dBFS): clean steady utterance -> MUST be kept & prompt_candidate=True
        place(4.5, 4.2, 0.38)
        # 3) [11.5s .. 15.0s] (3.5s, peak ~ -0.25 dBFS): shouted clipping -> MUST be rejected as clipping (> -1.5 dBFS)
        place(11.5, 3.5, 0.97)
        # 4) [18.5s .. 22.5s] (4.0s, peak ~ -8 dBFS): inside discarded span -> MUST be rejected as discarded
        place(18.5, 4.0, 0.38)

        session_dir = tmp_root / "sessions" / "20261111-220000-gametest"
        write_float32_wav(session_dir / "mic.wav", mic, sr)
        # Poison loopback.wav with full-scale noise everywhere to prove loopback.wav is NEVER touched
        poison_loop = np.ones(int(total_s * sr), dtype=np.float32) * 0.99
        write_float32_wav(session_dir / "loopback.wav", poison_loop, sr)

        labels = {
            "collection_id": "20261111-220000",
            "purpose": "evaluate",
            "group": {"id": "background", "condition": "background", "distance_m": 0},
            "takes": [
                {"id": "background-01", "kind": "background", "status": "ok", "start_ms": 0, "end_ms": 17500},
                {"id": "background-02", "kind": "background", "status": "discarded", "start_ms": 17500, "end_ms": 24000},
                {"id": "background-03", "kind": "background", "status": "ok", "start_ms": 24000, "end_ms": 28000},
            ],
        }
        (session_dir / "labels.json").write_text(json.dumps(labels, indent=2), "utf-8")

        # Test Mode A: Energy VAD
        out_energy = tmp_root / "out_energy"
        sum_energy = mine_sessions(tmp_root / "sessions", out_energy, vad_model=vad_model, force_energy_vad=True)
        st_e = sum_energy["stats"]
        assert st_e["kept_clips"] == 1, f"Energy VAD expected 1 kept clip, got {st_e}"
        assert st_e["rejected_too_short"] >= 1, f"Expected short rejection, got {st_e}"
        assert st_e["rejected_clipping"] == 1, f"Expected clipping rejection, got {st_e}"
        assert st_e["rejected_discarded_span"] >= 1, f"Expected discarded rejection, got {st_e}"
        assert st_e["prompt_candidates"] == 1, f"Expected 1 prompt candidate, got {st_e}"

        rec = json.loads((out_energy / "manifest.jsonl").read_text("utf-8").strip())
        with wave.open(str(out_energy / rec["audio"]), "rb") as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2

        # Test Mode B: Silero ONNX VAD (if available)
        silero_checked = False
        if vad_model is not None and vad_model.is_file():
            try:
                import onnxruntime  # noqa: F401
                out_silero = tmp_root / "out_silero"
                sum_silero = mine_sessions(tmp_root / "sessions", out_silero, vad_model=vad_model, force_energy_vad=False)
                st_s = sum_silero["stats"]
                assert st_s["vad_backend_silero"] == 1
                assert st_s["kept_clips"] == 1, f"Silero VAD expected 1 kept clip, got {st_s}"
                assert st_s["rejected_clipping"] == 1, f"Silero expected clipping rejection, got {st_s}"
                assert st_s["rejected_discarded_span"] >= 1, f"Silero expected discarded rejection, got {st_s}"
                assert st_s["prompt_candidates"] == 1, f"Silero expected 1 prompt candidate, got {st_s}"
                silero_checked = True
            except ImportError:
                pass

        # Also unit-test dual-model text normalization & 50/50 split determinism
        assert normalize_zh("<|zh|><|NEUTRAL|>小克，換一首歌！") == "小克换一首歌"
        splits_100 = [disagree_split("sess1", f"sess1/game-{i:04d}", 1) for i in range(100)]
        train_ratio = splits_100.count("train") / 100.0
        assert 0.35 <= train_ratio <= 0.65, f"Unbalanced single-session split ratio: {train_ratio}"

        print("SELF-TEST PASSED!")
        print("  Energy VAD summary:", json.dumps(sum_energy["stats"], ensure_ascii=False))
        if silero_checked:
            print("  Silero VAD summary:", json.dumps(sum_silero["stats"], ensure_ascii=False))
        print(f"  Exported valid clip: start_s={rec['start_s']}s, dur={rec['duration_s']}s, "
              f"peak={rec['peak_dbfs']}dBFS, snr={rec['snr_db']}dB, prompt_candidate={rec['prompt_candidate']}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sessions", type=Path, default=Path("data/sessions"),
                        help="session folder or parent directory of recorded sessions")
    parser.add_argument("--out", type=Path, default=Path("data/game_mined"),
                        help="output directory for mined 16 kHz clips and manifest.jsonl")
    parser.add_argument("--vad-model", type=Path, default=repo_root / "models" / "silero_vad.int8.onnx")
    parser.add_argument("--energy-vad-only", action="store_true",
                        help="force adaptive dual-threshold energy VAD even if onnxruntime is installed")
    parser.add_argument("--include-all-sessions", action="store_true",
                        help="also mine non-background labeled sessions (off by default to protect eval sets)")
    parser.add_argument("--transcribe", action="store_true",
                        help="run SenseVoice + faster-whisper large-v3 dual-model cross-verification")
    parser.add_argument("--sensevoice-dir", type=Path,
                        help="directory containing SenseVoice model(.int8).onnx and tokens.txt")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model name or path (default: large-v3)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--self-test", action="store_true",
                        help="run synthetic 48 kHz float WAV + discarded span unit test and exit")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.vad_model)

    summary = mine_sessions(
        sessions_root=args.sessions,
        out_dir=args.out,
        vad_model=args.vad_model,
        force_energy_vad=args.energy_vad_only,
        include_labeled_non_bg=args.include_all_sessions,
        transcribe=args.transcribe,
        sensevoice_dir=args.sensevoice_dir,
        whisper_model=args.whisper_model,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
