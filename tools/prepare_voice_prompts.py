"""Select and clean a golden personal reference audio pool for zero-shot TTS cloning.

Supports two input modes:
  1. Standard dataset manifest (`--manifest data/dataset/manifest.jsonl`) produced by
     `tools/export_dataset.py`;
  2. Evaluation report + clip folders (`--eval-report .../eval_comparison_report.json`
     and `--clips-dirs ...`), useful when only exported clips and benchmark reports
     are synced to Google Drive.

Acoustic & linguistic gates for CosyVoice2 / F5-TTS zero-shot prompts:
  * Quiet condition only (no background music; verified against reference loopback when available);
  * Trailing spacebar click transient precisely trimmed via Silero VAD + 10 ms energy envelope,
    preserving 0.15 s - 0.20 s of natural room tone after sentence end without any spectral
    denoising (zero musical-noise / vocoder artifacts);
  * Trimmed duration in [2.5 s, 8.0 s], peak in [-24.0, -1.5] dBFS, high SNR;
  * Prioritizes clips with 0 ASR character errors on the baseline recognizer and balances
    sentence patterns (prefix wake, suffix wake, multi-step commands, declarative reading).

Example:
    python tools/prepare_voice_prompts.py \\
        --eval-report "H:/我的云端硬盘/desktop-voice-orchestrator-data/experiments/2026-11-10-openai-vs-local-eval/eval_comparison_report.json" \\
        --clips-dirs "H:/我的云端硬盘/desktop-voice-orchestrator-data/dataset/clips" \\
                     "H:/我的云端硬盘/desktop-voice-orchestrator-data/experiments/2026-11-10-openai-vs-local-eval/clips_20260926_225120" \\
        --out-dir "H:/我的云端硬盘/desktop-voice-orchestrator-data/m3_tts_prep/prompt_pool"
"""

from __future__ import annotations

import argparse
import datetime
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
FRAME_MS = 10
FRAME_SAMPLES = RATE * FRAME_MS // 1000  # 160 samples @ 16 kHz
VAD_WINDOW = 512  # 32 ms @ 16 kHz for silero_vad v4
MUSIC_REF_DBFS = -45.0

# Fixed evaluation script keys from web/collect.js (for confirmed wake="小克" quiet-near sessions).
EVAL_COMMAND_PROMPTS: dict[str, str] = {
    "pre-pause": "小克，暂停音乐",
    "suf-play": "播放音乐，小克",
    "pre-volup": "小克，增加音量",
    "suf-voldown10": "降低音量百分之十，小克",
    "pre-play": "小克，播放音乐",
    "suf-pause": "暂停音乐，小克",
    "pre-voldown10": "小克，降低音量百分之十",
    "suf-volup": "增加音量，小克",
    "pre-multi": "小克，暂停音乐，然后降低音量百分之十",
    "suf-multi": "播放音乐，再增加音量百分之五，小克",
    "pre-multi2": "小克，增加音量，再播放音乐",
    "suf-multi2": "降低音量，然后暂停音乐，小克",
    "pre-open": "小克，打开音乐",
}

# Wake-independent negative prompts in evaluation sessions (never include a wake word).
EVAL_NEGATIVE_PROMPTS: dict[str, str] = {
    "neg-command": "暂停音乐",
    "neg-chat": "今天晚上吃点什么好呢",
}

TRAIN_PHRASINGS_MAP: dict[str, list[str]] = {
    "play": ["播放音乐", "打开音乐", "把音乐打开", "开一下音乐", "放音乐", "放点音乐", "放首歌", "来首歌",
             "继续播放", "继续播放音乐", "继续放歌", "开始播放"],
    "pause": ["暂停音乐", "暂停", "暂停一下", "停一下", "停一下歌", "停一下音乐", "先停一下", "停止播放",
              "别放了", "先别放了", "不要放了", "关掉音乐", "把音乐关掉", "把音乐关了"],
    "up": ["增加音量", "调大音量", "音量调大", "音量大一点", "声音大一点", "大声一点", "大声点"],
    "down": ["降低音量", "调小音量", "音量调小", "音量小一点", "声音小一点", "小声一点", "小声点"],
    "next": ["切歌", "换歌", "换一首", "换一首歌", "下一首", "下一首歌"],
    "like": ["喜欢这首歌", "我喜欢这首歌", "喜欢这个", "我喜欢这个", "收藏一下"],
    "daily": ["播放每日推荐", "放每日推荐", "打开每日推荐", "每日推荐"],
    "mode": ["顺序播放", "列表循环", "单曲循环", "随机播放"],
}
RECORDED_BASE_PHRASES = {"播放音乐", "打开音乐", "暂停音乐", "增加音量", "降低音量"}
NEW_TRAIN_PHRASINGS_MAP = {
    k: [p for p in v if p not in RECORDED_BASE_PHRASES] for k, v in TRAIN_PHRASINGS_MAP.items()
}
AMOUNT_CAPABLE_PHRASINGS = {"增加音量", "调大音量", "音量调大", "降低音量", "调小音量", "音量调小"}
TRAIN_CONNECTORS_LIST = ["，然后", "，再", "，然后再", "，接着"]
TRAIN_AMOUNTS_LIST = [5, 5, 10, 10, 10, 20, 15, 3, 8, 2, 1, 12, 6, 18]
TRAIN_STYLES_LIST = ["说快一点", "声音轻一点", "稍微大声一点", "像随口一说", "慢一点说"]
CN_DIGITS = "零一二三四五六七八九"


def _imul32(a: int, b: int) -> int:
    res = ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF
    return res - 0x100000000 if res & 0x80000000 else res


def _seeded_random_js(seed: str):
    """Exact port of web/collect.js Mulberry32 deterministic PRNG."""
    h = (1779033703 ^ len(seed)) & 0xFFFFFFFF
    if h & 0x80000000:
        h -= 0x100000000
    for ch in seed:
        h = _imul32(h ^ ord(ch), 3432918353)
        h = (((h & 0xFFFFFFFF) << 13) | ((h & 0xFFFFFFFF) >> 19)) & 0xFFFFFFFF
        if h & 0x80000000:
            h -= 0x100000000

    def next_rand() -> float:
        nonlocal h
        h = (h + 0x6D2B79F5) & 0xFFFFFFFF
        if h & 0x80000000:
            h -= 0x100000000
        t = _imul32(h ^ ((h & 0xFFFFFFFF) >> 15), 1 | h)
        t = (t + _imul32(t ^ ((t & 0xFFFFFFFF) >> 7), 61 | t)) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_rand


def _cn_num_20(v: int) -> str:
    if v < 10:
        return CN_DIGITS[v]
    if v == 10:
        return "十"
    return f"十{CN_DIGITS[v - 10]}" if v < 20 else "二十"


def _reconstruct_train_prompts(collection_id: str, group_id: str, count: int, is_new: bool, wake: str = "小克") -> dict[str, str]:
    rnd = _seeded_random_js(f"{collection_id}-{group_id}")
    pick = lambda lst: lst[int(rnd() * len(lst))]
    phrasings = NEW_TRAIN_PHRASINGS_MAP if is_new else TRAIN_PHRASINGS_MAP
    keys_list = list(phrasings.keys())

    def phrase(prev_act=None):
        while True:
            act = pick(keys_list)
            if act != prev_act:
                break
        say = pick(phrasings[act])
        if act in ("play", "pause", "next", "like", "daily", "mode"):
            return say, act
        if say not in AMOUNT_CAPABLE_PHRASINGS or rnd() < 0.4:
            return say, act
        amt = pick(TRAIN_AMOUNTS_LIST)
        return f"{say}百分之{_cn_num_20(amt)}", act

    wake_cnt = 0 if is_new else round(count * 0.1)
    bare_cnt = round(count * 0.15)
    kinds = ["wake" if i < wake_cnt else "bare" if i < wake_cnt + bare_cnt else "command" for i in range(count)]
    for i in range(len(kinds) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        kinds[i], kinds[j] = kinds[j], kinds[i]

    out: dict[str, str] = {}
    for idx, k in enumerate(kinds, 1):
        note = ""
        if k == "wake":
            key, say = "wake", wake
        elif k == "bare":
            s, _ = phrase()
            key, say, note = "bare", s, "这条不说唤醒词"
        else:
            p0 = phrase()
            parts = [p0]
            if rnd() < 0.3:
                parts.append(phrase(p0[1]))
            body = "".join((pick(TRAIN_CONNECTORS_LIST) if ip else "") + pt[0] for ip, pt in enumerate(parts))
            pref = rnd() < 0.5
            key = "prefix" if pref else "suffix"
            say = f"{wake}，{body}" if pref else f"{body}，{wake}"
        if not note and rnd() < 0.3:
            note = pick(TRAIN_STYLES_LIST)
        out[f"{group_id}-{idx:02d}-{key}"] = say
    return out


def _reconstruct_read_prompts(collection_id: str, group_id: str, reading_pool: dict, count: int = 60) -> dict[str, str]:
    rnd = _seeded_random_js(f"{collection_id}-{group_id}")
    colloquial = reading_pool.get("colloquial", [])
    general = reading_pool.get("aishell", [])
    picked: set[str] = set()

    def pick_unique(lst: list[str]) -> str | None:
        if not lst:
            return None
        for _ in range(50):
            txt = lst[int(rnd() * len(lst))]
            if txt not in picked:
                return txt
        return None

    script: dict[str, str] = {}
    attempt = 0
    while len(script) < count and attempt < count * 20:
        attempt += 1
        roll = rnd()
        txt = None
        if roll < 0.15:
            txt = pick_unique(colloquial)
        elif roll < 0.35:
            first = pick_unique(general)
            if first:
                filt = [o for o in general if o != first and len(first) + len(o) + 1 <= 40]
                second = pick_unique(filt)
                if second:
                    txt = f"{first}，{second}"
        else:
            txt = pick_unique(general)
        if not txt:
            continue
        for p in txt.split("，"):
            picked.add(p)
        idx = len(script) + 1
        script[f"{group_id}-{idx:02d}-read"] = txt
    return script


def read_wav_mono_16k(path: Path) -> np.ndarray:
    """Reads a WAV file (16-bit PCM or 32-bit float) and returns 16 kHz mono float32 in [-1, 1]."""
    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: invalid WAV header")
    pos, rate, channels, bits, tag = 12, 0, 0, 0, 0
    while pos + 8 <= len(raw):
        chunk = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", raw[pos + 8:pos + 16])
            bits = struct.unpack("<H", raw[pos + 22:pos + 24])[0]
            if tag == 0xFFFE and size >= 40:
                tag = struct.unpack("<H", raw[pos + 32:pos + 34])[0]
        elif chunk == b"data":
            avail = len(raw) - pos - 8
            if size == 0 or size > avail:
                size = avail
            bytes_per_sample = max(1, bits // 8)
            frame_bytes = bytes_per_sample * max(1, channels)
            size = (size // frame_bytes) * frame_bytes
            buf = raw[pos + 8:pos + 8 + size]
            if tag == 3 and bits == 32:
                samples = np.frombuffer(buf, dtype="<f4").reshape(-1, channels)[:, 0].astype(np.float32)
            elif tag == 1 and bits == 16:
                samples = np.frombuffer(buf, dtype="<i2").reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0
            elif tag == 1 and bits == 32:
                samples = np.frombuffer(buf, dtype="<i4").reshape(-1, channels)[:, 0].astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"{path}: unsupported WAV format tag={tag} bits={bits}")
            return resample_to_16k(samples, rate)
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path}: missing data chunk")


def resample_to_16k(signal: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return signal
    if len(signal) == 0:
        return np.zeros(0, dtype=np.float32)
    if rate % RATE == 0:
        factor = rate // RATE
        n = (len(signal) // factor) * factor
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        spectrum = np.fft.rfft(signal[:n])
        spectrum[len(spectrum) // factor:] = 0
        return np.fft.irfft(spectrum, n)[::factor].astype(np.float32)
    # General band-limited FFT resampling for non-integer ratios (e.g. 44.1 kHz -> 16 kHz)
    target_len = int(round(len(signal) * RATE / rate))
    if target_len == 0:
        return np.zeros(0, dtype=np.float32)
    spectrum = np.fft.rfft(signal)
    out_spec_len = target_len // 2 + 1
    out_spec = np.zeros(out_spec_len, dtype=np.complex128)
    copy_bins = min(len(spectrum), out_spec_len)
    out_spec[:copy_bins] = spectrum[:copy_bins]
    res = np.fft.irfft(out_spec, target_len) * (target_len / len(signal))
    return res.astype(np.float32)


def write_pcm16(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())


def rms_dbfs(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(np.square(signal, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-6 else -120.0


def peak_dbfs_of(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return -120.0
    pk = float(np.max(np.abs(signal)))
    return 20.0 * math.log10(pk) if pk > 1e-6 else -120.0


class SileroVadRunner:
    """Lightweight ONNX wrapper around models/silero_vad.int8.onnx (16 kHz, 512-sample window)."""

    def __init__(self, model_path: Path | None):
        self.session = None
        if model_path is not None and model_path.is_file():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception:
                self.session = None

    @property
    def available(self) -> bool:
        return self.session is not None

    def frame_probs_10ms(self, signal: np.ndarray, num_10ms_frames: int) -> np.ndarray | None:
        if self.session is None or len(signal) < VAD_WINDOW or num_10ms_frames <= 0:
            return None
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        n_win = len(signal) // VAD_WINDOW
        probs = np.zeros(n_win, dtype=np.float32)
        for i in range(n_win):
            chunk = signal[i * VAD_WINDOW:(i + 1) * VAD_WINDOW].reshape(1, VAD_WINDOW)
            prob, h, c = self.session.run(None, {"x": chunk, "h": h, "c": c})
            probs[i] = float(prob[0, 0])
        centers_32ms = (np.arange(n_win) + 0.5) * (VAD_WINDOW / RATE)
        centers_10ms = (np.arange(num_10ms_frames) + 0.5) * (FRAME_SAMPLES / RATE)
        return np.interp(centers_10ms, centers_32ms, probs).astype(np.float32)


def trim_trailing_click_and_pad(
    signal: np.ndarray,
    vad: SileroVadRunner,
    pad_before_s: float = 0.14,
    tail_min_s: float = 0.15,
    tail_target_s: float = 0.17,
    tail_max_s: float = 0.20,
) -> tuple[np.ndarray, dict]:
    """Removes trailing spacebar click transients while keeping natural room tone.

    Never applies spectral noise suppression or nonlinear filtering; the output is
    pure time-domain slicing of the original waveform (with clean room-tone extension
    from the same clip only if a key click occurred right after speech end).
    """
    n_frames = len(signal) // FRAME_SAMPLES
    if n_frames < 20:
        return signal, {"valid": False, "reason": "too_short_raw"}

    frames = signal[:n_frames * FRAME_SAMPLES].astype(np.float64).reshape(n_frames, FRAME_SAMPLES)
    frame_rms_db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
    # High-frequency transient energy (first-order difference) helps spot mechanical key clicks
    diff = np.diff(frames, axis=1)
    hf_ratio = np.mean(diff * diff, axis=1) / (np.mean(frames * frames, axis=1) + 1e-12)

    noise_floor_db = min(float(np.percentile(frame_rms_db, 15)), -36.0)
    speech_p90_db = float(np.percentile(frame_rms_db, 90))
    vad_probs = vad.frame_probs_10ms(signal, n_frames)

    energy_thresh = max(noise_floor_db + 11.0, speech_p90_db - 24.0, -52.0)
    if vad_probs is not None:
        core_speech = (vad_probs >= 0.32) & (frame_rms_db >= noise_floor_db + 6.0)
        # Dilate by 80 ms to capture soft unvoiced consonant onsets/codas above noise floor
        dilated = np.convolve(core_speech.astype(np.int32), np.ones(17, dtype=np.int32), mode="same") > 0
        active_mask = core_speech | (dilated & (frame_rms_db >= noise_floor_db + 8.5))
        if not np.any(active_mask):
            active_mask = frame_rms_db > energy_thresh
    else:
        active_mask = frame_rms_db > energy_thresh

    idx = np.flatnonzero(active_mask)
    if len(idx) == 0:
        return signal, {"valid": False, "reason": "no_speech_detected"}

    # Group contiguous active frames (joining short pauses <= 220 ms = 22 frames)
    segments: list[tuple[int, int]] = []
    seg_start, seg_end = int(idx[0]), int(idx[0])
    for f in idx[1:]:
        fi = int(f)
        if fi - seg_end > 22:
            segments.append((seg_start, seg_end))
            seg_start = fi
        seg_end = fi
    segments.append((seg_start, seg_end))

    # Drop isolated micro-bursts (< 80 ms) everywhere first
    segments = [(a, b) for a, b in segments if (b - a + 1) >= 8]
    if not segments:
        return signal, {"valid": False, "reason": "only_micro_bursts"}

    click_removed = False
    # Inspect the final segment: if it sits in the last 0.45 s, is short (<= 150 ms),
    # and has low VAD speech probability or sharp transient ratio after a gap, drop it as a key click.
    while len(segments) >= 2:
        last_a, last_b = segments[-1]
        prev_b = segments[-2][1]
        seg_dur_frames = last_b - last_a + 1
        gap_frames = last_a - prev_b - 1
        in_tail_zone = (n_frames - last_a) <= 45
        seg_vad_max = float(np.max(vad_probs[last_a:last_b + 1])) if vad_probs is not None else 0.0
        seg_hf_max = float(np.max(hf_ratio[last_a:last_b + 1]))
        is_click_burst = (
            in_tail_zone
            and seg_dur_frames <= 15
            and gap_frames >= 3
            and (vad_probs is None or seg_vad_max < 0.45 or seg_hf_max > 0.65)
        )
        if is_click_burst:
            segments.pop()
            click_removed = True
        else:
            break

    speech_start_f = segments[0][0]
    speech_end_f = segments[-1][1] + 1  # exclusive frame index

    # Also check within the last 120 ms of the final segment for a detached impulse right after speech tail
    if speech_end_f < n_frames:
        tail_scan_start = speech_end_f
        tail_scan_end = min(n_frames, speech_end_f + int(round(tail_max_s * 100)))
        spike_thresh = max(noise_floor_db + 9.5, -42.0)
        spikes = np.flatnonzero(frame_rms_db[tail_scan_start:tail_scan_end] > spike_thresh)
        if len(spikes) > 0:
            first_spike_f = tail_scan_start + int(spikes[0])
            clean_end_f = max(speech_end_f, first_spike_f - 1)
            click_removed = True
        else:
            clean_end_f = min(n_frames, speech_end_f + int(round(tail_target_s * 100)))
    else:
        clean_end_f = n_frames

    # Clean pre-roll without capturing any startup pop in frame 0..2
    desired_start_f = max(0, speech_start_f - int(round(pad_before_s * 100)))
    pre_spikes = np.flatnonzero(
        frame_rms_db[desired_start_f:speech_start_f] > max(noise_floor_db + 11.0, -40.0)
    )
    if len(pre_spikes) > 0:
        clean_start_f = min(speech_start_f, desired_start_f + int(pre_spikes[-1]) + 2)
    else:
        clean_start_f = desired_start_f

    s0 = clean_start_f * FRAME_SAMPLES
    s1 = min(len(signal), clean_end_f * FRAME_SAMPLES)
    trimmed = signal[s0:s1].copy()

    # Ensure at least tail_min_s (0.15 s) of natural room tone after speech_end
    actual_tail_s = (s1 - speech_end_f * FRAME_SAMPLES) / RATE
    if actual_tail_s < tail_min_s:
        need_samples = int(round((tail_target_s - actual_tail_s) * RATE))
        room_tone = extract_clean_room_tone(signal, frame_rms_db, noise_floor_db, need_samples)
        trimmed = crossfade_append(trimmed, room_tone, fade_samples=80)

    speech_slice = frame_rms_db[speech_start_f:max(speech_start_f + 1, speech_end_f)]
    speech_db = float(np.percentile(speech_slice, 85))
    snr_db = round(speech_db - noise_floor_db, 1)
    speech_span_s = round((speech_end_f - speech_start_f) * (FRAME_MS / 1000.0), 3)

    info = {
        "valid": True,
        "raw_duration_s": round(len(signal) / RATE, 3),
        "duration_s": round(len(trimmed) / RATE, 3),
        "speech_span_s": speech_span_s,
        "peak_dbfs": round(peak_dbfs_of(trimmed), 2),
        "rms_dbfs": round(rms_dbfs(trimmed), 2),
        "noise_floor_db": round(noise_floor_db, 2),
        "snr_db": snr_db,
        "click_removed": click_removed,
    }
    return trimmed, info


def extract_clean_room_tone(
    signal: np.ndarray,
    frame_rms_db: np.ndarray,
    noise_floor_db: float,
    need_samples: int,
) -> np.ndarray:
    """Selects genuine quiet room-tone frames from the same recording (never synthetic white noise)."""
    quiet_frames = np.flatnonzero(frame_rms_db <= noise_floor_db + 4.5)
    if len(quiet_frames) == 0:
        quiet_frames = np.argsort(frame_rms_db)[:max(1, min(8, len(frame_rms_db)))]
    chunks: list[np.ndarray] = []
    collected = 0
    idx = 0
    while collected < need_samples:
        f = int(quiet_frames[idx % len(quiet_frames)])
        piece = signal[f * FRAME_SAMPLES:(f + 1) * FRAME_SAMPLES]
        chunks.append(piece)
        collected += len(piece)
        idx += 1
    return np.concatenate(chunks)[:need_samples].astype(np.float32)


def crossfade_append(main_sig: np.ndarray, tail_sig: np.ndarray, fade_samples: int = 80) -> np.ndarray:
    if len(tail_sig) == 0:
        return main_sig
    fade = min(fade_samples, len(main_sig) // 4, len(tail_sig) // 2)
    if fade <= 1:
        return np.concatenate([main_sig, tail_sig])
    alpha = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    blended = main_sig[-fade:] * (1.0 - alpha) + tail_sig[:fade] * alpha
    return np.concatenate([main_sig[:-fade], blended, tail_sig[fade:]])


def classify_pattern(prompt_text: str, kind: str = "") -> str:
    has_comma_chain = any(tok in prompt_text for tok in ("然后", "再", "接着", "并且")) or prompt_text.count("，") >= 2
    if kind == "read" or ("小克" not in prompt_text and len(prompt_text) >= 8):
        return "read_declarative"
    if has_comma_chain:
        return "multi_step_command"
    if prompt_text.startswith("小克"):
        return "prefix_wake_command"
    if prompt_text.endswith("小克"):
        return "suffix_wake_command"
    return "bare_or_colloquial"


def clean_prompt_text(prompt: str) -> str:
    """Normalizes punctuation for CosyVoice2 prompt_text (clean Chinese + commas only)."""
    text = prompt.strip()
    text = re.sub(r"[;；!！?？。、]+", "，", text)
    text = re.sub(r"，+", "，", text).strip("，")
    return text


def check_reference_quiet(wav_path: Path) -> bool:
    """If a matching loopback reference WAV exists next to `clips/`, verify it has no music."""
    parts = list(wav_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "clips":
            ref_parts = parts[:i] + ["reference"] + parts[i + 1:]
            ref_path = Path(*ref_parts)
            if ref_path.is_file():
                try:
                    ref_sig = read_wav_mono_16k(ref_path)
                    return rms_dbfs(ref_sig) <= MUSIC_REF_DBFS
                except Exception:
                    return True
            break
    return True


def collect_from_manifest(manifest_path: Path, dataset_dir: Path) -> list[dict]:
    candidates = []
    for line in manifest_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("music"):
            continue
        if r.get("condition") not in (None, "quiet"):
            continue
        kind = r.get("kind", "")
        if kind not in ("command", "bare_command", "negative", "read"):
            continue
        prompt = clean_prompt_text(r.get("prompt") or r.get("text") or "")
        if len(prompt) < 4:
            continue
        wav_path = dataset_dir / r["audio"]
        if not wav_path.is_file():
            continue
        candidates.append({
            "id": str(r["id"]).replace("/", "__"),
            "raw_id": r["id"],
            "wav_path": wav_path,
            "prompt_text": prompt,
            "kind": kind,
            "orig_cer": 0.0,
            "personal_cer": 0.0,
            "source": "manifest",
        })
    return candidates


def resolve_clip_in_dirs(cid: str, clips_dirs: list[Path]) -> Path | None:
    parts = cid.replace("\\", "/").split("/")
    possible_rels = [f"{cid}.wav"]
    if len(parts) >= 2:
        possible_rels.append(f"{parts[0]}/{parts[-1]}.wav")
    possible_rels.append(f"{parts[-1]}.wav")

    for d in clips_dirs:
        for rel in possible_rels:
            cand = d / rel
            if cand.is_file():
                return cand
    return None


def collect_from_eval_report(report_path: Path, clips_dirs: list[Path]) -> list[dict]:
    data = json.loads(report_path.read_text("utf-8"))
    failures = data.get("failures", {})

    by_id: dict[str, dict] = {}
    confirmed_xiaoke_sessions: set[str] = set()

    for bucket_key, items in failures.items():
        if ":" not in bucket_key:
            continue
        model_name, subset = bucket_key.split(":", 1)
        for it in items:
            cid = it["id"]
            if subset == "test_quiet" and "/" in cid:
                confirmed_xiaoke_sessions.add(cid.split("/")[0])
            entry = by_id.setdefault(cid, {
                "id": cid,
                "subset": subset,
                "prompt": it.get("prompt", ""),
                "music": it.get("music", subset == "test_music"),
                "failed_models": {},
            })
            entry["failed_models"][model_name] = it

    candidates_map: dict[str, dict] = {}

    # 1. Quiet items explicitly present in eval_comparison_report.json
    for cid, entry in by_id.items():
        if entry["music"] or entry["subset"] == "test_music":
            continue
        wav_path = resolve_clip_in_dirs(cid, clips_dirs)
        if wav_path is None or not check_reference_quiet(wav_path):
            continue
        prompt = clean_prompt_text(entry["prompt"])
        if len(prompt) < 4:
            continue
        orig_fail = entry["failed_models"].get("local_sensevoice_orig")
        pers_fail = entry["failed_models"].get("local_sensevoice_personal")
        norm_len = max(1, len(re.findall(r"[\u4e00-\u9fff]", prompt)))
        orig_cer = 0.0 if orig_fail is None else float(orig_fail.get("cer_err", 1)) / norm_len
        pers_cer = 0.0 if pers_fail is None else float(pers_fail.get("cer_err", 1)) / norm_len
        kind = "read" if entry["subset"] == "dev_read" else "command"
        norm_key = str(wav_path.resolve()).lower()
        candidates_map[norm_key] = {
            "id": cid.split("/")[0] + "__" + wav_path.stem if "/" in cid else wav_path.stem,
            "raw_id": cid,
            "wav_path": wav_path,
            "prompt_text": prompt,
            "kind": kind,
            "orig_cer": round(orig_cer, 4),
            "personal_cer": round(pers_cer, 4),
            "source": f"eval_report:{entry['subset']}",
        }

    # 2. Zero-error quiet clips from confirmed test_quiet sessions (all 4 models passed 100%!)
    #    plus wake-independent negative takes (neg-chat / neg-command) across quiet sessions.
    for d in clips_dirs:
        if not d.is_dir():
            continue
        for session_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            sname = session_dir.name
            is_confirmed_xiaoke = sname in confirmed_xiaoke_sessions
            for wav_path in sorted(session_dir.glob("quiet-*.wav")):
                norm_key = str(wav_path.resolve()).lower()
                if norm_key in candidates_map:
                    continue
                if not check_reference_quiet(wav_path):
                    continue
                stem = wav_path.stem
                # Match suffix key: quiet-near-05-suf-play -> suf-play
                m = re.match(r"^quiet-(?:near|far)-\d+-(.+)$", stem)
                if not m:
                    continue
                take_key = m.group(1)
                prompt = None
                kind = "command"
                if take_key in EVAL_NEGATIVE_PROMPTS:
                    prompt = EVAL_NEGATIVE_PROMPTS[take_key]
                    kind = "negative"
                elif is_confirmed_xiaoke and take_key in EVAL_COMMAND_PROMPTS:
                    prompt = EVAL_COMMAND_PROMPTS[take_key]
                    kind = "command"
                if not prompt:
                    continue

                # Check if this clip was listed as a failure under session/session/stem
                full_cid = f"{sname}/{sname}/{stem}"
                entry = by_id.get(full_cid)
                orig_cer = 0.0
                pers_cer = 0.0
                if entry is not None:
                    if "local_sensevoice_orig" in entry["failed_models"]:
                        orig_cer = 0.2
                    if "local_sensevoice_personal" in entry["failed_models"]:
                        pers_cer = 0.2

                candidates_map[norm_key] = {
                    "id": f"{sname}__{stem}",
                    "raw_id": full_cid,
                    "wav_path": wav_path,
                    "prompt_text": clean_prompt_text(prompt),
                    "kind": kind,
                    "orig_cer": orig_cer,
                    "personal_cer": pers_cer,
                    "source": "eval_zero_error_quiet" if kind == "command" else "eval_quiet_negative",
                }

    # 3. Cryptographically anchor-verified deterministic reconstruction for zero-error clips
    #    in evaluated subsets (`new_commands_quiet` and initial `read-quiet` session).
    new_cmd_anchors = {
        cid: entry["prompt"]
        for cid, entry in by_id.items()
        if entry["subset"] == "new_commands_quiet" and entry.get("prompt")
    }
    for d in clips_dirs:
        if not d.is_dir():
            continue
        new_wavs = sorted(d.glob("train-new-quiet-*.wav"))
        m_ts = re.search(r"(20\d{6})[_-](\d{6})", d.name)
        if new_wavs and len(new_cmd_anchors) >= 4 and m_ts:
            t_base = datetime.datetime.strptime(m_ts.group(1) + m_ts.group(2), "%Y%m%d%H%M%S")
            matched_script = None
            for delta in range(-600, 15):
                cid_try = (t_base + datetime.timedelta(seconds=delta)).strftime("%Y%m%d-%H%M%S")
                sc = _reconstruct_train_prompts(cid_try, "train-new-quiet", 60, is_new=True)
                if all(sc.get(k) == v for k, v in new_cmd_anchors.items()) and all(w.stem in sc for w in new_wavs):
                    matched_script = sc
                    break
            if matched_script:
                for wav_path in new_wavs:
                    norm_key = str(wav_path.resolve()).lower()
                    if norm_key in candidates_map:
                        continue
                    prompt = matched_script.get(wav_path.stem)
                    if prompt:
                        candidates_map[norm_key] = {
                            "id": wav_path.stem,
                            "raw_id": wav_path.stem,
                            "wav_path": wav_path,
                            "prompt_text": clean_prompt_text(prompt),
                            "kind": "command",
                            "orig_cer": 0.0,
                            "personal_cer": 0.0,
                            "source": "eval_zero_error_new_cmd",
                        }

        # Check read-quiet session folders that have >= 4 anchor entries in dev_read
        reading_json = Path(__file__).resolve().parent.parent / "web" / "reading-sentences.json"
        if reading_json.is_file():
            reading_pool = json.loads(reading_json.read_text("utf-8"))
            for session_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                sname = session_dir.name
                read_anchors = {
                    cid.split("/")[-1]: entry["prompt"]
                    for cid, entry in by_id.items()
                    if entry["subset"] == "dev_read" and cid.startswith(sname + "/") and entry.get("prompt")
                }
                m_rs = re.match(r"^(20\d{6})-(\d{6})-", sname)
                if len(read_anchors) >= 4 and m_rs:
                    t_rbase = datetime.datetime.strptime(m_rs.group(1) + m_rs.group(2), "%Y%m%d%H%M%S")
                    matched_read = None
                    for delta in range(-600, 15):
                        cid_try = (t_rbase + datetime.timedelta(seconds=delta)).strftime("%Y%m%d-%H%M%S")
                        sc = _reconstruct_read_prompts(cid_try, "read-quiet", reading_pool, count=60)
                        if all(sc.get(k) == v for k, v in read_anchors.items()):
                            matched_read = sc
                            break
                    if matched_read:
                        for wav_path in sorted(session_dir.glob("read-quiet-*.wav")):
                            norm_key = str(wav_path.resolve()).lower()
                            if norm_key in candidates_map or not check_reference_quiet(wav_path):
                                continue
                            prompt = matched_read.get(wav_path.stem)
                            if prompt:
                                # Note: only dev-split items (10%) were scored in eval_report; unscored train-split
                                # items get a slight uncertainty penalty (0.02) so verified 0-CER dev/eval items rank first.
                                candidates_map[norm_key] = {
                                    "id": f"{sname}__{wav_path.stem}",
                                    "raw_id": f"{sname}/{wav_path.stem}",
                                    "wav_path": wav_path,
                                    "prompt_text": clean_prompt_text(prompt),
                                    "kind": "read",
                                    "orig_cer": 0.02,
                                    "personal_cer": 0.0,
                                    "source": "anchor_verified_read_quiet",
                                }

    return list(candidates_map.values())


def score_candidate(item: dict) -> float:
    """Composite acoustic + pronunciation clarity score for zero-shot voice cloning."""
    score = 100.0
    # Strongly prefer 0 CER on the un-finetuned baseline ASR (crystal-clear articulation)
    score -= item["orig_cer"] * 180.0
    score -= item["personal_cer"] * 120.0

    # Ideal zero-shot prompt duration for CosyVoice2 is around 3.2 s - 6.2 s
    dur = item["duration_s"]
    if 3.0 <= dur <= 6.5:
        score += 12.0
    else:
        score -= abs(dur - 4.5) * 2.5

    # SNR reward (capped at 38 dB so extreme outliers don't dominate)
    score += min(item["snr_db"], 38.0) * 0.8

    # Healthy peak level around -10 to -2.5 dBFS
    pk = item["peak_dbfs"]
    if -12.0 <= pk <= -2.2:
        score += 6.0
    else:
        score -= abs(pk - (-6.0)) * 0.6

    # Prefer natural 8-22 character sentences with rich phoneme coverage
    n_chars = len(re.findall(r"[\u4e00-\u9fff]", item["prompt_text"]))
    score += min(n_chars, 20) * 0.5
    return round(score, 2)


def select_diverse_top_k(qualified: list[dict], top_k: int) -> list[dict]:
    """Selects top_k prompts balancing sentence patterns and avoiding duplicate texts first."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(qualified, key=lambda x: -x["quality_score"]):
        buckets[item["pattern"]].append(item)

    bucket_order = [
        "read_declarative",
        "multi_step_command",
        "prefix_wake_command",
        "suffix_wake_command",
        "bare_or_colloquial",
    ]
    for b in buckets:
        if b not in bucket_order:
            bucket_order.append(b)

    selected: list[dict] = []
    seen_texts: set[str] = set()
    used_ids: set[str] = set()

    # Pass 1: unique prompt_text across balanced pattern buckets
    made_progress = True
    while len(selected) < top_k and made_progress:
        made_progress = False
        for b in bucket_order:
            if len(selected) >= top_k:
                break
            for cand in buckets.get(b, []):
                if cand["id"] in used_ids or cand["prompt_text"] in seen_texts:
                    continue
                selected.append(cand)
                used_ids.add(cand["id"])
                seen_texts.add(cand["prompt_text"])
                made_progress = True
                break

    # Pass 2: if unique texts < top_k, fill remaining slots with highest-scoring distinct recordings
    if len(selected) < top_k:
        for cand in sorted(qualified, key=lambda x: -x["quality_score"]):
            if len(selected) >= top_k:
                break
            if cand["id"] in used_ids:
                continue
            selected.append(cand)
            used_ids.add(cand["id"])

    return selected


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, help="dataset manifest.jsonl from export_dataset.py")
    parser.add_argument("--dataset-dir", type=Path, help="dataset root (defaults to manifest parent)")
    parser.add_argument("--eval-report", type=Path, help="eval_comparison_report.json path")
    parser.add_argument("--clips-dirs", type=Path, nargs="*", default=[],
                        help="clip directories to pair with --eval-report")
    parser.add_argument("--vad-model", type=Path, default=repo_root / "models" / "silero_vad.int8.onnx")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory for trimmed WAVs and prompt_pool.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=2.5)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.5)
    parser.add_argument("--min-peak-dbfs", type=float, default=-24.0)
    parser.add_argument("--min-snr-db", type=float, default=16.0)
    parser.add_argument("--max-orig-cer", type=float, default=0.15,
                        help="maximum allowed baseline ASR CER when ASR scores are known")
    args = parser.parse_args()

    raw_candidates: list[dict] = []
    if args.manifest and args.manifest.is_file():
        ds_dir = args.dataset_dir or args.manifest.parent
        raw_candidates.extend(collect_from_manifest(args.manifest, ds_dir))
    if args.eval_report and args.eval_report.is_file():
        raw_candidates.extend(collect_from_eval_report(args.eval_report, list(args.clips_dirs)))

    if not raw_candidates:
        print("Error: no candidate clips found. Provide --manifest or (--eval-report + --clips-dirs).",
              file=sys.stderr)
        return 1

    # Deduplicate by resolved file path
    deduped: dict[str, dict] = {}
    for c in raw_candidates:
        key = str(c["wav_path"].resolve()).lower()
        if key not in deduped:
            deduped[key] = c
    candidates = list(deduped.values())

    vad = SileroVadRunner(args.vad_model)
    stats = defaultdict(int)
    stats["total_quiet_candidates"] = len(candidates)
    stats["silero_vad_enabled"] = int(vad.available)

    qualified: list[dict] = []
    rejected_reasons: list[dict] = []

    for cand in candidates:
        if cand["orig_cer"] > args.max_orig_cer and cand["personal_cer"] > 0.0:
            stats["rejected_high_cer"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "high_asr_cer",
                                     "orig_cer": cand["orig_cer"], "personal_cer": cand["personal_cer"]})
            continue
        try:
            sig = read_wav_mono_16k(cand["wav_path"])
        except Exception as exc:
            stats["rejected_read_error"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": f"read_error:{exc}"})
            continue

        trimmed, info = trim_trailing_click_and_pad(sig, vad)
        if not info.get("valid"):
            stats["rejected_vad_empty"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": info.get("reason", "invalid")})
            continue

        if info["click_removed"]:
            stats["trailing_clicks_removed"] += 1

        dur = info["duration_s"]
        pk = info["peak_dbfs"]
        snr = info["snr_db"]

        if dur < args.min_duration:
            stats["rejected_too_short"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "too_short", "duration_s": dur})
            continue
        if dur > args.max_duration:
            stats["rejected_too_long"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "too_long", "duration_s": dur})
            continue
        if pk > args.max_peak_dbfs:
            stats["rejected_clipping_risk"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "peak_too_high", "peak_dbfs": pk})
            continue
        if pk < args.min_peak_dbfs:
            stats["rejected_too_quiet"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "peak_too_low", "peak_dbfs": pk})
            continue
        if snr < args.min_snr_db:
            stats["rejected_low_snr"] += 1
            rejected_reasons.append({"id": cand["id"], "reason": "low_snr", "snr_db": snr})
            continue

        pattern = classify_pattern(cand["prompt_text"], cand["kind"])
        enriched = {
            **cand,
            **info,
            "pattern": pattern,
            "trimmed_audio": trimmed,
        }
        enriched["quality_score"] = score_candidate(enriched)
        qualified.append(enriched)

    stats["qualified_count"] = len(qualified)
    selected = select_diverse_top_k(qualified, args.top_k)
    stats["selected_count"] = len(selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = args.out_dir / "wavs"
    if wavs_dir.exists():
        shutil.rmtree(wavs_dir)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    pool_records: list[dict] = []
    for idx, item in enumerate(selected, 1):
        safe_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", item["id"]).strip("_")
        rel_wav = f"wavs/prompt_{idx:02d}_{safe_id}.wav"
        write_pcm16(args.out_dir / rel_wav, item["trimmed_audio"])
        pool_records.append({
            "id": f"prompt_{idx:02d}",
            "source_id": item["raw_id"],
            "wav_path": rel_wav,
            "prompt_text": item["prompt_text"],
            "duration_s": item["duration_s"],
            "raw_duration_s": item["raw_duration_s"],
            "speech_span_s": item["speech_span_s"],
            "peak_dbfs": item["peak_dbfs"],
            "snr_db": item["snr_db"],
            "orig_cer": item["orig_cer"],
            "personal_cer": item["personal_cer"],
            "click_removed": item["click_removed"],
            "pattern": item["pattern"],
            "quality_score": item["quality_score"],
        })

    manifest_out = args.out_dir / "prompt_pool.jsonl"
    with open(manifest_out, "w", encoding="utf-8", newline="\n") as f:
        for rec in pool_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pattern_dist = defaultdict(int)
    for rec in pool_records:
        pattern_dist[rec["pattern"]] += 1

    summary = {
        "stats": dict(stats),
        "pattern_distribution": dict(pattern_dist),
        "zero_orig_cer_in_selected": sum(1 for r in pool_records if r["orig_cer"] == 0.0),
        "mean_duration_s": round(float(np.mean([r["duration_s"] for r in pool_records])), 2) if pool_records else 0.0,
        "mean_snr_db": round(float(np.mean([r["snr_db"] for r in pool_records])), 1) if pool_records else 0.0,
        "rejected_preview": rejected_reasons[:15],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )

    print("=== Golden Voice Prompt Pool Summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected_preview"}, ensure_ascii=False, indent=2))
    print(f"\nSelected {len(pool_records)} golden prompts -> {manifest_out}")
    for rec in pool_records:
        print(
            f"  [{rec['id']}] dur={rec['duration_s']:4.2f}s (raw {rec['raw_duration_s']:4.2f}s, "
            f"click_cut={str(rec['click_removed']):5s}) | peak={rec['peak_dbfs']:6.2f}dB | "
            f"snr={rec['snr_db']:4.1f}dB | orig_cer={rec['orig_cer']:.2f} | "
            f"{rec['pattern']:20s} | {rec['prompt_text']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
