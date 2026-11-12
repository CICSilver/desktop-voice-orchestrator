"""Zero-shot personal voice cloning synthesizer with closed-loop ASR judge quality gate.

Synthesizes natural-language command & song-request utterances with Alibaba's open-source
`CosyVoice2-0.5B` (`inference_zero_shot`), guarded by an independent un-finetuned
`sherpa-onnx` SenseVoice recognizer so distorted, truncated, repeated, or mispronounced
samples NEVER enter the ASR fine-tuning dataset.

Key safeguards:
  1. Multi-speaker anti-overfitting mix (`--personal-ratio 0.8`): 80% cloned from the user's
     golden `prompt_pool.jsonl` + 20% from auxiliary clean speakers (AISHELL-1 / extra pool /
     preset speaker) so the fine-tuned ASR learns vocabulary rather than overfitting to a
     single vocoder fingerprint;
  2. Anti-aliased polyphase low-pass resampling (`scipy.signal.resample_poly`) to 16 kHz mono
     int16 PCM;
  3. Two-stage closed-loop quality gate:
       - Stage A (Acoustic & Speaking-Rate Gate): verifies absence of clipping, truncation, or
         dead air, and enforces speaking rate within `[1.8, 7.5]` Chinese characters/sec;
       - Stage B (Independent ASR Judge Gate): decodes the 16 kHz WAV with the original
         un-finetuned `sherpa-onnx` SenseVoice model. Command utterances require 0 character
         errors (`CER == 0.0`); song/artist proper-noun utterances allow homophones only when
         toneless pinyin matches 100% (`pinyin_cer == 0.0` and `char_cer <= 0.20`);
  4. Automatic prompt rotation on failure (up to `--max-retries 3`); persistent failures are
     quarantined in `rejected.jsonl` and excluded from `manifest.jsonl`.

Usage:
    # Pilot 25 samples for 1-minute human listening check:
    python training/asr/synthesize_personal_tts.py --mode pilot \\
        --prompts /path/to/m3_tts_prep/pilot_prompts_25.jsonl \\
        --prompt-pool /path/to/m3_tts_prep/prompt_pool/prompt_pool.jsonl \\
        --cosyvoice-model /path/to/CosyVoice2-0.5B \\
        --judge-model-dir /path/to/sherpa-onnx-sense-voice \\
        --out-dir /path/to/m3_tts_prep/pilot_audio

    # Run built-in gate & resampling self-test (no GPU required):
    python training/asr/synthesize_personal_tts.py --self-test
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np

TARGET_SR = 16000


# Optional Mandarin phonetic transliterations emitted by Chinese-biased ASR judges for English app names.
# Strict English matching is enforced by default; homophone fallback is only active if `--allow-en-app-homophones` is set.
EN_APP_HOMOPHONE_MAP: dict[str, tuple[str, ...]] = {
    "steam": ("斯蒂姆", "死盯", "丝蒂姆"),
    "clash": ("可拉什", "克拉什", "克拉斯"),
    "discord": ("迪斯科", "迪斯扣", "迪斯考德"),
    "powershell": ("泡我秀", "帕瓦谢尔"),
    "qq": ("扣扣",),
    "vscode": ("维埃斯扣德",),
}


def edit_distance_seq(ref: list[str] | str, hyp: list[str] | str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i]
        for j, h in enumerate(hyp, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = curr
    return prev[-1]


def normalize_asr_text(text: str) -> str:
    """Strips SenseVoice tags (`<|zh|>`, etc.), lowercases ASCII, and retains `[a-z0-9\\u4e00-\\u9fff]`."""
    cleaned = re.sub(r"<\|[^|]*\|>", "", text)
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", cleaned.lower()))


# Backward-compatible alias
normalize_cjk = normalize_asr_text


def count_speech_units(norm_text: str) -> int:
    """Counts spoken syllable-equivalent units so multi-letter English words (e.g. `powershell`)
    do not artificially inflate characters-per-second speaking-rate checks."""
    hz = len(re.findall(r"[\u4e00-\u9fff]", norm_text))
    en_runs = re.findall(r"[a-z0-9]+", norm_text.lower())
    en_units = sum(max(2, min(4, round(len(tok) * 0.4))) for tok in en_runs)
    return hz + en_units


def to_toneless_pinyin(text: str) -> list[str]:
    """Tokenizes CJK into toneless pinyin syllables while keeping lowercase ASCII runs intact as tokens."""
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[a-z0-9]+", chunk):
            tokens.append(chunk)
        else:
            try:
                from pypinyin import Style, lazy_pinyin
                tokens.extend(lazy_pinyin(chunk, style=Style.NORMAL, errors="ignore"))
            except ImportError:
                tokens.extend(list(chunk))
    return tokens


def resample_to_16k_poly(signal: np.ndarray, orig_sr: int) -> np.ndarray:
    """High-quality anti-aliased polyphase low-pass resampling to 16 kHz."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    if orig_sr == TARGET_SR or len(sig) == 0:
        return sig
    g = math.gcd(TARGET_SR, orig_sr)
    up = TARGET_SR // g
    down = orig_sr // g
    try:
        from scipy.signal import resample_poly
        return resample_poly(sig, up=up, down=down).astype(np.float32)
    except ImportError:
        # Band-limited FFT fallback if scipy is absent in minimal test environments
        target_len = int(round(len(sig) * TARGET_SR / orig_sr))
        if target_len == 0:
            return np.zeros(0, dtype=np.float32)
        spec = np.fft.rfft(sig)
        out_spec = np.zeros(target_len // 2 + 1, dtype=np.complex128)
        bins = min(len(spec), len(out_spec))
        out_spec[:bins] = spec[:bins]
        return (np.fft.irfft(out_spec, target_len) * (target_len / len(sig))).astype(np.float32)


def _load_wav_soundfile_tensor(wav_path: str | Path, target_sr: int | None = None):
    """Deterministic, idempotent audio loader using soundfile + polyphase resampling.

    Loads WAV as float32, averages channels to mono, resamples via polyphase filter
    if target_sr is requested and differs from file sr, and returns a (1, N) float32 tensor
    (torch.Tensor if torch is installed, else numpy array).
    If target_sr is None, returns (tensor_1xN, sr); otherwise returns tensor_1xN.
    """
    path_str = str(wav_path)
    try:
        import soundfile as sf
        data, sr = sf.read(path_str, dtype="float32", always_2d=True)
    except Exception:
        # Fallback to standard library wave for simple PCM WAV
        import wave
        with wave.open(path_str, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)
            if sampwidth == 2:
                data = (np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0).reshape(-1, n_ch)
            elif sampwidth == 4:
                data = (np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / 2147483648.0).reshape(-1, n_ch)
            elif sampwidth == 1:
                data = ((np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0).reshape(-1, n_ch)
            else:
                data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, n_ch)

    if data.shape[1] > 1:
        mono = np.mean(data, axis=1)
    else:
        mono = data[:, 0]

    out_sr = sr
    if target_sr is not None and target_sr != sr and len(mono) > 0:
        g = math.gcd(int(target_sr), int(sr))
        up = int(target_sr) // g
        down = int(sr) // g
        try:
            from scipy.signal import resample_poly
            mono = resample_poly(mono, up=up, down=down).astype(np.float32)
        except ImportError:
            target_len = int(round(len(mono) * target_sr / sr))
            if target_len == 0:
                mono = np.zeros(0, dtype=np.float32)
            else:
                spec = np.fft.rfft(mono)
                out_spec = np.zeros(target_len // 2 + 1, dtype=np.complex128)
                bins = min(len(spec), len(out_spec))
                out_spec[:bins] = spec[:bins]
                mono = (np.fft.irfft(out_spec, target_len) * (target_len / len(mono))).astype(np.float32)
        out_sr = target_sr

    mono_2d = mono.reshape(1, -1).astype(np.float32, copy=False)
    try:
        import torch
        res_tensor = torch.from_numpy(mono_2d)
    except ImportError:
        res_tensor = mono_2d

    if target_sr is None:
        return res_tensor, out_sr
    return res_tensor


def _install_torchaudio_soundfile_shim() -> None:
    """Installs an idempotent torchaudio.load shim bypassing torchcodec on file paths."""
    try:
        import torchaudio
    except ImportError:
        return

    if getattr(torchaudio.load, "_dvo_sf_patched", False):
        return

    orig_load = torchaudio.load

    def _sf_load(uri, frame_offset: int = 0, num_frames: int = -1, normalize: bool = True, channels_first: bool = True, format: str | None = None, buffer_size: int = 4096):
        import torch
        if isinstance(uri, (str, os.PathLike)) and os.path.exists(str(uri)):
            try:
                try:
                    import soundfile as sf
                    start = frame_offset if frame_offset > 0 else 0
                    frames = num_frames if num_frames > 0 else -1
                    data, sr = sf.read(str(uri), frames=frames, start=start, dtype="float32", always_2d=True)
                    # data is (frames, channels)
                    sig = data.T if channels_first else data
                    return torch.from_numpy(sig.astype(np.float32, copy=False)), sr
                except Exception:
                    data, sr = _load_wav_soundfile_tensor(uri, target_sr=None)
                    if not isinstance(data, torch.Tensor):
                        data = torch.from_numpy(data)
                    return data, sr
            except Exception:
                pass

        try:
            return orig_load(uri, frame_offset=frame_offset, num_frames=num_frames, normalize=normalize, channels_first=channels_first, format=format, buffer_size=buffer_size)
        except (RuntimeError, ImportError):
            if isinstance(uri, (str, os.PathLike)) and os.path.exists(str(uri)):
                data, sr = _load_wav_soundfile_tensor(uri, target_sr=None)
                if not isinstance(data, torch.Tensor):
                    data = torch.from_numpy(data)
                return data, sr
            raise

    _sf_load._dvo_sf_patched = True
    _sf_load._orig_load = orig_load
    torchaudio.load = _sf_load


def write_pcm16_16k(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())


def check_acoustic_and_speed_gate(
    audio_16k: np.ndarray,
    target_norm_text: str,
    min_cps: float = 1.8,
    max_cps: float = 7.5,
    max_duration_s: float = 12.0,
) -> tuple[bool, np.ndarray, dict]:
    """Stage A quality gate: trims excess dead air, checks truncation/clipping/speaking rate."""
    n_chars = count_speech_units(normalize_asr_text(target_norm_text))
    if len(audio_16k) < int(0.35 * TARGET_SR) or n_chars == 0:
        return False, audio_16k, {"reason": "empty_or_too_short_raw"}

    raw_dur = len(audio_16k) / TARGET_SR
    if raw_dur > max_duration_s:
        return False, audio_16k, {"reason": "exceeds_max_duration_looping", "raw_duration_s": round(raw_dur, 3)}

    raw_peak = float(np.max(np.abs(audio_16k)))
    if raw_peak < 1e-3:
        return False, audio_16k, {"reason": "near_silent_output"}

    # Gently normalize hot vocoder peaks above -1.0 dBFS to -2.0 dBFS before PCM16 quantization;
    # reject hard-clipped flat-top bursts where >0.5% of samples hit >= 0.999.
    clipped_ratio = float(np.mean(np.abs(audio_16k) >= 0.999))
    if clipped_ratio > 0.005:
        return False, audio_16k, {"reason": "severe_waveform_clipping", "clipped_ratio": round(clipped_ratio, 4)}

    sig = audio_16k.copy()
    if raw_peak > 0.89:
        sig = sig * (0.794 / raw_peak)  # -2.0 dBFS ceiling

    hop = 160  # 10 ms frames
    n_frames = len(sig) // hop
    frames = sig[:n_frames * hop].astype(np.float64).reshape(n_frames, hop)
    f_db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
    peak_frame_db = float(np.max(f_db))
    active_thresh = max(peak_frame_db - 32.0, -48.0)
    active = np.flatnonzero(f_db > active_thresh)
    if len(active) < 15:
        return False, sig, {"reason": "insufficient_voiced_frames"}

    first_f, last_f = int(active[0]), int(active[-1])
    # Check abrupt end truncation: if the very last 20 ms of raw output is still loud (> peak - 10 dB)
    if float(np.max(f_db[-2:])) > peak_frame_db - 10.0 and last_f >= n_frames - 2:
        return False, sig, {"reason": "abrupt_tail_truncation"}

    # Check internal dead-air collapse (> 1.25 s total silence inside a short utterance)
    voiced_span_frames = last_f - first_f + 1
    silent_run = 0
    max_silent_run = 0
    for val in f_db[first_f:last_f + 1]:
        if val < -50.0:
            silent_run += 1
            max_silent_run = max(max_silent_run, silent_run)
        else:
            silent_run = 0
    if max_silent_run > 125:
        return False, sig, {"reason": "mid_utterance_dead_air", "max_gap_s": round(max_silent_run * 0.01, 2)}

    effective_speech_s = voiced_span_frames * 0.01
    chars_per_sec = n_chars / max(effective_speech_s, 0.05)
    if not (min_cps <= chars_per_sec <= max_cps):
        return False, sig, {
            "reason": "abnormal_speaking_rate",
            "chars_per_sec": round(chars_per_sec, 2),
            "effective_speech_s": round(effective_speech_s, 3),
        }

    # Keep clean 0.10 s lead-in and 0.16 s tail padding for natural ASR boundary framing
    s0 = max(0, (first_f - 10) * hop)
    s1 = min(len(sig), (last_f + 16) * hop)
    cleaned = sig[s0:s1]
    if (first_f * hop - s0) < int(0.08 * TARGET_SR):
        cleaned = np.concatenate([np.zeros(int(0.08 * TARGET_SR), dtype=np.float32), cleaned])
    if (s1 - (last_f + 1) * hop) < int(0.12 * TARGET_SR):
        cleaned = np.concatenate([cleaned, np.zeros(int(0.14 * TARGET_SR), dtype=np.float32)])

    peak_db = 20.0 * math.log10(max(float(np.max(np.abs(cleaned))), 1e-6))
    metrics = {
        "duration_s": round(len(cleaned) / TARGET_SR, 3),
        "effective_speech_s": round(effective_speech_s, 3),
        "chars_per_sec": round(chars_per_sec, 2),
        "peak_dbfs": round(peak_db, 2),
    }
    return True, cleaned, metrics


def check_judge_asr_gate(
    target_norm: str,
    hyp_raw: str,
    strict_exact: bool,
    max_cmd_cer: float = 0.0,
    max_entity_cer: float = 0.20,
    allow_en_app_homophones: bool = False,
) -> tuple[bool, dict]:
    """Stage B quality gate: compares independent SenseVoice transcription against target.

    Normalizes both `target_norm` and `hyp_raw` to lowercase `[a-z0-9\\u4e00-\\u9fff]` so mixed
    Chinese-English app names (e.g. `"小克，打开 Steam"` vs `"<|zh|>小克打开STEAM"`) match with 0 CER.
    Strict English matching is preferred by default (`allow_en_app_homophones=False`).
    """
    ref_norm = normalize_asr_text(target_norm)
    hyp_norm = normalize_asr_text(hyp_raw)
    if not hyp_norm or not ref_norm:
        return False, {"reason": "judge_empty_transcript", "judge_text": "", "char_cer": 1.0, "pinyin_cer": 1.0}

    char_err = edit_distance_seq(ref_norm, hyp_norm)
    char_cer = char_err / max(1, len(ref_norm))

    ref_py = to_toneless_pinyin(ref_norm)
    hyp_py = to_toneless_pinyin(hyp_norm)
    py_err = edit_distance_seq(ref_py, hyp_py)
    pinyin_cer = py_err / max(1, len(ref_py))

    info: dict = {
        "judge_text": hyp_norm,
        "char_cer": round(char_cer, 4),
        "pinyin_cer": round(pinyin_cer, 4),
    }

    if strict_exact:
        if char_cer <= max_cmd_cer:
            return True, info
        if allow_en_app_homophones and re.search(r"[a-z]", ref_norm):
            hyp_mapped = hyp_norm
            for en_key, zh_forms in EN_APP_HOMOPHONE_MAP.items():
                if en_key in ref_norm:
                    for zf in zh_forms:
                        hyp_mapped = hyp_mapped.replace(zf, en_key)
            if hyp_mapped != hyp_norm:
                mapped_cer = edit_distance_seq(ref_norm, hyp_mapped) / max(1, len(ref_norm))
                if mapped_cer <= max_cmd_cer:
                    return True, {**info, "matched_en_homophone": True, "mapped_judge_text": hyp_mapped}
        return False, {**info, "reason": "command_cer_exceeded"}

    # Proper-noun song/artist request: pass if exact match OR toneless pinyin matches 100% within CER cap
    if char_cer == 0.0 or (pinyin_cer == 0.0 and char_cer <= max_entity_cer):
        return True, info
    return False, {**info, "reason": "entity_cer_or_pinyin_mismatch"}


def load_prompt_pool(pool_jsonl: Path) -> list[dict]:
    base_dir = pool_jsonl.parent
    pool = []
    for line in pool_jsonl.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        wp = Path(r["wav_path"])
        if not wp.is_absolute():
            wp = base_dir / wp
        if wp.is_file() and r.get("prompt_text"):
            pool.append({
                "id": r.get("id", wp.stem),
                "wav_path": wp,
                "prompt_text": r["prompt_text"],
            })
    return pool


def build_aishell_aux_pool(aishell_root: Path, count: int = 24, seed: int = 42) -> list[dict]:
    """Optionally builds a clean multi-speaker reference pool from AISHELL-1 to diversify 20% of TTS."""
    trans_file = aishell_root / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    wav_root = aishell_root / "aishell_wav"
    if not trans_file.is_file() or not wav_root.is_dir():
        return []
    transcript: dict[str, str] = {}
    for line in trans_file.read_text("utf-8").splitlines():
        key, _, txt = line.partition(" ")
        clean = txt.replace(" ", "").strip()
        if 8 <= len(clean) <= 18:
            transcript[key] = clean
    wavs = [w for w in sorted(wav_root.rglob("*.wav")) if w.stem in transcript]
    rng = random.Random(seed)
    rng.shuffle(wavs)
    aux: list[dict] = []
    seen_spk: set[str] = set()
    for w in wavs:
        spk = w.parent.name
        if spk in seen_spk and len(aux) < count // 2:
            continue
        try:
            with wave.open(str(w), "rb") as wf:
                dur = wf.getnframes() / max(1, wf.getframerate())
            if not (3.0 <= dur <= 6.5):
                continue
        except Exception:
            continue
        seen_spk.add(spk)
        aux.append({"id": f"aishell_{w.stem}", "wav_path": w, "prompt_text": transcript[w.stem]})
        if len(aux) >= count:
            break
    return aux


class CosyVoice2Synthesizer:
    """Wrapper around CosyVoice2-0.5B handling sys.path injection and API version differences."""

    def __init__(self, model_dir: Path, cosyvoice_repo: Path | None = None, fp16: bool = True):
        repo = cosyvoice_repo or Path(os.environ.get("COSYVOICE_REPO", "/content/CosyVoice"))
        if repo.is_dir():
            matcha = repo / "third_party" / "Matcha-TTS"
            if matcha.is_dir() and str(matcha) not in sys.path:
                sys.path.insert(0, str(matcha))
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

        _install_torchaudio_soundfile_shim()
        from cosyvoice.cli.cosyvoice import CosyVoice2
        import cosyvoice.utils.file_utils as cosy_file_utils

        # Patch cosyvoice load_wav to bypass torchcodec
        def _cosy_safe_load_wav(wav_file, target_sr=16000):
            return _load_wav_soundfile_tensor(wav_file, target_sr=target_sr)

        cosy_file_utils.load_wav = _cosy_safe_load_wav
        self._load_wav = _cosy_safe_load_wav

        init_sig = inspect.signature(CosyVoice2.__init__)
        kwargs = {}
        if "fp16" in init_sig.parameters:
            kwargs["fp16"] = fp16
        if "load_jit" in init_sig.parameters:
            kwargs["load_jit"] = False
        if "load_trt" in init_sig.parameters:
            kwargs["load_trt"] = False
        self.model = CosyVoice2(str(model_dir), **kwargs)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self._prompt_expects_path: bool | None = None

    def zero_shot(self, tts_text: str, prompt_text: str, prompt_wav: Path, speed: float = 1.0) -> np.ndarray:
        try:
            import torch
        except ImportError:
            torch = None

        def _collect(gen) -> np.ndarray:
            chunks = []
            for out in gen:
                speech = out["tts_speech"]
                if torch is not None and isinstance(speech, torch.Tensor):
                    speech = speech.detach().cpu().numpy()
                chunks.append(np.asarray(speech, dtype=np.float32).reshape(-1))
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        sig_params = inspect.signature(self.model.inference_zero_shot).parameters
        extra_kw = {"stream": False}
        if "speed" in sig_params:
            extra_kw["speed"] = speed

        if self._prompt_expects_path is True:
            return _collect(self.model.inference_zero_shot(tts_text, prompt_text, str(prompt_wav), **extra_kw))
        if self._prompt_expects_path is False:
            prompt_16k = self._load_wav(str(prompt_wav), 16000)
            return _collect(self.model.inference_zero_shot(tts_text, prompt_text, prompt_16k, **extra_kw))

        # Auto-probe once on first call so both newer (str path) and legacy (Tensor) CosyVoice2 APIs work.
        # Probe str(prompt_wav) FIRST matching modern CosyVoice2!
        try:
            audio = _collect(self.model.inference_zero_shot(tts_text, prompt_text, str(prompt_wav), **extra_kw))
            if len(audio) > 0:
                self._prompt_expects_path = True
            return audio
        except (TypeError, AssertionError, AttributeError, RuntimeError, ValueError):
            prompt_16k = self._load_wav(str(prompt_wav), 16000)
            audio = _collect(self.model.inference_zero_shot(tts_text, prompt_text, prompt_16k, **extra_kw))
            if len(audio) > 0:
                self._prompt_expects_path = False
            return audio


class SenseVoiceJudge:
    """Original un-finetuned sherpa-onnx SenseVoice recognizer acting as independent quality judge."""

    def __init__(self, model_dir: Path, num_threads: int = 4):
        import sherpa_onnx

        onnx_path = model_dir / "model.int8.onnx"
        if not onnx_path.is_file():
            onnx_path = model_dir / "model.onnx"
        tokens_path = model_dir / "tokens.txt"
        if not onnx_path.is_file() or not tokens_path.is_file():
            raise FileNotFoundError(f"Judge model/tokens missing in {model_dir}")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(onnx_path),
            tokens=str(tokens_path),
            language="zh",
            use_itn=False,
            num_threads=num_threads,
        )

    def transcribe(self, audio_16k: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(TARGET_SR, audio_16k.tolist())
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


def run_self_test() -> int:
    """Verifies resampling, speaking-rate gate, truncation/clipping detection, and pinyin CER gate."""
    rng = np.random.default_rng(42)
    # 1. Polyphase anti-aliased resampling 24 kHz -> 16 kHz
    sr_in = 24000
    t = np.arange(int(3.2 * sr_in), dtype=np.float64) / sr_in
    voiced = (0.35 * np.sin(2.0 * math.pi * 180.0 * t) * (0.7 + 0.3 * np.sin(2.0 * math.pi * 4.0 * t))).astype(np.float32)
    # Add 0.15 s lead and 0.20 s tail silence
    pad_in = np.concatenate([
        np.zeros(int(0.15 * sr_in), dtype=np.float32),
        voiced,
        np.zeros(int(0.20 * sr_in), dtype=np.float32),
    ])
    out_16k = resample_to_16k_poly(pad_in, sr_in)
    assert abs(len(out_16k) / TARGET_SR - len(pad_in) / sr_in) < 0.02

    # 2. Normal speaking rate (11 chars in ~3.2s = ~3.4 chars/s) -> PASS
    ok, cleaned, m = check_acoustic_and_speed_gate(out_16k, "小克我想听周杰伦的晴天")
    assert ok, f"Expected pass, got {m}"
    assert 2.5 <= m["chars_per_sec"] <= 4.5

    # 3. Stretched vocoder glitch (4 chars in 3.2s = 1.25 chars/s < 1.8) -> REJECT
    ok_slow, _, m_slow = check_acoustic_and_speed_gate(out_16k, "小克换歌")
    assert not ok_slow and m_slow["reason"] == "abnormal_speaking_rate"

    # 4. Judge ASR gate: strict command vs proper-noun homophone pinyin tolerance
    ok_cmd, _ = check_judge_asr_gate("小克换歌", "<|zh|>小克，换歌。", strict_exact=True)
    assert ok_cmd
    ok_cmd_bad, _ = check_judge_asr_gate("小克换歌", "<|zh|>小克放歌", strict_exact=True)
    assert not ok_cmd_bad

    # Proper-noun homophone: "晴天" vs "擎天" (same toneless pinyin qing tian, 1/11 char diff <= 0.20)
    ok_ent, info_ent = check_judge_asr_gate("小克我想听周杰伦的晴天", "小克，我想听周杰伦的擎天。", strict_exact=False)
    assert ok_ent and info_ent["pinyin_cer"] == 0.0
    # Non-homophone error in entity prompt -> REJECT
    ok_ent_bad, _ = check_judge_asr_gate("小克我想听周杰伦的晴天", "小克，我想听周杰伦的夜曲。", strict_exact=False)
    assert not ok_ent_bad

    # 5. Mixed Chinese-English desktop app normalization & strict judge checks
    ok_steam, info_steam = check_judge_asr_gate("小克，打开 Steam", "小克打开steam", strict_exact=True)
    assert ok_steam and info_steam["char_cer"] == 0.0, f"Expected 0 CER on Steam, got {info_steam}"
    ok_steam_upper, info_steam_upper = check_judge_asr_gate(
        "小克打开steam", "<|zh|>小克，打开 STEAM。", strict_exact=True
    )
    assert ok_steam_upper and info_steam_upper["char_cer"] == 0.0
    ok_ps, info_ps = check_judge_asr_gate(
        "小克，打开 PowerShell", "<|zh|>小克打开 Power Shell", strict_exact=True
    )
    assert ok_ps and info_ps["char_cer"] == 0.0
    # Strict mode rejects Chinese phonetic substitute by default, optional homophone mode accepts it
    ok_hom_strict, _ = check_judge_asr_gate("小克，打开 Steam", "小克打开斯蒂姆", strict_exact=True)
    assert not ok_hom_strict
    ok_hom_opt, info_hom_opt = check_judge_asr_gate(
        "小克，打开 Steam", "小克打开斯蒂姆", strict_exact=True, allow_en_app_homophones=True
    )
    assert ok_hom_opt and info_hom_opt.get("matched_en_homophone") is True
    # Wrong English app name must always fail
    ok_wrong_app, _ = check_judge_asr_gate(
        "小克，打开 Steam", "小克打开clash", strict_exact=True, allow_en_app_homophones=True
    )
    assert not ok_wrong_app

    # 6. Unit test (a): torchaudio soundfile shim and _load_wav_soundfile_tensor
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "test_24k.wav"
        sr_test = 24000
        n_samples = int(0.5 * sr_test)
        samples = (0.2 * np.sin(2.0 * math.pi * 440.0 * np.arange(n_samples) / sr_test)).astype(np.float32)
        pcm_bytes = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()
        with wave.open(str(tmp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_test)
            wf.writeframes(pcm_bytes)

        _install_torchaudio_soundfile_shim()

        try:
            import torchaudio
            t_wave, t_sr = torchaudio.load(str(tmp_wav))
            assert t_sr == 24000
            assert t_wave.shape[0] == 1
        except ImportError:
            t_arr, t_sr = _load_wav_soundfile_tensor(tmp_wav, target_sr=None)
            assert t_sr == 24000
            assert t_arr.shape[0] == 1

        res_16k = _load_wav_soundfile_tensor(tmp_wav, target_sr=16000)
        assert res_16k.shape[0] == 1
        expected_len = int(round(n_samples * 16000 / 24000))
        assert abs(res_16k.shape[1] - expected_len) <= 2

    # 7. Unit test (b): zero_shot auto-probe logic mocked with fake model
    class MockCosyModelPathOnly:
        def __init__(self):
            self.sample_rate = 24000
            self.calls = []

        def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, **kwargs):
            self.calls.append(type(prompt_wav))
            if isinstance(prompt_wav, str):
                return [{"tts_speech": np.zeros(1600, dtype=np.float32)}]
            raise RuntimeError("Failed to create AudioDecoder for tensor(...): video_tensor must be kUInt8")

    class MockCosyModelTensorOnly:
        def __init__(self):
            self.sample_rate = 24000
            self.calls = []

        def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, **kwargs):
            self.calls.append(type(prompt_wav))
            if not isinstance(prompt_wav, str):
                return [{"tts_speech": np.zeros(1600, dtype=np.float32)}]
            raise TypeError("expected Tensor, got str")

    with tempfile.TemporaryDirectory() as td:
        dummy_wav = Path(td) / "dummy.wav"
        with wave.open(str(dummy_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(160, dtype="<i2").tobytes())

        # Test path-only model (probe str succeeds first)
        synth_path = CosyVoice2Synthesizer.__new__(CosyVoice2Synthesizer)
        synth_path.model = MockCosyModelPathOnly()
        synth_path._load_wav = _load_wav_soundfile_tensor
        synth_path.sample_rate = 24000
        synth_path._prompt_expects_path = None
        out_a = synth_path.zero_shot("测试", "参考", dummy_wav)
        assert len(out_a) == 1600
        assert synth_path._prompt_expects_path is True

        # Test tensor-only model (probe str fails with TypeError, falls back to tensor)
        synth_tensor = CosyVoice2Synthesizer.__new__(CosyVoice2Synthesizer)
        synth_tensor.model = MockCosyModelTensorOnly()
        synth_tensor._load_wav = _load_wav_soundfile_tensor
        synth_tensor.sample_rate = 24000
        synth_tensor._prompt_expects_path = None
        out_b = synth_tensor.zero_shot("测试", "参考", dummy_wav)
        assert len(out_b) == 1600
        assert synth_tensor._prompt_expects_path is False

    # 8. Unit test (c): safe --resume filtering of tts_exception entries
    with tempfile.TemporaryDirectory() as td:
        test_dir = Path(td)
        man_f = test_dir / "manifest.jsonl"
        rej_f = test_dir / "rejected.jsonl"
        man_f.write_text(json.dumps({"id": "item_ok", "audio": "clip.wav"}) + "\n", "utf-8")
        rej_content = [
            {"id": "item_crashed", "attempts": [{"attempt": 1, "reason": "tts_exception:RuntimeError('torchcodec')"}]},
            {"id": "item_quality_fail", "attempts": [{"attempt": 1, "reason": "abnormal_speaking_rate"}]},
        ]
        rej_f.write_text("\n".join(json.dumps(x) for x in rej_content) + "\n", "utf-8")

        d_ids: set[str] = set()
        v_rejs: list[str] = []
        if man_f.is_file():
            for line in man_f.read_text("utf-8").splitlines():
                if line.strip():
                    d_ids.add(json.loads(line)["id"])
        if rej_f.is_file():
            for line in rej_f.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                atts = rec.get("attempts", [])
                pure_c = bool(atts) and all(str(a.get("reason", "")).startswith("tts_exception:") for a in atts)
                if pure_c:
                    continue
                d_ids.add(rec["id"])
                v_rejs.append(line)
            rej_f.write_text("\n".join(v_rejs) + ("\n" if v_rejs else ""), "utf-8")

        assert "item_ok" in d_ids
        assert "item_quality_fail" in d_ids
        assert "item_crashed" not in d_ids
        remaining_rejs = [json.loads(x)["id"] for x in rej_f.read_text("utf-8").splitlines() if x.strip()]
        assert remaining_rejs == ["item_quality_fail"]

    print(
        "SELF-TEST PASSED! Resampling, acoustic/rate gate, and mixed ZH-EN SenseVoice judge gate verified."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--prompts", type=Path, help="input prompts JSONL (from build_nl_tts_prompts.py)")
    parser.add_argument("--prompt-pool", type=Path, help="personal golden prompt_pool.jsonl")
    parser.add_argument("--extra-prompt-pool", type=Path,
                        help="optional auxiliary multi-speaker prompt_pool.jsonl for the 20%% diversity share")
    parser.add_argument("--aishell-dir", type=Path,
                        help="optional AISHELL-1 root to auto-extract clean auxiliary prompts if --extra-prompt-pool is unset")
    parser.add_argument("--personal-ratio", type=float, default=0.80,
                        help="fraction of utterances cloned from the user's personal voice pool (default: 0.80)")
    parser.add_argument("--cosyvoice-model", type=Path, help="path to CosyVoice2-0.5B directory")
    parser.add_argument("--cosyvoice-repo", type=Path, default=None,
                        help="path to cloned CosyVoice repository (or env COSYVOICE_REPO)")
    parser.add_argument("--judge-model-dir", type=Path,
                        help="path to original un-finetuned sherpa-onnx SenseVoice directory")
    parser.add_argument("--out-dir", type=Path, help="output directory for WAV clips, manifest.jsonl, rejected.jsonl")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="extra retries with different reference prompts on quality-gate failure")
    parser.add_argument("--min-cps", type=float, default=1.8)
    parser.add_argument("--max-cps", type=float, default=7.5)
    parser.add_argument("--max-cmd-cer", type=float, default=0.0)
    parser.add_argument("--max-entity-cer", type=float, default=0.20)
    parser.add_argument("--allow-en-app-homophones", action="store_true",
                        help="accept curated Chinese phonetic transliterations for English app names if judge outputs CJK")
    parser.add_argument("--limit", type=int, default=0, help="optional cap on number of prompts to process")
    parser.add_argument("--resume", action="store_true",
                        help="skip prompts already present in existing manifest.jsonl or rejected.jsonl")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--self-test", action="store_true", help="run unit tests without GPU models and exit")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    required = [
        ("--prompts", args.prompts),
        ("--prompt-pool", args.prompt_pool),
        ("--cosyvoice-model", args.cosyvoice_model),
        ("--judge-model-dir", args.judge_model_dir),
        ("--out-dir", args.out_dir),
    ]
    missing = [flag for flag, val in required if val is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    personal_pool = load_prompt_pool(args.prompt_pool)
    if not personal_pool:
        raise RuntimeError(f"No valid reference prompts loaded from {args.prompt_pool}")

    aux_pool: list[dict] = []
    if args.extra_prompt_pool and args.extra_prompt_pool.is_file():
        aux_pool = load_prompt_pool(args.extra_prompt_pool)
    elif args.aishell_dir and args.aishell_dir.is_dir():
        aux_pool = build_aishell_aux_pool(args.aishell_dir, count=24, seed=args.seed)

    prompt_items = [
        json.loads(line)
        for line in args.prompts.read_text("utf-8").splitlines()
        if line.strip()
    ]
    if args.mode == "pilot" and args.limit == 0:
        prompt_items = prompt_items[:25]
    elif args.limit > 0:
        prompt_items = prompt_items[:args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    rejected_path = args.out_dir / "rejected.jsonl"

    done_ids: set[str] = set()
    accepted_count = 0
    rejected_count = 0
    if args.resume:
        if manifest_path.is_file():
            for line in manifest_path.read_text("utf-8").splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
                    accepted_count += 1
        if rejected_path.is_file():
            valid_rejections: list[str] = []
            for line in rejected_path.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                attempts = entry.get("attempts", [])
                pure_crash = bool(attempts) and all(
                    str(att.get("reason", "")).startswith("tts_exception:")
                    for att in attempts
                )
                if pure_crash:
                    # Environment crash rather than genuine quality gate rejection; do NOT skip on resume!
                    continue
                done_ids.add(entry["id"])
                valid_rejections.append(line)
                rejected_count += 1
            # Rewrite rejected_path keeping only genuine quality-gate rejections
            rejected_path.write_text(
                "\n".join(valid_rejections) + ("\n" if valid_rejections else ""),
                "utf-8",
            )
    else:
        manifest_path.unlink(missing_ok=True)
        rejected_path.unlink(missing_ok=True)

    print(f"Loading CosyVoice2 from {args.cosyvoice_model} ...")
    tts_engine = CosyVoice2Synthesizer(args.cosyvoice_model, cosyvoice_repo=args.cosyvoice_repo)
    print(f"Loading independent SenseVoice judge from {args.judge_model_dir} ...")
    judge = SenseVoiceJudge(args.judge_model_dir)

    rng = random.Random(args.seed)
    stats = Counter()
    t_start = time.perf_counter()
    consecutive_tts_exceptions = 0
    last_exc = ""

    with open(manifest_path, "a", encoding="utf-8", newline="\n") as mf, \
         open(rejected_path, "a", encoding="utf-8", newline="\n") as rf:

        for idx, item in enumerate(prompt_items, 1):
            pid = item["id"]
            if pid in done_ids:
                continue

            tts_text = item["tts_text"]
            target_norm = item.get("text") or normalize_cjk(tts_text)
            strict_exact = bool(item.get("strict_exact", not bool(item.get("entities"))))

            use_personal = (rng.random() < args.personal_ratio)
            active_pool = personal_pool if (use_personal or not aux_pool) else aux_pool
            voice_source = "personal" if active_pool is personal_pool else "aux_diverse"
            # When diversify slot falls back to personal pool (no aux_pool provided), vary speed slightly
            speed = 1.0 if use_personal or aux_pool else rng.choice([0.94, 1.06])

            # Shuffle candidate references so retries always try a fresh reference prompt
            ref_candidates = list(active_pool)
            rng.shuffle(ref_candidates)
            max_attempts = 1 + max(0, args.max_retries)

            passed_record = None
            attempt_logs = []

            for attempt in range(1, max_attempts + 1):
                ref = ref_candidates[(attempt - 1) % len(ref_candidates)]
                try:
                    raw_wav = tts_engine.zero_shot(
                        tts_text=tts_text,
                        prompt_text=ref["prompt_text"],
                        prompt_wav=ref["wav_path"],
                        speed=speed,
                    )
                    wav_16k = resample_to_16k_poly(raw_wav, tts_engine.sample_rate)
                except Exception as exc:
                    attempt_logs.append({"attempt": attempt, "ref_id": ref["id"], "reason": f"tts_exception:{exc}"})
                    continue

                ok_a, cleaned_16k, meta_a = check_acoustic_and_speed_gate(
                    wav_16k, target_norm, min_cps=args.min_cps, max_cps=args.max_cps
                )
                if not ok_a:
                    stats[f"retry_{meta_a['reason']}"] += 1
                    attempt_logs.append({"attempt": attempt, "ref_id": ref["id"], **meta_a})
                    continue

                hyp_raw = judge.transcribe(cleaned_16k)
                ok_b, meta_b = check_judge_asr_gate(
                    target_norm=target_norm,
                    hyp_raw=hyp_raw,
                    strict_exact=strict_exact,
                    max_cmd_cer=args.max_cmd_cer,
                    max_entity_cer=args.max_entity_cer,
                    allow_en_app_homophones=args.allow_en_app_homophones,
                )
                if not ok_b:
                    stats[f"retry_{meta_b['reason']}"] += 1
                    attempt_logs.append({"attempt": attempt, "ref_id": ref["id"], **meta_a, **meta_b})
                    continue

                rel_wav = f"clips/tts/{pid}.wav"
                write_pcm16_16k(args.out_dir / rel_wav, cleaned_16k)
                passed_record = {
                    "id": pid,
                    "audio": rel_wav,
                    "reference": None,
                    "split": "train",
                    "kind": "tts_cloned",
                    "category": item.get("category", "tts"),
                    "prompt": tts_text,
                    "text": target_norm,
                    "music": False,
                    "condition": "quiet",
                    "voice_source": voice_source if use_personal or aux_pool else "personal_speed_perturbed",
                    "ref_prompt_id": ref["id"],
                    "attempt": attempt,
                    **meta_a,
                    **meta_b,
                }
                break

            if passed_record is not None:
                mf.write(json.dumps(passed_record, ensure_ascii=False) + "\n")
                mf.flush()
                accepted_count += 1
                stats["accepted"] += 1
                consecutive_tts_exceptions = 0
                if passed_record["attempt"] > 1:
                    stats["recovered_by_retry"] += 1
            else:
                rej_entry = {
                    "id": pid,
                    "tts_text": tts_text,
                    "text": target_norm,
                    "strict_exact": strict_exact,
                    "attempts": attempt_logs,
                }
                rf.write(json.dumps(rej_entry, ensure_ascii=False) + "\n")
                rf.flush()
                rejected_count += 1
                stats["quarantined_rejected"] += 1

                # Fail-fast check: if the first 5 processed items in a row fail with pure tts_exception
                is_pure_tts_exc = bool(attempt_logs) and all(
                    str(att.get("reason", "")).startswith("tts_exception:")
                    for att in attempt_logs
                )
                if is_pure_tts_exc:
                    consecutive_tts_exceptions += 1
                    last_exc = attempt_logs[-1].get("reason", "")
                    if consecutive_tts_exceptions >= 5:
                        err_msg = f"Aborting: {consecutive_tts_exceptions} consecutive items failed with TTS exception: {last_exc}"
                        print(f"\n[FATAL ERROR] {err_msg}", file=sys.stderr)
                        raise RuntimeError(err_msg)
                else:
                    consecutive_tts_exceptions = 0

            if idx % 25 == 0 or idx == len(prompt_items):
                elapsed = time.perf_counter() - t_start
                total_seen = accepted_count + rejected_count
                acc_rate = 100.0 * accepted_count / max(1, total_seen)
                print(
                    f"[{idx}/{len(prompt_items)}] accepted={accepted_count} rejected={rejected_count} "
                    f"(pass={acc_rate:.1f}%) elapsed={elapsed:.1f}s"
                )

    total_processed = accepted_count + rejected_count
    report = {
        "mode": args.mode,
        "total_processed": total_processed,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "pass_rate_percent": round(100.0 * accepted_count / max(1, total_processed), 2),
        "gate_stats": dict(stats),
    }
    (args.out_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print("\n=== TTS Synthesis & Closed-Loop Quality Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if len(prompt_items) > 0 and accepted_count == 0 and total_processed > 0:
        top_reasons = stats.most_common(5) if hasattr(stats, "most_common") else list(stats.items())[:5]
        print(
            f"\n[ERROR] 0/{total_processed} items accepted! Top rejection/exception reasons: {top_reasons}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
