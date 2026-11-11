"""Fine-tune the runtime's final decoder (Fun-ASR-Nano CTC branch) on the speaker.

The runtime runs sherpa-onnx's export of Fun-ASR-Nano-2512's CTC branch:
SenseVoiceEncoderSmall -> 5-layer Transformer ("ctc_decoder") -> CTC output
layer. This trains that branch with sherpa-onnx's own PyTorch definition
(scripts/sense-voice/rknn/nano.py), so training/asr/export.py produces a
drop-in replacement for models/sherpa-onnx-sense-voice-funasr-nano-*.

Training data:
  * The speaker's real transcribed takes (commands, read sentences, and dual-ASR
    verified game utterances) repeated `--repeat` times per epoch so the user's
    real microphone acoustics remain the primary anchor;
  * Optional synthesized personal-cloned TTS utterances (`--extra-manifests`),
    dynamically subsampled up to `--tts-max-per-epoch` per epoch so vocabulary
    and song titles are expanded without diluting loudspeaker-music robustness;
  * Optional slice of AISHELL-1 (`--aishell-utterances`) to preserve general Mandarin.

Half of ALL utterances (including synthesized TTS clips!) get recorded speaker
music mixed in at 0-15 dB SNR, which both preserves music-playback accuracy and
masks high-frequency vocoder artifacts.

Runs in Google Colab (L4 / A100) or WSL conda env `funasr`:

    python training/asr/finetune.py --exp-dir exp/asr-personal-v2 \\
        --dataset /path/to/dataset \\
        --extra-manifests /path/to/synth_dataset/manifest.jsonl \\
        --repeat 4 --tts-max-per-epoch 1800 --unfreeze-encoder-layers 3
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RATE = 16000
SPEECH_KINDS = {"command", "wake", "bare_command", "negative", "read", "game_verified"}
TTS_KINDS = {"tts_cloned", "tts"}
NOISE_KINDS = {"background", "silence", "unlabeled"}


def default_sherpa_nano_dir() -> Path:
    env_val = os.environ.get("DVO_SHERPA_NANO_DIR")
    if env_val:
        return Path(env_val)
    return HOME / "dvo/sherpa-onnx/scripts/sense-voice/rknn"


def default_tokens_path() -> Path:
    env_val = os.environ.get("DVO_SHERPA_TOKENS")
    if env_val:
        return Path(env_val)
    local_candidate = REPO_ROOT / "models/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17/tokens.txt"
    if local_candidate.is_file():
        return local_candidate
    return Path("/mnt/e/workspace/desktop-voice-orchestrator/models/"
                "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17/tokens.txt")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("/mnt/e/workspace/desktop-voice-orchestrator/data/dataset"))
    parser.add_argument("--extra-manifests", type=Path, nargs="*", default=[],
                        help="additional manifest.jsonl files (e.g. cloned TTS corpus, verified game speech)")
    parser.add_argument("--tts-max-per-epoch", type=int, default=1800,
                        help="max synthesized TTS utterances randomly sampled per epoch (0 = use all)")
    parser.add_argument("--sherpa-nano-dir", type=Path, default=default_sherpa_nano_dir(),
                        help="directory containing sherpa-onnx's nano.py (or env DVO_SHERPA_NANO_DIR)")
    parser.add_argument("--tokens", type=Path, default=default_tokens_path(),
                        help="path to sherpa-onnx Fun-ASR-Nano tokens.txt (or env DVO_SHERPA_TOKENS)")
    parser.add_argument("--pretrained", type=Path, default=HOME / "dvo/models/Fun-ASR-Nano-2512")
    parser.add_argument("--aishell", type=Path, default=HOME / "dvo/corpora")
    parser.add_argument("--aishell-utterances", type=int, default=3000,
                        help="number of AISHELL-1 utterances to include (set 0 to skip AISHELL)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="how often each real personal utterance appears per epoch")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4, help="ctc_decoder learning rate")
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=0,
                        help="also train this many of the encoder's last tp_encoders blocks")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--train-output-layer", action="store_true",
                        help="also train the CTC output layer (moves the vocabulary prior)")
    parser.add_argument("--accum", type=int, default=16, help="utterances per optimizer step")
    parser.add_argument("--mix-probability", type=float, default=0.5)
    parser.add_argument("--snr", type=float, nargs=2, default=(0.0, 15.0))
    parser.add_argument("--concat-probability", type=float, default=0.3)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def import_nano_module(sherpa_nano_dir: Path):
    nano_py = sherpa_nano_dir / "nano.py"
    if not nano_py.is_file():
        raise FileNotFoundError(
            f"Cannot find sherpa-onnx's nano.py at '{nano_py}'.\n"
            f"Please pass --sherpa-nano-dir <path/to/sherpa-onnx/scripts/sense-voice/rknn> "
            f"or set environment variable DVO_SHERPA_NANO_DIR."
        )
    if str(sherpa_nano_dir) not in sys.path:
        sys.path.insert(0, str(sherpa_nano_dir))
    import nano  # type: ignore
    return nano


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2, (
            f"{path}: expected 16 kHz 16-bit WAV, got {w.getframerate()} Hz / {w.getsampwidth() * 8}-bit"
        )
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]
    return audio.astype(np.float32) / 32768.0


def features(samples: np.ndarray, knf_mod, torch_mod):
    """Exactly sherpa-onnx's front end for this model: 80-bin fbank, LFR 7/6."""
    opts = knf_mod.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = RATE
    opts.mel_opts.num_bins = 80
    fbank = knf_mod.OnlineFbank(opts)
    fbank.accept_waveform(RATE, (samples * 32768.0).tolist())
    fbank.input_finished()
    if fbank.num_frames_ready < 7:
        pad_needed = int((0.15 * RATE))
        fbank.accept_waveform(RATE, [0.0] * pad_needed)
        fbank.input_finished()
    frames = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    count = (frames.shape[0] - 7) // 6 + 1
    lfr = np.stack([frames[t * 6:t * 6 + 7].reshape(-1) for t in range(count)])
    return torch_mod.from_numpy(lfr.astype(np.float32))


class Tokenizer:
    def __init__(self, pretrained: Path, sherpa_tokens: Path):
        from funasr.models.sense_voice.whisper_lib.tokenizer import get_tokenizer

        if not sherpa_tokens.is_file():
            raise FileNotFoundError(
                f"sherpa-onnx tokens.txt not found at '{sherpa_tokens}'. "
                f"Pass --tokens <path/to/tokens.txt> or set DVO_SHERPA_TOKENS."
            )
        self.encoder = get_tokenizer(
            multilingual=True,
            num_languages=8749,
            vocab_path=str(pretrained / "multilingual.tiktoken"),
        )
        with open(sherpa_tokens, encoding="utf-8") as f:
            lines = f.readlines()
        self.blank = len(lines) - 1
        self.pieces = {}
        for line in lines:
            parts = line.strip().split()
            self.pieces[int(parts[-1])] = base64.b64decode(parts[0]) if len(parts) == 2 else b" "

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.pieces[i] for i in ids if i != self.blank).decode("utf-8", "replace")


def resolve_audio_path(record_audio: str, manifest_parent: Path, fallback_root: Path | None = None) -> Path:
    p = Path(record_audio)
    if p.is_absolute() and p.is_file():
        return p
    cand = manifest_parent / p
    if cand.is_file():
        return cand
    if fallback_root is not None:
        cand2 = fallback_root / p
        if cand2.is_file():
            return cand2
    return cand


def load_manifest_records(
    manifest_path: Path,
    split: str,
    fallback_root: Path | None = None,
) -> tuple[list[tuple[np.ndarray, str]], list[tuple[np.ndarray, str]]]:
    """Returns (real_personal_items, tts_items) for the requested split."""
    real_items: list[tuple[np.ndarray, str]] = []
    tts_items: list[tuple[np.ndarray, str]] = []
    base_dir = manifest_path.parent
    for line in manifest_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") != split or not record.get("text"):
            continue
        kind = record.get("kind", "")
        wav_path = resolve_audio_path(record["audio"], base_dir, fallback_root)
        if not wav_path.is_file():
            continue
        if kind in SPEECH_KINDS:
            real_items.append((read_wav(wav_path), record["text"]))
        elif kind in TTS_KINDS:
            tts_items.append((read_wav(wav_path), record["text"]))
    return real_items, tts_items


def load_noise(dataset: Path, extra_manifests: list[Path] | None = None) -> list[np.ndarray]:
    noise: list[np.ndarray] = []
    manifest = dataset / "manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") in ("train", "unlabeled") and record.get("kind") in NOISE_KINDS \
                    and record.get("music"):
                wav_path = dataset / record["audio"]
                if wav_path.is_file():
                    noise.append(read_wav(wav_path))

    # Fallback for Colab setups where reference/ loopback music clips exist even if background music
    # clips are scarce: speaker loopback wavs provide authentic music interference for SNR mixing.
    if not noise and (dataset / "reference").is_dir():
        for ref_wav in sorted((dataset / "reference").rglob("*.wav")):
            sig = read_wav(ref_wav)
            if len(sig) >= RATE * 2 and float(np.sqrt(np.mean(sig ** 2))) > 0.006:
                noise.append(sig)

    for extra in (extra_manifests or []):
        if extra.is_file():
            for line in extra.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") == "train" and r.get("kind") in NOISE_KINDS and r.get("music"):
                    wp = resolve_audio_path(r["audio"], extra.parent, dataset)
                    if wp.is_file():
                        noise.append(read_wav(wp))
    return noise


def load_aishell(root: Path, count: int, seed: int, skip: int = 0) -> list[tuple[np.ndarray, str]]:
    if count <= 0:
        return []
    trans_path = root / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    wav_dir = root / "aishell_wav"
    if not trans_path.is_file() or not wav_dir.is_dir():
        logging.warning("AISHELL-1 not found under %s; skipping general Mandarin slice.", root)
        return []
    transcript = {}
    for line in trans_path.read_text("utf-8").splitlines():
        key, _, text = line.partition(" ")
        transcript[key] = text.replace(" ", "")
    wavs = [w for w in sorted(wav_dir.rglob("*.wav")) if w.stem in transcript]
    random.Random(seed).shuffle(wavs)
    return [(read_wav(w), transcript[w.stem]) for w in wavs[skip:skip + count]]


def mix(speech: np.ndarray, noise: list[np.ndarray], snr: tuple[float, float],
        rng: random.Random) -> np.ndarray:
    if not noise:
        return speech
    source = noise[rng.randrange(len(noise))]
    pieces, needed = [], len(speech)
    while needed > 0:
        start = rng.randrange(max(1, len(source) - min(needed, len(source)) + 1))
        piece = source[start:start + needed]
        pieces.append(piece)
        needed -= len(piece)
    background = np.concatenate(pieces)[:len(speech)]
    speech_power = float(np.mean(speech ** 2)) + 1e-10
    noise_power = float(np.mean(background ** 2)) + 1e-10
    target = speech_power / (10 ** (rng.uniform(*snr) / 10))
    mixed = speech + background * np.sqrt(target / noise_power)
    peak = float(np.max(np.abs(mixed)))
    return (mixed / peak * 0.95 if peak > 0.99 else mixed).astype(np.float32)


def build_epoch_pool(
    personal: list[tuple[np.ndarray, str]],
    tts_pool: list[tuple[np.ndarray, str]],
    general: list[tuple[np.ndarray, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[tuple[np.ndarray, str]]:
    """Constructs one epoch's balanced pool: anchored on repeated real speech + capped TTS slice."""
    if tts_pool:
        if args.tts_max_per_epoch > 0 and len(tts_pool) > args.tts_max_per_epoch:
            tts_slice = rng.sample(tts_pool, args.tts_max_per_epoch)
        else:
            tts_slice = list(tts_pool)
    else:
        tts_slice = []
    return personal * args.repeat + tts_slice + general


def epoch_items(pool: list[tuple[np.ndarray, str]], args: argparse.Namespace, rng: random.Random):
    """Yields (samples, text), with some utterances concatenated."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    for index in order:
        audio, text = pool[index]
        if rng.random() < args.concat_probability:
            parts, texts = [audio], [text]
            for _ in range(rng.randint(1, 2)):
                other_audio, other_text = pool[rng.randrange(len(pool))]
                total = sum(len(p) for p in parts) + len(other_audio)
                if total / RATE > args.max_seconds:
                    break
                gap = np.zeros(int(rng.uniform(0.2, 0.6) * RATE), dtype=np.float32)
                parts += [gap, other_audio]
                texts.append(other_text)
            audio, text = np.concatenate(parts), "".join(texts)
        yield audio, text


def cer(reference: str, hypothesis: str) -> tuple[int, int]:
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1], len(reference)


def evaluate(model, items, tokenizer, device, knf_mod, torch_mod) -> float | None:
    if not items:
        return None
    was_training = model.training
    model.eval()
    errors = total = 0
    with torch_mod.no_grad():
        for audio, text in items:
            logits = model(features(audio, knf_mod, torch_mod)[None].to(device))[0]
            ids = torch_mod.unique_consecutive(logits.argmax(-1)).tolist()
            e, n = cer(text, tokenizer.decode(ids).strip())
            errors += e
            total += n
    model.train(was_training)
    return round(errors / max(1, total), 5)


def set_trainable(model, args: argparse.Namespace) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = [{"params": list(model.ctc_decoder.parameters()), "lr": args.lr}]
    if args.train_output_layer:
        groups[0]["params"] += list(model.ctc.parameters())
    if args.unfreeze_encoder_layers > 0:
        encoder = model.audio_encoder
        blocks = list(encoder.tp_encoders)[-args.unfreeze_encoder_layers:]
        params = [p for block in blocks for p in block.parameters()] + list(encoder.tp_norm.parameters())
        groups.append({"params": params, "lr": args.encoder_lr})
    for group in groups:
        for parameter in group["params"]:
            parameter.requires_grad_(True)
    return groups


def main() -> None:
    args = get_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    import kaldi_native_fbank as knf
    import torch

    nano = import_nano_module(args.sherpa_nano_dir)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA is not available; training on CPU will be slow.")

    tokenizer = Tokenizer(args.pretrained, args.tokens)
    model = nano.Nano()
    state = torch.load(args.pretrained / "model.pt", map_location="cpu", weights_only=False)
    state = {k: v for k, v in state.items() if not k.startswith(("llm", "audio_adaptor"))}
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    groups = set_trainable(model, args)
    trainable = sum(p.numel() for g in groups for p in g["params"])
    logging.info("training %.1fM of %.1fM parameters (unfreeze_encoder_layers=%d)",
                 trainable / 1e6, sum(p.numel() for p in model.parameters()) / 1e6,
                 args.unfreeze_encoder_layers)

    main_manifest = args.dataset / "manifest.jsonl"
    if not main_manifest.is_file():
        raise FileNotFoundError(f"Primary dataset manifest not found: {main_manifest}")

    personal, tts_pool = load_manifest_records(main_manifest, "train", args.dataset)
    dev, _ = load_manifest_records(main_manifest, "dev", args.dataset)

    for extra_mf in args.extra_manifests:
        if not extra_mf.is_file():
            raise FileNotFoundError(f"Extra manifest not found: {extra_mf}")
        ext_real, ext_tts = load_manifest_records(extra_mf, "train", args.dataset)
        ext_dev, _ = load_manifest_records(extra_mf, "dev", args.dataset)
        personal.extend(ext_real)
        tts_pool.extend(ext_tts)
        dev.extend(ext_dev)

    general = load_aishell(args.aishell, args.aishell_utterances, args.seed)
    general_dev = load_aishell(args.aishell, 200 if args.aishell_utterances > 0 else 0,
                               args.seed, skip=args.aishell_utterances)
    noise = load_noise(args.dataset, list(args.extra_manifests))

    effective_tts_per_epoch = (
        min(len(tts_pool), args.tts_max_per_epoch) if args.tts_max_per_epoch > 0 else len(tts_pool)
    )
    logging.info(
        "personal real %d (%.1f min) x%d, tts_pool %d (per-epoch cap %d), "
        "dev %d, aishell %d (+%d dev), music/noise clips %d",
        len(personal), sum(len(a) for a, _ in personal) / RATE / 60, args.repeat,
        len(tts_pool), effective_tts_per_epoch,
        len(dev), len(general), len(general_dev), len(noise),
    )

    approx_pool_len = len(personal) * args.repeat + effective_tts_per_epoch + len(general)
    if approx_pool_len == 0:
        raise RuntimeError("Training pool is empty! Check --dataset and --extra-manifests.")

    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = max(1, approx_pool_len // args.accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup = min(100, max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.1, 1 - s / max(1, total_steps)),
    )

    summary = {
        "args": {k: ([str(x) for x in v] if isinstance(v, list) else str(v)) for k, v in vars(args).items()},
        "data_counts": {
            "personal_real": len(personal),
            "personal_repeat": args.repeat,
            "tts_total": len(tts_pool),
            "tts_per_epoch": effective_tts_per_epoch,
            "aishell": len(general),
            "noise_clips": len(noise),
        },
        "epochs": [],
    }
    summary["pretrained"] = {
        "dev_cer": evaluate(model, dev, tokenizer, device, knf, torch),
        "aishell_dev_cer": evaluate(model, general_dev, tokenizer, device, knf, torch),
    }
    logging.info("pretrained %s", summary["pretrained"])

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_pool = build_epoch_pool(personal, tts_pool, general, args, rng)
        model.train()
        if args.unfreeze_encoder_layers == 0:
            model.audio_encoder.eval()  # frozen: no dropout
        running, count = 0.0, 0
        optimizer.zero_grad()
        for audio, text in epoch_items(epoch_pool, args, rng):
            # Dynamically mix speaker background music into both real AND synthesized TTS speech
            # (masks high-frequency vocoder artifacts while preserving loudspeaker noise robustness)
            if rng.random() < args.mix_probability:
                audio = mix(audio, noise, tuple(args.snr), rng)
            x = features(audio, knf, torch)[None].to(device)
            targets = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            if len(targets) == 0:
                continue
            log_probs = model(x).log_softmax(-1).transpose(0, 1)  # (T, 1, V)
            if log_probs.size(0) < len(targets):
                continue
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                targets[None].to(device),
                input_lengths=torch.tensor([log_probs.size(0)]),
                target_lengths=torch.tensor([len(targets)]),
                blank=tokenizer.blank,
                reduction="mean",
                zero_infinity=True,
            )
            (loss / args.accum).backward()
            running += loss.item()
            count += 1
            if count % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 25 == 0:
                    logging.info("epoch %d step %d loss %.4f lr %.2e", epoch, step,
                                 running / args.accum / 25, scheduler.get_last_lr()[0])
                    running = 0.0
        if count % args.accum != 0:
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        entry = {
            "epoch": epoch,
            "steps": step,
            "epoch_utterances": len(epoch_pool),
            "dev_cer": evaluate(model, dev, tokenizer, device, knf, torch),
            "aishell_dev_cer": evaluate(model, general_dev, tokenizer, device, knf, torch),
        }
        logging.info("epoch %d %s", epoch, entry)
        summary["epochs"].append(entry)
        torch.save({"model": model.state_dict()}, args.exp_dir / f"epoch-{epoch}.pt")
        (args.exp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
