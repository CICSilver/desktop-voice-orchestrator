"""Fine-tune the runtime's final decoder (Fun-ASR-Nano CTC branch) on the speaker.

The runtime runs sherpa-onnx's export of Fun-ASR-Nano-2512's CTC branch:
SenseVoiceEncoderSmall -> 5-layer Transformer ("ctc_decoder") -> CTC output
layer. This trains that branch with sherpa-onnx's own PyTorch definition
(scripts/sense-voice/rknn/nano.py), so training/asr/export.py produces a
drop-in replacement for models/sherpa-onnx-sense-voice-funasr-nano-*.

Training data: the speaker's transcribed takes (commands and read sentences)
from tools/export_dataset.py, plus a slice of AISHELL-1 so the model keeps
recognizing ordinary Mandarin. Half of the utterances get the speaker's
recorded music mixed in at 0-15 dB SNR, and some are concatenated into
longer utterances (up to 15 s) with short pauses.

By default only the ctc_decoder Transformer trains, as in the official
Fun-ASR-Nano recipe; the encoder and the output layer (which holds the
vocabulary prior) stay frozen. Utterances are processed one at a time, since
the exported model has no padding mask, and gradients are accumulated.

Runs in WSL, conda env `funasr` (see docs/WSL_TRAINING_SETUP.md):

    python training/asr/finetune.py --exp-dir ~/dvo/exp/asr-ft1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import wave
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import torch

HOME = Path.home()
sys.path.insert(0, str(HOME / "dvo/sherpa-onnx/scripts/sense-voice/rknn"))

import nano  # noqa: E402
from funasr.models.sense_voice.whisper_lib.tokenizer import get_tokenizer  # noqa: E402

RATE = 16000
SPEECH_KINDS = {"command", "wake", "bare_command", "negative", "read"}
NOISE_KINDS = {"background", "silence", "unlabeled"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("/mnt/e/workspace/desktop-voice-orchestrator/data/dataset"))
    parser.add_argument("--pretrained", type=Path, default=HOME / "dvo/models/Fun-ASR-Nano-2512")
    parser.add_argument("--aishell", type=Path, default=HOME / "dvo/corpora")
    parser.add_argument("--aishell-utterances", type=int, default=3000)
    parser.add_argument("--repeat", type=int, default=3,
                        help="how often each personal utterance appears per epoch")
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


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768


def features(samples: np.ndarray) -> torch.Tensor:
    """Exactly sherpa-onnx's front end for this model: 80-bin fbank, LFR 7/6."""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = RATE
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(RATE, (samples * 32768).tolist())
    fbank.input_finished()
    frames = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    count = (frames.shape[0] - 7) // 6 + 1
    lfr = np.stack([frames[t * 6:t * 6 + 7].reshape(-1) for t in range(count)])
    return torch.from_numpy(lfr.astype(np.float32))


class Tokenizer:
    def __init__(self, pretrained: Path, sherpa_tokens: Path):
        self.encoder = get_tokenizer(multilingual=True, num_languages=8749,
                                     vocab_path=str(pretrained / "multilingual.tiktoken"))
        self.blank = sum(1 for _ in open(sherpa_tokens, encoding="utf-8")) - 1
        self.pieces = {}
        import base64
        for line in open(sherpa_tokens, encoding="utf-8"):
            parts = line.strip().split()
            self.pieces[int(parts[-1])] = base64.b64decode(parts[0]) if len(parts) == 2 else b" "

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.pieces[i] for i in ids if i != self.blank).decode("utf-8", "replace")


def load_personal(dataset: Path, split: str) -> list[tuple[np.ndarray, str]]:
    items = []
    for line in (dataset / "manifest.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        if record["split"] == split and record["kind"] in SPEECH_KINDS and record["text"]:
            items.append((read_wav(dataset / record["audio"]), record["text"]))
    return items


def load_noise(dataset: Path) -> list[np.ndarray]:
    noise = []
    for line in (dataset / "manifest.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        if record["split"] in ("train", "unlabeled") and record["kind"] in NOISE_KINDS \
                and record["music"]:
            noise.append(read_wav(dataset / record["audio"]))
    return noise


def load_aishell(root: Path, count: int, seed: int, skip: int = 0) -> list[tuple[np.ndarray, str]]:
    transcript = {}
    for line in (root / "data_aishell/transcript/aishell_transcript_v0.8.txt").read_text(
            "utf-8").splitlines():
        key, _, text = line.partition(" ")
        transcript[key] = text.replace(" ", "")
    wavs = [w for w in sorted((root / "aishell_wav").rglob("*.wav")) if w.stem in transcript]
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
    return mixed / peak * 0.95 if peak > 0.99 else mixed


def epoch_items(pool, args, rng: random.Random):
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


@torch.no_grad()
def evaluate(model, items, tokenizer, device) -> float:
    was_training = model.training
    model.eval()
    errors = total = 0
    for audio, text in items:
        logits = model(features(audio)[None].to(device))[0]
        ids = torch.unique_consecutive(logits.argmax(-1)).tolist()
        e, n = cer(text, tokenizer.decode(ids).strip())
        errors += e
        total += n
    model.train(was_training)
    return errors / max(1, total)


def set_trainable(model, args) -> list[dict]:
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
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    sherpa_tokens = Path("/mnt/e/workspace/desktop-voice-orchestrator/models/"
                         "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17/tokens.txt")
    tokenizer = Tokenizer(args.pretrained, sherpa_tokens)
    model = nano.Nano()
    state = torch.load(args.pretrained / "model.pt", map_location="cpu", weights_only=False)
    state = {k: v for k, v in state.items() if not k.startswith(("llm", "audio_adaptor"))}
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    groups = set_trainable(model, args)
    trainable = sum(p.numel() for g in groups for p in g["params"])
    logging.info("training %.1fM of %.1fM parameters", trainable / 1e6,
                 sum(p.numel() for p in model.parameters()) / 1e6)

    personal = load_personal(args.dataset, "train")
    dev = load_personal(args.dataset, "dev")
    general = load_aishell(args.aishell, args.aishell_utterances, args.seed)
    general_dev = load_aishell(args.aishell, 200, args.seed, skip=args.aishell_utterances)
    noise = load_noise(args.dataset)
    logging.info("personal %d (%.1f min) x%d, dev %d, aishell %d (+%d dev), noise clips %d",
                 len(personal), sum(len(a) for a, _ in personal) / RATE / 60, args.repeat,
                 len(dev), len(general), len(general_dev), len(noise))
    pool = personal * args.repeat + general

    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = len(pool) // args.accum
    total_steps = steps_per_epoch * args.num_epochs
    warmup = min(100, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) *
        max(0.1, 1 - s / max(1, total_steps)))

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "epochs": []}
    summary["pretrained"] = {"dev_cer": evaluate(model, dev, tokenizer, device),
                             "aishell_dev_cer": evaluate(model, general_dev, tokenizer, device)}
    logging.info("pretrained %s", summary["pretrained"])
    step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if args.unfreeze_encoder_layers == 0:
            model.audio_encoder.eval()  # frozen: no dropout
        running, count = 0.0, 0
        optimizer.zero_grad()
        for audio, text in epoch_items(pool, args, rng):
            if rng.random() < args.mix_probability:
                audio = mix(audio, noise, tuple(args.snr), rng)
            x = features(audio)[None].to(device)
            targets = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            log_probs = model(x).log_softmax(-1).transpose(0, 1)  # (T, 1, V)
            if log_probs.size(0) < len(targets):
                continue
            loss = torch.nn.functional.ctc_loss(
                log_probs, targets[None].to(device),
                input_lengths=torch.tensor([log_probs.size(0)]),
                target_lengths=torch.tensor([len(targets)]),
                blank=tokenizer.blank, reduction="mean", zero_infinity=True)
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
        entry = {"epoch": epoch, "steps": step,
                 "dev_cer": evaluate(model, dev, tokenizer, device),
                 "aishell_dev_cer": evaluate(model, general_dev, tokenizer, device)}
        logging.info("epoch %d %s", epoch, entry)
        summary["epochs"].append(entry)
        torch.save({"model": model.state_dict()}, args.exp_dir / f"epoch-{epoch}.pt")
        (args.exp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
