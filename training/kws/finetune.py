"""Fine-tune the streaming zipformer KWS model on the speaker's own recordings.

Starts from icefall-kws-zipformer-zh-en-3M-2025-12-20 (the model the runtime
ships) and trains on:

  * the speaker's transcribed takes from tools/export_dataset.py (train split),
    repeated so they carry weight next to the general data;
  * a slice of AISHELL-1, so the model keeps recognizing ordinary Mandarin;
  * both mixed with the speaker's recorded music (background and silence clips
    of the train split) at 0-15 dB SNR for half of the utterances.

Runs in WSL, conda env `kws`, with icefall on PYTHONPATH (see
docs/WSL_TRAINING_SETUP.md). Checkpoints are written in icefall's format, so
icefall's zipformer/export-onnx-streaming.py exports them unchanged.

    python training/kws/finetune.py --exp-dir ~/dvo/exp/kws-ft1 --num-epochs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cppinyin
import k2
import torch
from lhotse import CutSet, Fbank, FbankConfig, MonoCut, Recording, SupervisionSegment
from lhotse.dataset import (CutMix, DynamicBucketingSampler, K2SpeechRecognitionDataset,
                            SimpleCutSampler, SpecAugment)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

HOME = Path.home()
RECIPE = HOME / "dvo/icefall/egs/librispeech/ASR/zipformer"
sys.path.insert(0, str(RECIPE))

from optim import Eden, ScaledAdam  # noqa: E402
from train import get_model  # noqa: E402

from icefall.utils import AttributeDict, get_parameter_groups_with_lrs  # noqa: E402

# Utterances the speaker read from prompts; silence/background carry no text.
SPEECH_KINDS = {"command", "wake", "bare_command", "negative", "read"}
NOISE_KINDS = {"background", "silence", "unlabeled"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("/mnt/e/workspace/desktop-voice-orchestrator/data/dataset"))
    parser.add_argument("--pretrained", type=Path,
                        default=HOME / "dvo/models/icefall-kws-zipformer-zh-en-3M-2025-12-20")
    parser.add_argument("--aishell", type=Path, default=HOME / "dvo/corpora")
    parser.add_argument("--aishell-utterances", type=int, default=4000,
                        help="AISHELL utterances per epoch (0 disables)")
    parser.add_argument("--repeat", type=int, default=12,
                        help="how often each personal utterance appears per epoch")
    parser.add_argument("--pseudo-labels", type=Path,
                        help="training/kws/pseudo_label.py output: the speaker's own "
                             "conversation, trained on as ordinary speech")
    parser.add_argument("--pseudo-repeat", type=int, default=2)
    parser.add_argument("--freeze-decoder", action="store_true",
                        help="train only the acoustic encoder; the prediction network and "
                             "joiner keep the pretrained token priors")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.0005)
    parser.add_argument("--max-duration", type=float, default=300.0,
                        help="seconds of audio per batch")
    parser.add_argument("--mix-probability", type=float, default=0.5)
    parser.add_argument("--snr", type=float, nargs=2, default=(0.0, 15.0))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_params(pretrained: Path) -> AttributeDict:
    """Model hyper-parameters exactly as the checkpoint was trained with."""
    checkpoint = torch.load(pretrained / "checkpoint/epoch-13.pt", map_location="cpu",
                            weights_only=False)
    skip = {"model", "optimizer", "scheduler", "grad_scaler", "sampler", "model_avg"}
    return AttributeDict({k: v for k, v in checkpoint.items() if k not in skip})


class Tokenizer:
    def __init__(self, pretrained: Path):
        lang = pretrained / "data/lang_phone"
        self.encoder = cppinyin.Encoder(str(lang / "g2p.dict"))
        self.ids = {}
        for line in (lang / "tokens.txt").read_text("utf-8").splitlines():
            token, index = line.rsplit(" ", 1)
            self.ids[token] = int(index)

    def encode(self, text: str) -> list[int] | None:
        tokens = self.encoder.encode(text, tone="normal", partial=True)
        if not tokens or any(token not in self.ids for token in tokens):
            return None
        return [self.ids[token] for token in tokens]


def cut_from_file(cut_id: str, path: Path, text: str | None) -> MonoCut:
    recording = Recording.from_file(path, recording_id=cut_id)
    supervisions = []
    if text is not None:
        supervisions.append(SupervisionSegment(id=cut_id, recording_id=cut_id, start=0.0,
                                               duration=recording.duration, text=text))
    return MonoCut(id=cut_id, start=0.0, duration=recording.duration, channel=0,
                   recording=recording, supervisions=supervisions)


def personal_cuts(dataset: Path, split: str, tokenizer: Tokenizer) -> CutSet:
    cuts = []
    for line in (dataset / "manifest.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        if record["split"] != split or record["kind"] not in SPEECH_KINDS or not record["text"]:
            continue
        if tokenizer.encode(record["text"]) is None:
            logging.warning("skipping %s: text %s has no token mapping", record["id"], record["text"])
            continue
        cuts.append(cut_from_file(record["id"], dataset / record["audio"], record["text"]))
    return CutSet.from_cuts(cuts)


def pseudo_cuts(dataset: Path, labels: Path | None, tokenizer: Tokenizer) -> CutSet:
    cuts = []
    if labels is not None:
        for line in labels.read_text("utf-8").splitlines():
            record = json.loads(line)
            if tokenizer.encode(record["text"]) is not None:
                cuts.append(cut_from_file(f"pseudo/{record['id']}", dataset / record["audio"],
                                          record["text"]))
    return CutSet.from_cuts(cuts)


def noise_cuts(dataset: Path) -> CutSet:
    """Music-only audio from the train and unlabeled splits, in 4 s windows."""
    cuts = []
    for line in (dataset / "manifest.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        if record["split"] in ("train", "unlabeled") and record["kind"] in NOISE_KINDS \
                and record["music"]:
            cuts.append(cut_from_file(record["id"], dataset / record["audio"], None))
    windows = CutSet.from_cuts(cuts).cut_into_windows(4.0)
    return CutSet.from_cuts(c for c in windows if c.duration >= 2.0)


def aishell_cuts(root: Path, limit: int, tokenizer: Tokenizer, seed: int) -> CutSet:
    if limit <= 0:
        return CutSet.from_cuts([])
    transcript = {}
    for line in (root / "data_aishell/transcript/aishell_transcript_v0.8.txt").read_text(
            "utf-8").splitlines():
        key, _, text = line.partition(" ")
        transcript[key] = text.replace(" ", "")
    wavs = sorted((root / "aishell_wav").rglob("*.wav"))
    random.Random(seed).shuffle(wavs)
    cuts = []
    for wav in wavs:
        text = transcript.get(wav.stem)
        if text and tokenizer.encode(text) is not None:
            cuts.append(cut_from_file(f"aishell/{wav.stem}", wav, text))
        if len(cuts) >= limit:
            break
    return CutSet.from_cuts(cuts)


def make_loader(cuts: CutSet, noise: CutSet | None, args, train: bool) -> DataLoader:
    transforms = []
    if train and noise is not None and len(noise) > 0:
        transforms.append(CutMix(cuts=noise, snr=tuple(args.snr), p=args.mix_probability,
                                 preserve_id=True, random_mix_offset=True, seed=args.seed))
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        cut_transforms=transforms,
        input_transforms=[SpecAugment(time_warp_factor=80)] if train else None,
        return_cuts=True,
    )
    if train:
        sampler = DynamicBucketingSampler(cuts, max_duration=args.max_duration, shuffle=True,
                                          num_buckets=8, seed=args.seed)
    else:
        sampler = SimpleCutSampler(cuts, max_duration=args.max_duration, shuffle=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)


def set_batch_count(model: torch.nn.Module, count: float) -> None:
    # Zipformer schedules dropout and balancers on this; the pretrained model
    # is far past every schedule, and fine-tuning keeps it there.
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = count


def compute_loss(model, batch, tokenizer: Tokenizer, params, device, train: bool):
    features = batch["inputs"].to(device)
    lengths = batch["supervisions"]["num_frames"].to(device)
    y = k2.RaggedTensor([tokenizer.encode(text) for text in batch["supervisions"]["text"]])
    with torch.set_grad_enabled(train):
        simple, pruned, ctc, _, _ = model(x=features, x_lens=lengths, y=y,
                                          prune_range=params.prune_range,
                                          am_scale=params.am_scale, lm_scale=params.lm_scale)
        loss = params.simple_loss_scale * simple + pruned + params.ctc_loss_scale * ctc
    frames = int((lengths // params.subsampling_factor).sum().item())
    return loss, frames


def evaluate(model, loader, tokenizer, params, device) -> float:
    model.eval()
    total, frames = 0.0, 0
    for batch in loader:
        loss, n = compute_loss(model, batch, tokenizer, params, device, train=False)
        total += loss.item()
        frames += n
    model.train()
    return total / max(1, frames)


def main() -> None:
    args = get_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    params = load_params(args.pretrained)
    tokenizer = Tokenizer(args.pretrained)
    model = get_model(params)
    state = torch.load(args.pretrained / "checkpoint/pretrained-epoch-13-avg-2.pt",
                       map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    set_batch_count(model, 1.0e6)

    personal = personal_cuts(args.dataset, "train", tokenizer)
    dev = personal_cuts(args.dataset, "dev", tokenizer)
    general = aishell_cuts(args.aishell, args.aishell_utterances, tokenizer, args.seed)
    conversation = pseudo_cuts(args.dataset, args.pseudo_labels, tokenizer)
    noise = noise_cuts(args.dataset)
    logging.info("personal %d cuts (%.1f min) x%d, conversation %d x%d, dev %d, aishell %d, "
                 "noise windows %d", len(personal), sum(c.duration for c in personal) / 60,
                 args.repeat, len(conversation), args.pseudo_repeat, len(dev), len(general),
                 len(noise))
    train_cuts = personal.repeat(times=args.repeat) + general
    if len(conversation):
        train_cuts = train_cuts + conversation.repeat(times=args.pseudo_repeat)
    train_cuts = train_cuts.shuffle(random.Random(args.seed))
    train_loader = make_loader(train_cuts, noise, args, train=True)
    dev_loader = make_loader(dev + general.subset(first=200), None, args, train=False) \
        if len(dev) else None

    if args.freeze_decoder:
        # Fine-tuning mostly on "wake word + command" otherwise raises the
        # prediction network's prior for the wake word's tokens everywhere.
        for name, parameter in model.named_parameters():
            if name.startswith(("decoder.", "joiner.", "simple_lm_proj.")):
                parameter.requires_grad_(False)
    groups = get_parameter_groups_with_lrs(model, lr=args.base_lr, include_names=True)
    for group in groups:
        group["named_params"] = [(n, p) for n, p in group["named_params"] if p.requires_grad]
    groups = [group for group in groups if group["named_params"]]
    logging.info("training %d of %d parameters",
                 sum(p.numel() for p in model.parameters() if p.requires_grad),
                 sum(p.numel() for p in model.parameters()))
    optimizer = ScaledAdam(groups, lr=args.base_lr, clipping_scale=2.0)
    # A near-constant learning rate: lr_batches and lr_epochs far beyond the run.
    scheduler = Eden(optimizer, lr_batches=100000, lr_epochs=100, warmup_batches=100,
                     warmup_start=0.5)

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "epochs": []}
    if dev_loader is not None:
        summary["pretrained_dev_loss"] = evaluate(model, dev_loader, tokenizer, params, device)
        logging.info("pretrained dev loss %.4f", summary["pretrained_dev_loss"])
    step = 0
    for epoch in range(1, args.num_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        scheduler.step_epoch(epoch - 1)
        running, running_frames = 0.0, 0
        for batch in train_loader:
            loss, frames = compute_loss(model, batch, tokenizer, params, device, train=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step_batch(step)
            step += 1
            set_batch_count(model, 1.0e6 + step)
            running += loss.item()
            running_frames += frames
            if step % 50 == 0:
                logging.info("epoch %d step %d loss %.4f lr %.2e", epoch, step,
                             running / max(1, running_frames), scheduler.get_last_lr()[0])
                running, running_frames = 0.0, 0
        entry = {"epoch": epoch, "steps": step}
        if dev_loader is not None:
            entry["dev_loss"] = evaluate(model, dev_loader, tokenizer, params, device)
            logging.info("epoch %d dev loss %.4f", epoch, entry["dev_loss"])
        torch.save({"model": model.state_dict()}, args.exp_dir / f"epoch-{epoch}.pt")
        summary["epochs"].append(entry)
        (args.exp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
