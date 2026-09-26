"""Score a final-decoder model on the dataset's test split.

  * character error rate on command takes, quiet and music separately;
  * how many daily-background clips come out containing a command phrase,
    which is how a recognizer biased towards commands would show up.

    python training/asr/evaluate.py --model-dir <dir with model.int8.onnx, tokens.txt>
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import sherpa_onnx

COMMAND_PHRASES = re.compile("播放音乐|打开音乐|暂停音乐|增加音量|降低音量")


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def normalize(text: str) -> str:
    return "".join(re.findall(r"[一-鿿A-Za-z0-9%]", re.sub(r"<\|[^|]*\|>", "", text)))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-file", default="model.int8.onnx")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/mnt/e/workspace/desktop-voice-orchestrator/data/dataset"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(args.model_dir / args.model_file), tokens=str(args.model_dir / "tokens.txt"),
        language="zh", use_itn=False, num_threads=8)
    records = [json.loads(line) for line in
               (args.dataset / "manifest.jsonl").read_text("utf-8").splitlines()]
    records = [r for r in records if r["split"] == "test"]

    errors = defaultdict(lambda: [0, 0])
    commands_in_background, background_clips, examples, worst = 0, 0, [], []
    for r in records:
        if r["kind"] not in ("command", "negative", "background"):
            continue
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, load(args.dataset / r["audio"]))
        recognizer.decode_stream(stream)
        text = normalize(stream.result.text)
        if r["kind"] == "background":
            background_clips += 1
            if COMMAND_PHRASES.search(text):
                commands_in_background += 1
                examples.append({"id": r["id"], "text": text})
            continue
        condition = "music" if r["music"] else "quiet"
        e = edit_distance(r["text"], text)
        errors[condition][0] += e
        errors[condition][1] += len(r["text"])
        if e:
            worst.append({"id": r["id"], "reference": r["text"], "hypothesis": text, "errors": e})

    report = {
        "model": str(args.model_dir / args.model_file),
        "cer": {k: round(v[0] / v[1], 4) for k, v in errors.items()},
        "characters": {k: v[1] for k, v in errors.items()},
        "background_clips": background_clips,
        "background_clips_with_command_phrase": commands_in_background,
        "background_examples": examples,
        "errors": sorted(worst, key=lambda x: -x["errors"]),
    }
    print(f"CER quiet {report['cer'].get('quiet')} music {report['cer'].get('music')} | "
          f"background clips with a command phrase {commands_in_background}/{background_clips}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
