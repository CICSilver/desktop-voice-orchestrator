"""Transcribe the speaker's unscripted audio for use as KWS negatives.

A KWS model fine-tuned mostly on "wake word + command" learns to associate the
speaker's voice itself with the wake word, and then fires on their ordinary
conversation. Daily-background and unlabeled clips (train/unlabeled splits
only) contain plenty of that conversation; this labels them with the offline
recognizer so they can be trained on as ordinary speech. Clips whose
transcript contains the wake word or a near homophone are left out, since they
may be real wake-ups.

    python training/kws/pseudo_label.py --out ~/dvo/data/pseudo_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

REPO = Path("/mnt/e/workspace/desktop-voice-orchestrator")
MODEL = REPO / "models/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17"
# 小克 and its homophones / near homophones the recognizer tends to produce.
WAKE_LIKE = re.compile("小[克客课可科柯刻]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=REPO / "data/dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=4,
                        help="skip clips with fewer recognized characters (mostly music)")
    args = parser.parse_args()

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(MODEL / "model.int8.onnx"), tokens=str(MODEL / "tokens.txt"),
        language="zh", use_itn=False, num_threads=8)
    kept = skipped_wake = skipped_short = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for line in (args.dataset / "manifest.jsonl").read_text("utf-8").splitlines():
            record = json.loads(line)
            if record["split"] not in ("train", "unlabeled") or \
                    record["kind"] not in ("background", "unlabeled"):
                continue
            with wave.open(str(args.dataset / record["audio"])) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, audio.astype(np.float32) / 32768)
            recognizer.decode_stream(stream)
            text = re.sub(r"<\|[^|]*\|>", "", stream.result.text)
            text = "".join(re.findall(r"[一-鿿]", text))
            if WAKE_LIKE.search(text):
                skipped_wake += 1
                continue
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            out.write(json.dumps({"id": record["id"], "audio": record["audio"], "text": text},
                                 ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}, skipped {skipped_wake} wake-like and {skipped_short} near-empty clips")


if __name__ == "__main__":
    main()
