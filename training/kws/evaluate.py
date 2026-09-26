"""Score a KWS model on the dataset's test split with the runtime's settings.

Recall: command takes (each contains the wake word once) with a detection.
False alarms: detections in negatives, silence and daily-background clips of
the test split, also reported per hour.

    python training/kws/evaluate.py --model-dir <dir with encoder/decoder/joiner .onnx> \
        [--suffix -epoch-13-avg-2-chunk-16-left-64] [--thresholds 0.15 0.25 0.35]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import sherpa_onnx

RATE = 16000
TOKENS = Path("/mnt/e/workspace/desktop-voice-orchestrator/models/"
              "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt")
# Must match config/keywords.txt.
KEYWORD_LINE = "x iǎo k è @小克"


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def detections(spotter, audio: np.ndarray) -> int:
    stream = spotter.create_stream()
    stream.accept_waveform(RATE, audio)
    stream.accept_waveform(RATE, np.zeros(int(0.6 * RATE), dtype=np.float32))
    stream.input_finished()
    count = 0
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        if spotter.get_result(stream):
            count += 1
            spotter.reset_stream(stream)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="", help="file name suffix, e.g. -epoch-13-avg-2-chunk-16-left-64")
    parser.add_argument("--int8", action="store_true", help="use .int8.onnx encoder and joiner")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/mnt/e/workspace/desktop-voice-orchestrator/data/dataset"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25])
    parser.add_argument("--boost", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    quant = ".int8" if args.int8 else ""
    files = {
        "encoder": args.model_dir / f"encoder{args.suffix}{quant}.onnx",
        "decoder": args.model_dir / f"decoder{args.suffix}.onnx",
        "joiner": args.model_dir / f"joiner{args.suffix}{quant}.onnx",
    }
    keywords = Path(tempfile.mkstemp(suffix=".txt")[1])
    keywords.write_text(KEYWORD_LINE + "\n", "utf-8")

    records = [json.loads(line) for line in
               (args.dataset / "manifest.jsonl").read_text("utf-8").splitlines()]
    records = [r for r in records if r["split"] == "test"]
    audio = {r["id"]: load(args.dataset / r["audio"]) for r in records}

    report = {"model": {k: str(v) for k, v in files.items()}, "thresholds": {}}
    for threshold in args.thresholds:
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(TOKENS), encoder=str(files["encoder"]), decoder=str(files["decoder"]),
            joiner=str(files["joiner"]), num_threads=4, max_active_paths=4,
            keywords_file=str(keywords), keywords_score=args.boost,
            keywords_threshold=threshold, num_trailing_blanks=1, provider="cpu")
        recall = defaultdict(lambda: [0, 0])
        false_alarms, negative_seconds = 0, 0.0
        missed, alarms = [], []
        for r in records:
            n = detections(spotter, audio[r["id"]])
            condition = "music" if r["music"] else "quiet"
            if r["kind"] == "command":
                recall[condition][0] += int(n > 0)
                recall[condition][1] += 1
                if n == 0:
                    missed.append(r["id"])
            elif r["kind"] in ("negative", "silence", "background"):
                false_alarms += n
                negative_seconds += r["duration_s"]
                if n:
                    alarms.append({"id": r["id"], "kind": r["kind"], "detections": n})
        hours = negative_seconds / 3600
        entry = {
            "recall": {k: f"{v[0]}/{v[1]}" for k, v in recall.items()},
            "false_alarms": false_alarms,
            "negative_minutes": round(negative_seconds / 60, 1),
            "false_alarms_per_hour": round(false_alarms / hours, 2) if hours else None,
            "missed": missed,
            "false_alarm_clips": alarms,
        }
        report["thresholds"][str(threshold)] = entry
        print(f"threshold {threshold}: recall quiet {entry['recall'].get('quiet')} "
              f"music {entry['recall'].get('music')} | false alarms {false_alarms} in "
              f"{entry['negative_minutes']} min ({entry['false_alarms_per_hour']}/h)")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    keywords.unlink()


if __name__ == "__main__":
    main()
