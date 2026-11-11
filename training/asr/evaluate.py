"""Score a final-decoder model on the dataset's test split and M3 new-command benchmark.

Reports:
  * character error rate (CER) on command takes, quiet and music separately;
  * optional CER & exact-phrase match rate on `new_commands_quiet` clips (`--new-commands-manifest`
    or `--new-commands-dir`) and `dev_read` general Mandarin sentences;
  * how many daily-background (or game stress-test) clips come out containing a command phrase,
    guarding against command-hallucination bias.

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

COMMAND_PHRASES = re.compile(
    "播放音乐|打开音乐|暂停音乐|增加音量|降低音量|调大音量|调小音量|"
    "切歌|换歌|换一首|下一首|每日推荐|顺序播放|列表循环|单曲循环|随机播放|"
    "打开桌面歌词|关闭桌面歌词|收藏这首歌|喜欢这首歌|收藏一下"
)


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]
        return audio.astype(np.float32) / 32768.0


def normalize(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]", re.sub(r"<\|[^|]*\|>", "", text)))


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
    parser.add_argument("--extra-test-manifests", type=Path, nargs="*", default=[],
                        help="optional extra manifests with split='test' (e.g. mined game negatives)")
    parser.add_argument("--new-commands-manifest", type=Path,
                        help="optional JSONL of new_commands_quiet items ({id, wav_path/audio, text})")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import sherpa_onnx

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(args.model_dir / args.model_file),
        tokens=str(args.model_dir / "tokens.txt"),
        language="zh",
        use_itn=False,
        num_threads=4,
    )

    def decode_wav(wav_path: Path) -> str:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, load(wav_path))
        recognizer.decode_stream(stream)
        return normalize(stream.result.text)

    records = []
    manifest_file = args.dataset / "manifest.jsonl"
    if manifest_file.is_file():
        for line in manifest_file.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_root"] = args.dataset
                records.append(r)

    for ext in args.extra_test_manifests:
        if ext.is_file():
            for line in ext.read_text("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_root"] = ext.parent
                    records.append(r)

    test_records = [r for r in records if r.get("split") == "test"]
    dev_read_records = [r for r in records if r.get("split") == "dev" and r.get("kind") == "read"]

    errors = defaultdict(lambda: [0, 0])
    commands_in_background, background_clips, examples, worst = 0, 0, [], []

    for r in test_records:
        if r.get("kind") not in ("command", "negative", "background"):
            continue
        wav_path = Path(r["audio"]) if Path(r["audio"]).is_absolute() else (r["_root"] / r["audio"])
        if not wav_path.is_file():
            continue
        text = decode_wav(wav_path)
        if r["kind"] == "background":
            background_clips += 1
            if COMMAND_PHRASES.search(text):
                commands_in_background += 1
                examples.append({"id": r["id"], "text": text})
            continue
        condition = "music" if r.get("music") else "quiet"
        e = edit_distance(r["text"], text)
        errors[condition][0] += e
        errors[condition][1] += len(r["text"])
        if e:
            worst.append({"id": r["id"], "reference": r["text"], "hypothesis": text, "errors": e})

    for r in dev_read_records:
        wav_path = Path(r["audio"]) if Path(r["audio"]).is_absolute() else (r["_root"] / r["audio"])
        if not wav_path.is_file():
            continue
        text = decode_wav(wav_path)
        e = edit_distance(r["text"], text)
        errors["dev_read"][0] += e
        errors["dev_read"][1] += len(r["text"])

    new_cmd_exact = None
    new_cmd_total = 0
    if args.new_commands_manifest and args.new_commands_manifest.is_file():
        new_cmd_exact = 0
        base = args.new_commands_manifest.parent
        for line in args.new_commands_manifest.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rel = item.get("audio") or item.get("wav_path")
            wp = Path(rel) if Path(rel).is_absolute() else (base / rel)
            if not wp.is_file():
                continue
            ref = normalize(item.get("text") or item.get("prompt", ""))
            hyp = decode_wav(wp)
            e = edit_distance(ref, hyp)
            errors["new_commands_quiet"][0] += e
            errors["new_commands_quiet"][1] += len(ref)
            new_cmd_total += 1
            if e == 0:
                new_cmd_exact += 1
            else:
                worst.append({"id": item["id"], "reference": ref, "hypothesis": hyp, "errors": e})

    report = {
        "model": str(args.model_dir / args.model_file),
        "cer": {k: round(v[0] / max(1, v[1]), 4) for k, v in errors.items()},
        "characters": {k: v[1] for k, v in errors.items()},
        "new_commands_exact_match": (
            f"{new_cmd_exact}/{new_cmd_total}" if new_cmd_exact is not None else None
        ),
        "background_clips": background_clips,
        "background_clips_with_command_phrase": commands_in_background,
        "background_examples": examples,
        "errors": sorted(worst, key=lambda x: -x["errors"]),
    }
    msg = (
        f"CER quiet={report['cer'].get('quiet')} music={report['cer'].get('music')} "
        f"new_cmd={report['cer'].get('new_commands_quiet')} dev_read={report['cer'].get('dev_read')} | "
        f"background clips with command phrase: {commands_in_background}/{background_clips}"
    )
    print(msg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
