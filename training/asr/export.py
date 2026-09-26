"""Export a fine-tuned Fun-ASR-Nano CTC checkpoint for the runtime.

Mirrors sherpa-onnx's scripts/sense-voice/export_onnx_nano.py (same input,
opset, metadata and QUInt8 quantization), so the output directory replaces
models/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17:

    python training/asr/export.py --checkpoint ~/dvo/exp/asr-ft1/epoch-3.pt \
        --out ~/dvo/exp/asr-ft1/export-epoch-3
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

HOME = Path.home()
sys.path.insert(0, str(HOME / "dvo/sherpa-onnx/scripts/sense-voice/rknn"))

import nano  # noqa: E402

TOKENS = Path("/mnt/e/workspace/desktop-voice-orchestrator/models/"
              "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17/tokens.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comment", default="Fun-ASR-Nano-2512 CTC branch, personal fine-tune")
    args = parser.parse_args()

    model = nano.Nano()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = args.out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, torch.randn(1, 30, 560), str(fp32), opset_version=13,
                          input_names=["x"], output_names=["logits"], dynamic_axes={"x": {1: "T"}},
                          dynamo=False)
    vocab_size = sum(1 for _ in open(TOKENS, encoding="utf-8"))
    meta = {
        "lfr_window_size": 7, "lfr_window_shift": 6, "normalize_samples": 0,
        "model_type": "sense_voice_ctc", "version": "1", "model_author": "FunAudioLLM",
        "maintainer": "k2-fsa", "vocab_size": vocab_size, "blank_id": vocab_size - 1,
        "comment": args.comment, "url": "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
    }
    exported = onnx.load(str(fp32))
    while len(exported.metadata_props):
        exported.metadata_props.pop()
    for key, value in meta.items():
        prop = exported.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(exported, str(fp32))
    # QUInt8, as upstream: onnxruntime's C++ API gives wrong results with QInt8 here.
    quantize_dynamic(model_input=str(fp32), model_output=str(args.out / "model.int8.onnx"),
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8)
    shutil.copy(TOKENS, args.out / "tokens.txt")
    print("exported", args.out)


if __name__ == "__main__":
    main()
