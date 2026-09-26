# 个性化训练脚本

在 WSL 中运行（环境见 [docs/WSL_TRAINING_SETUP.md](../docs/WSL_TRAINING_SETUP.md)，计划与结果见
[docs/PERSONALIZATION.md](../docs/PERSONALIZATION.md)）。数据来自 `tools/export_dataset.py` 生成的
`data/dataset/`，在 WSL 中通过 `/mnt/e/workspace/desktop-voice-orchestrator/data/dataset` 读取。

## 唤醒模型（`kws/`）

| 文件 | 作用 |
| --- | --- |
| `kws/finetune.py` | 从 `icefall-kws-zipformer-zh-en-3M-2025-12-20` 的检查点微调；个人录音 + AISHELL-1 + 伪标注的日常说话，一半样本叠加录到的音乐 |
| `kws/pseudo_label.py` | 用识别模型给背景录音里的日常说话打文字，作为“不是唤醒词”的训练样本 |
| `kws/evaluate.py` | 用程序相同的参数（阈值、加分、结尾空白、活跃路径）在 test split 上统计召回和误触发 |
| `kws/run.sh` | 微调、逐轮导出流式 ONNX（fp32 与 int8）并评估 |

```bash
bash training/kws/run.sh kws-ft4 --num-epochs 6 --repeat 4 \
  --pseudo-labels ~/dvo/data/pseudo_labels.jsonl --pseudo-repeat 5 --freeze-decoder
```

产物在 `~/dvo/exp/<实验名>/`。把选中的 `encoder-*.int8.onnx`、`decoder-*.onnx`、`joiner-*.int8.onnx`
和原模型的 `tokens.txt` 拷到 `models/` 下，再把 `kws.encoder/decoder/joiner/tokens` 指过去即可替换。

## 整句识别模型（`asr/`）

程序的整句识别（`asr.final_decoder = "sense_voice"`）用的是 sherpa-onnx 从 Fun-ASR-Nano-2512 导出的 CTC 分支：
SenseVoiceEncoderSmall → 5 层 Transformer（`ctc_decoder`）→ CTC 输出层。训练直接使用 sherpa-onnx 的 PyTorch 定义
（`~/dvo/sherpa-onnx/scripts/sense-voice/rknn/nano.py`），导出流程与上游 `export_onnx_nano.py` 相同，因此结果可以直接替换。

| 文件 | 作用 |
| --- | --- |
| `asr/finetune.py` | 默认只训练 `ctc_decoder`（与官方配方一致，编码器和输出层冻结）；个人录音 + AISHELL-1，一半叠加音乐，约 30% 拼成长句 |
| `asr/export.py` | 导出 `model.onnx` 与 QUInt8 的 `model.int8.onnx`，写入与上游相同的元数据 |
| `asr/evaluate.py` | test split 上的字错率（安静/音乐），以及日常背景里被识别出命令短语的片段数 |
| `asr/run.sh` | 微调、逐轮导出并评估 |

```bash
bash training/asr/run.sh asr-ft1 --num-epochs 3
```

替换时把 `export-epoch-N/model.int8.onnx` 和 `tokens.txt` 拷到 `models/` 下，修改 `asr.final_model` 和 `asr.final_tokens`。
二级唤醒同样使用这个模型。
