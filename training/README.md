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
