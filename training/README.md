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
| `asr/finetune.py` | 默认只训练 `ctc_decoder`（支持 `--unfreeze-encoder-layers` 解冻编码器尾层）；支持 `--extra-manifests` 与 `--tts-max-per-epoch` 多源均衡采样，一半叠加音乐，约 30% 拼成长句 |
| `asr/synthesize_personal_tts.py` | 基于 `CosyVoice2-0.5B` 零样本克隆个人音色 + 原始 `SenseVoice` 独立裁判闭环防失真质检（拦截错字、语速异常与截断坏样本） |
| `asr/export.py` | 导出 `model.onnx` 与 QUInt8 的 `model.int8.onnx`，写入与上游相同的元数据（支持 `--sherpa-nano-dir` / `--tokens`） |
| `asr/evaluate.py` | test split 上的字错率（安静/音乐）、新命令子集准确率，以及日常背景里被识别出命令短语的片段数 |
| `asr/run.sh` | 微调、逐轮导出并评估（WSL） |
| `colab_telemetry_hook.py` | 纯 Python 标准库 Colab 监控与远程抢救 Hook：原子同步 `status.json`/`events.jsonl`/`tail.log` 到云盘，支持本地无感等待与原地热修复，任务完结自动断开 A100 |
| `colab_m3_pipeline.ipynb` | Google Colab（L4 / A100）一键运行流水线：试听 25 条克隆小样 → 可选游戏录音双模型打标 → 3,600 条扩词造数 → 微调并导出 `asr-personal-v2` |

```bash
bash training/asr/run.sh asr-ft1 --num-epochs 3
```

替换时把 `export-epoch-N/model.int8.onnx` 和 `tokens.txt` 拷到 `models/` 下，修改 `asr.final_model` 和 `asr.final_tokens`。
二级唤醒同样使用这个模型。

## 零 Token 监控与远程抢救机制 (Telemetry & Remote Rescue)

在 Colab A100 上运行 `training/colab_m3_pipeline.ipynb` 或 `training/router/colab_m4_router_a100.ipynb` 时，内置的 `ColabJobHook` 将全程负责训练状态原子同步与故障抢救：

1. **零 Token 阻塞监控**：
   本地通过 `python tools/watch_colab_job.py --watch --job m3_asr`（或 `m4_router`）利用本地阻塞脚本等待，不消耗任何 Agent Token。
2. **状态契约 (`colab_runs/<job_name>/`)**：
   - `status.json`：原子写入当前阶段、进度（step、epoch、loss、ETA）、显存与心跳序号。
   - `events.jsonl`：流式记录关键生命周期事件（阶段开始/完成、收到命令、任务成功/失败、释放算力）。
   - `tail.log`：滚动保存最近 120 行日志，出差错时本地 watcher 可立即输出原始日志片段。
3. **原地热修复 (`waiting_for_fix`)**：
   当某阶段发生异常时，Hook 进入 `waiting_for_fix` 状态并维持心跳。本地修复代码后通过 `python tools/watch_colab_job.py --job <job_name> --send-command sync_and_retry` 即可将热补丁推送到云盘 `repo_sync/`，Colab 端会自动同步并就地重试该阶段（无需重启 Runtime 或重新加载基础模型）。
4. **算力自动释放保护**：
   训练成功（或抢救超时退出）后，Hook 自动调用 `google.colab.runtime.unassign()` 释放 A100 实例，防止闲置产生额外计算单元消耗。

## M3 个人音色克隆扩词与 Colab 训练工具链

当本机未启用 WSL 或需要批量扩充新命令、歌手/歌名与自然语言口语句式时，按以下步骤配合 Google Drive 与 Colab 运行：

1. **筛选本人黄金参考音频池**（自动裁除尾部空格键瞬态咔哒声，仅保留高信噪比安静原声，不做频谱降噪）：
   ```powershell
   python tools/prepare_voice_prompts.py --manifest data/dataset/manifest.jsonl `
     --out-dir "H:\我的云端硬盘\desktop-voice-orchestrator-data\m3_tts_prep\prompt_pool"
   ```
2. **生成防失真 TTS 扩词清单**（自动读取本机网易云 `webdb.dat` 华语曲库并转换全部数字/`%` 为纯中文，单句严格限制 4~20 字）：
   ```powershell
   python tools/build_nl_tts_prompts.py `
     --out-dir "H:\我的云端硬盘\desktop-voice-orchestrator-data\m3_tts_prep"
   ```
3. **（可选）清洗戴耳机游戏背景录音**（只读 `mic.wav`，遵守 `discarded` 区间，Silero VAD 切分 + Colab 双异构模型 `SenseVoice` & `faster-whisper large-v3` 100% 逐字一致交叉打标）：
   ```powershell
   python tools/mine_game_sessions.py --sessions data/sessions --out data/game_mined
   ```
4. **Colab 一键合成、微调与导出**：在 Google Colab 打开 `training/colab_m3_pipeline.ipynb`，依次运行 Cell 1~6，先在线试听 25 条 Pilot 小样确认音色自然无电音，再批量合成并导出 `models/asr-personal-v2/model.int8.onnx`。
