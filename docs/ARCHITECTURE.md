# 系统架构

## 1. 关键约束

后置唤醒词改变了传统“先唤醒、再录命令”的状态机。系统必须在未确认唤醒时就保留最近一段语音，否则在听到句尾唤醒词后无法恢复前面的命令。为兼顾隐私和资源占用，系统只把经过 AEC 的 PCM 放在固定容量的内存环形缓冲区中；未触发的数据被自动覆盖，完整 ASR 只在 KWS 命中后运行。

USB 音箱由电脑直接输出，因此可以用对应 render endpoint 的 WASAPI Loopback 得到参考信号。这个条件适合先做软件 AEC，不要求一开始增加 XVF3800 一类硬件音频前端。

## 2. 唤醒词模型与录音边界

主检测器采用 sherpa-onnx 开放词表 KWS。用户提供唤醒短语的文字、拼音或音素表示，配置层生成关键词 token，并为每个短语保存 `boosting score` 和触发阈值。MVP 优先验证当前中英 Zipformer KWS 模型；模型命中结果中的 token 时间戳用于判断唤醒词位于话段开头还是结尾。

配置流程仍提供录音步骤，但录音只承担以下职责：

- 验证短语在 0.5 米、1 米、2 米以及音乐播放条件下是否容易检出
- 调整关键词分数和触发阈值
- 建立与校准集分离的回归测试数据

录音不用于生成声纹或限制身份，运行链路不包含说话人验证。任何说话人正确说出配置短语并达到阈值，都可以唤醒系统。Query-by-Example 只作为无法用文字、拼音或音素稳定描述特殊发音时的后续候选。

## 3. 进程与线程

```text
Audio service (native, real-time priority where appropriate)
  ├─ microphone capture thread
  ├─ render loopback capture thread
  ├─ alignment / drift correction
  ├─ WebRTC AEC3 processing thread
  └─ 16 kHz PCM ring buffer + KWS/VAD feed

Orchestrator service
  ├─ trigger coordinator
  ├─ utterance extractor
  ├─ ASR worker
  ├─ command planner
  ├─ ordered executor
  └─ event / audit log

Desktop shell
  ├─ tray status
  ├─ wake-word enrollment
  ├─ aliases and permissions
  └─ diagnostics and benchmark runner
```

音频实时线程不执行模型加载、磁盘写入、网络请求或桌面操作。它只负责稳定地产生带单调时间戳的 PCM 帧，并通过有界队列把事件交给非实时线程。

## 4. 音频时序

- 内部处理帧建议固定为 10 ms
- AEC 输入保持相同采样率、帧长和单调时间基准
- loopback 参考信号先进入反向流处理，麦克风信号随后进入正向流处理
- 每帧保留设备位置、QPC 时间戳、预计流延迟和丢帧标记
- 当麦克风与 render endpoint 使用不同硬件时钟时，持续估计漂移并做小幅重采样补偿
- AEC 后再下采样到 KWS/ASR 所需的 16 kHz 单声道，避免把回声写进后置唤醒环形缓冲区

建议环形缓冲区保存 20 秒音频，触发后向前搜索最近一次满足静音阈值的 VAD 边界。若没有可靠边界，则按最大命令窗口截取，避免无限向前扩展。

## 5. 双位置唤醒算法

KWS 命中时产生 `{start_ms, end_ms, score}`。协调器同时查看命中前后的 VAD 状态：

### 前置唤醒

1. 唤醒词前方是静音或缓冲区边界
2. 从唤醒词结束后等待语音段完成，允许短暂停顿连接多条命令
3. 保留包含唤醒词的连续整句语音送入 ASR，在文本层剥离唤醒词

### 后置唤醒

1. 唤醒词前方存在连续语音，唤醒词后方进入静音
2. 从唤醒词开始位置向前回溯到最近的可靠 VAD 起点
3. 保留包含唤醒词的连续整句语音送入 ASR，在文本层剥离唤醒词

### 歧义处理

若唤醒词位于话段中间，系统保留原始连续话语作为一个命令序列，并记录 `wake_position=embedded`。若 KWS 分数过低、语音窗口超长或边界不可靠，则拒绝执行并给出短提示，不凭猜测补全命令。

KWS 用于触发判断和定位，ASR 不负责判断是否唤醒。为避免破坏连读语境，声学 PCM 不再切除唤醒区间；权威 ASR 文本会按本次 KWS 命中的关键词剥离一次，再进入命令解析。

## 6. 命令规划与执行

文本处理分为四层：

1. 规范化：数字、停用语、应用别名、路径别名和连接词
2. 切分：根据连接词、停顿和动作词生成有序命令片段
3. 映射：把每个片段映射到注册的动作类型和白名单参数
4. 校验：检查置信度、参数、权限、顺序和风险级别

解析器输出完整计划后才开始执行。执行器使用单消费者队列，按照 `sequence` 串行调用适配器。每一步产生 `queued/running/succeeded/failed/skipped` 状态和持续时间。任何适配器都不能自行插入未出现在计划中的动作。

## 7. Windows 适配优先级

1. 媒体控制优先使用 Global System Media Transport Controls
2. 打开程序和路径优先使用 ShellExecute 或明确配置的可执行文件
3. 应用专有协议或官方 CLI 优先于 UI 模拟
4. UI Automation 只用于没有稳定 API 的受控场景
5. 键鼠模拟作为最后手段，并默认关闭

## 8. 隐私与安全

- 未触发 PCM 只存在于固定容量内存中，不写入磁盘
- 诊断录音必须由用户显式开启，并显示正在录音的状态
- 应用和路径通过别名白名单配置，不允许从 ASR 文本直接拼接 shell 命令
- 删除、关闭、发送、支付等动作必须标记为高风险并二次确认
- 日志默认记录文本、动作和结果，不记录原始音频
- 触发时播放短提示音或显示可见状态，避免无感执行

## 9. 可观测性

每次触发至少记录：

- 音频设备 ID、采样格式和关键延迟统计
- KWS 模型版本、命中位置、分数和阈值
- 截取窗口和 VAD 边界
- ASR 模型版本、文本和置信度
- 完整动作计划与逐项执行结果
- AEC 的 ERLE 等可用质量指标及丢帧、漂移统计

日志使用同一个 `utterance_id` 串联，但不得包含访问令牌、完整环境变量或未经允许的原始音频。

## 10. 参考资料

- [WASAPI Loopback Recording](https://learn.microsoft.com/windows/win32/coreaudio/loopback-recording)
- [IAcousticEchoCancellationControl](https://learn.microsoft.com/windows/win32/api/audioclient/nn-audioclient-iacousticechocancellationcontrol)
- [WebRTC AudioProcessing API](https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing.h)
- [sherpa-onnx Keyword Spotting](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
- [sherpa-onnx 中英 KWS 模型](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html)
- [FunASR](https://github.com/modelscope/FunASR)
- [GlobalSystemMediaTransportControlsSessionManager](https://learn.microsoft.com/uwp/api/windows.media.control.globalsystemmediatransportcontrolssessionmanager)
