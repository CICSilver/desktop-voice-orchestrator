# 第二部分：AEC、流式识别与媒体命令

## 数据流与线程边界

两个 WASAPI 采集线程只写各自的 SPSC 队列。音频处理线程按 QPC 时间轴把所有
mic/loopback 包依次提交给 `IAudioPreprocessor`，输出 16 kHz、单声道、10 ms 帧。
WebRTC AEC3 不可用或 render 失效时会旁路麦克风并显式标记 degraded；不会停止
KWS/VAD，但不连续或时序不可信的话段不能执行系统动作。

多通道麦克风默认选择通道 0，也可在 `aec.microphone_channel_index` 中指定其他
通道，或用 `-1` 显式求平均。AEC 延迟提示由真实（非补零）loopback 与原始 mic 的
归一化互相关自动估计，再叠加 `aec.delay_offset_ms` 人工修正；低能量或相关度不足时
拒绝更新，不用近端人声强行产生延迟值。

```text
mic + loopback -> timeline/AEC3 -> ring + KWS + VAD + segmenter
                                      |
                              KWS关联ASR队列
                                      |
                    sherpa Online Paraformer worker
                                      |
                      strict rule parser -> ordered executor
                                      |                 |
                                    GSMTC      IAudioEndpointVolume
                                      |
                         non-blocking announcement queue
```

ASR、Windows 动作、录制、遥测 JSON/WebSocket 各有独立的有界工作队列。周期性
遥测在处理线程中只复制定长数值快照，JSON 构造和网络推送在独立线程完成。任何队列
溢出都以计数或事件报告并丢弃对应工作，绝不反压实时音频路径。回放和基准模式强制
dry-run。

## 模型与依赖

- sherpa-onnx 固定为 `v1.13.2`。
- vcpkg 固定到 builtin baseline `f87344cac03158cbf1467264565f1fd36b382a24`。
  完整构建必须让本地 vcpkg checkout 与该提交一致，否则 overlay port 的上游文件哈希
  可能不同，CMake 会拒绝继续构建。
- 在线 Paraformer 使用 FunASR 来源、为 sherpa-onnx 重新导出的
  `csukuangfj/sherpa-onnx-streaming-paraformer-zh`，固定提交为
  `2a7f71bb58885c1b522ed4e683abd397355d9fc4`。encoder、decoder、tokens 的
  文件名与 SHA-256 均固定在 `models/manifest.toml`。
- 不得把 `csukuangfj/streaming-paraformer-zh` 提交
  `2db94bce7ce81909a08fb3142c71086f489e7e16` 的 `model_quant.onnx` 用作
  sherpa encoder。使用 sherpa-onnx v1.13.2 的
  `SherpaOnnxCreateOnlineRecognizer` 真实初始化时，该导出会在
  `online-paraformer-model.cc:InitEncoder:147` 触发
  `'vocab_size' does not exist in the metadata` 并直接结束进程；这不是可由
  C++ 异常回退的加载失败。兼容重导出与通用导出使用不同模型目录，避免混用。
- `windows-x64` preset 启用 vcpkg feature `webrtc-aec`；它包含 WebRTC APM 和
  SpeexDSP。`windows-x64-no-sherpa` 保留为快速核心测试 preset，并以可诊断旁路
  验证状态机。
- 模型只通过 `fetch_models` 目标下载，不在运行时联网。

真实模型初始化可独立复跑：

```powershell
cmake --preset windows-x64-sherpa-no-aec
cmake --build --preset asr-release --target dvo_asr_tests
ctest --preset asr -R "^asr::downloaded sherpa Online Paraformer model initializes$"
```

## 命令语法

解析器必须消费完整文本，不做模糊匹配：

- `播放音乐`
- `暂停音乐`
- `增加音量` / `降低音量`：未带数值时默认 5 个百分点。
- 数值允许 `百分之十`、`百分之10`、`10%`，范围 1..20。
- 一句话最多八条命令，可直接相邻，或由标点、`然后/再/接着/并且`连接。
- 默认语法不推断连接词同义词，也不接受 `然后再` 这类连续连接词；开头标点和
  连续标点形成的空子句会拒绝整句。

任意未知残留、空子句、非法数值或不完整候选都会拒绝整个计划。只有精确最终
重解码可以产生计划；partial 永不执行。播放/暂停只控制当前 GSMTC 会话，没有
会话时失败。音量调整与 loopback 使用同一输出端点，并保持原 mute 状态。

命令短语、连接词、默认音量步长和动作数量上限可热更新。每个识别会话在开始时
冻结解析器和配置 revision，因而同一候选不会混用新旧语法；这些命令参数以及
ASR partial/feed chunk、遥测频率的热更新不会重建或重置 AEC/KWS/VAD，也不会取消
当前候选。命令幂等键包含运行时 session、utterance、ASR generation、final revision
和命令配置 revision：重复 final 只产生去重结果，不重复控制系统。

动作事件按 `command_plan`、`action_queued`、`action_started`、终态顺序发布；队列拒绝
只产生 `action_submit_failed`，不产生虚假的 queued 事件。GSMTC 的 manager 获取和
播放/暂停请求共享 `commands.action_timeout_ms` 截止时间，超时产生
`action_failed/action_timeout`；同步 Core Audio 音量调用无法安全抢占，只记录耗时。

## 连续激活与分段

KWS 命中后创建一次 activation。关键词话段保留原有 prefix、suffix、embedded 定位；
关键词话段结束后，系统在默认 6 秒空闲窗口内继续接受不带唤醒词的 follow-up 话段。
follow-up 从 VAD 起音开始流式识别，端点后仍以候选完整 PCM 的 `exact-final` 重解码作为
唯一可执行结果，partial 只用于诊断显示。

`segmentation.endpoint_silence_ms` 默认保持 900 ms。新语音在 900 ms 边界到达时按语音
优先处理，因而间隔不超过 900 ms 的内容属于同一候选，并生成一次合并计划；只有间隔超过
900 ms 才关闭前一候选并创建新的 follow-up 候选。前一计划进入全局 FIFO 后，音频采集和
下一话段识别可以继续进行，但后续计划不能越过前一计划执行。

空闲窗口由成功解析并入队的 follow-up 计划刷新；拒绝、ASR 错误和队列拒绝不续期。
activation 从首次唤醒起还受默认 20 秒硬上限约束，截止前已经起音的话段可以完成。
激活期间再次命中 KWS 会取消旧 activation 并建立新 activation。设备/时间线重置、配置热
更新、replay seek/open、恢复 live 或停止运行时都会取消当前 activation。所有期限使用
16 kHz 样本时间，保证 replay 速度不会改变分段结果。

连续激活不提供会话式自然语言理解。每个话段仍须独立满足完整白名单语法，系统不会根据
上一条命令推断 `再小点`、`继续`或代词所指。

## 播报接口

执行器从全局 FIFO 取出一个计划后、首个 `action_started` 之前提交一次结构化
`AnnouncementRequest`。播报文字只根据已校验动作生成，例如
`正在执行：暂停音乐，然后降低音量 5%`，不会直接复述 ASR 原文。一个合并计划只播报一次；
被 900 ms 端点分开的计划分别播报。

`AnnouncementDispatcher` 使用独立有界 FIFO 调用 `IAnnouncementBackend`，不等待播报
完成才执行动作。当前 `log` 后端只发布 `announcement` 结构化事件，供 Web 命令卡和录制
审计显示；后端变慢、抛错或队列已满只产生诊断，不取消、阻塞或重排动作。replay/benchmark
同样保留文字事件，但不得输出真实音频。

接口为后续真实 TTS 后端预留。TTS 播放期间及结束后的默认 200 ms 尾音保护期内，前端仍
采集音频并运行 AEC，但不创建 KWS/follow-up 候选；空闲期限可按播放时长顺延，但不能突破
20 秒硬上限。真实后端必须使用进入 loopback/AEC 参考链路的 render endpoint，并报告实际
endpoint。当前版本没有合成或播放语音。

## 诊断和验收

调试台展示 raw mic/AEC 输出、AEC状态与延迟/漂移/ERLE、Paraformer partial/final、
命令计划、连续激活倒计时、文字播报及逐动作结果。录制 manifest 和 NDJSON 事件保存时间线 epoch/sequence、
模型哈希、AEC/ASR配置和动作审计信息。

`benchmark <session>` 会读取会话原有 `events.ndjson`，并在报告的 `comparison` 中输出：

- 派生事件类型序列、计数、LCS 相似度；
- 录制/重放候选边界逐项误差及是否落在一个 10 ms 帧内；
- ASR final 的 latency/inference/RTF P50、P95 和最大值；
- 历史与本次命令动作序列准确率，以及本次所有 dry-run 终态动作明细。

WebSocket 自动测试覆盖临时令牌、同源拒绝、版本化 envelope、历史快照重连、命令
端点鉴权、慢客户端有界丢弃和端口占用启动失败。服务绑定完成后 `start()` 才返回，
因此端口冲突不再静默退化为“页面打不开”。

自动测试覆盖时间线漂移/重置、AEC fallback、自动延迟、麦克风通道选择、ASR
generation与队列、严格中文规则、有序执行、幂等、音量边界、replay dry-run、
benchmark 对比和 WebSocket 端到端边界。现场验证还需在目标麦克风和音箱上完成
合成回声、安静/播放音乐、设备拔插和 8 小时耐久测试。

## 当前完成度与剩余工作

已接通并通过自动测试的工程链路包括：AEC3/SpeexDSP 编译与运行时降级、双设备时间线、
固定块采集池、KWS 关联的在线 Paraformer、候选最终精确重解码、严格命令解析、串行动作
执行、回放强制 dry-run、AEC/ASR/动作遥测，以及三路录音和旧会话兼容回放。

本轮已经补齐：

- AEC 自动延迟估计、相关度/估计值遥测、人工修正和可配置 mic 通道选择；全部参数可热更新，
  并写入配置覆盖与录制 manifest。
- benchmark 历史事件/候选边界对比、ASR 延迟/RTF 聚合、命令计划准确率和 dry-run 动作明细。
- WebSocket 令牌、同源、schema/history 重连、慢客户端及启动失败的独立端到端测试。
- 周期性遥测 JSON 和网络推送移出音频处理线程；无 Web 的 benchmark 不再生成遥测 JSON。
- SpeexDSP 漂移修正的分数比率限制为原生整数 Hz 精度，避免实时估计形成大互质比率而触发
  `RESAMPLER_ERR_OVERFLOW`；错误码和输入消费量保留在 AEC 诊断中。目标设备 live 冒烟中
  重采样失败、音频队列丢帧和遥测丢弃均为 0。

以下项目只完成了主路径，仍需继续工程化：

- WASAPI 回调已使用预分配块池和非阻塞入队，周期性遥测也已移出处理线程；但 KWS 命中后的
  ASR chunk 仍需创建共享 PCM 块，录音开启时仍需复制队列数据，异常/候选事件仍会构造 JSON。
  因而尚未达到“包括触发和诊断路径在内严格零动态分配”。
- 原计划指定的通用 Paraformer 导出与 sherpa-onnx v1.13.2 不兼容，现使用同一 FunASR
  来源、带 sherpa 元数据的固定兼容重导出；模型提交与原计划给出的提交不同。

以下项目尚未实现：

- 已支持多通道选择，但不包含波束形成、阵列几何校准或自动选取最佳通道。
- benchmark 尚不根据标注集自动计算唤醒召回率、误唤醒率和命令计划准确率；准确率仍需由
  后续标注验收工具汇总。当前的命令计划准确率只比较同一录制会话中的历史/重跑动作序列，
  不等价于对人工真值标签的准确率。
- 模糊同义词、自然语言理解、播放器自动启动、媒体键回退、绝对音量、NS/AGC 和声源定位
  属于本阶段明确排除的功能。

以下能力已有代码路径和单元测试，但尚未在目标环境完成验收，不能视为指标已达标：

- 40–200 ms 回声延迟、±300 ppm 漂移和可变 FIR 条件下的 15 dB 残余回声改善，以及近端
  人声损失不超过 3 dB。
- 0.5 米/2 米和扬声器播放场景的唤醒、ASR、命令计划准确率与 P95 延迟。
- 真实 GSMTC 播放/暂停、默认输出端点切换、静音保持和音量读回的人工联调。
- USB 拔插、休眠恢复、浏览器/磁盘/执行器阻塞注入，以及 8 小时混合播放耐久测试。
- Edge 中的波形/AEC/ASR/命令时间轴视觉验收。本轮静态资源、HTTP、API 和 JavaScript
  语法检查已通过，但自动浏览器控制运行时初始化失败，因此没有把视觉检查记为通过。

当前已知的回放对比缺口：旧会话 `20260714-192412-44e73c` 可完整重跑，音频与遥测队列
丢弃均为 0，并生成 1 条候选和 1 条 ASR final；但与历史事件配对后的最大候选边界误差为
37088 个样本，未落在一个 10 ms 帧内。该会话也没有历史命令计划可供比较。需要用当前版本
重新录制带明确命令的受控会话，区分旧配置/旧模型差异与实际回放回归，再完成验收。
