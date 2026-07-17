# Desktop Voice Orchestrator

一个面向 Windows 桌面的本地优先语音控制项目。目标是在电脑附近约 2 米范围内，使用用户配置并录制校准样本的唤醒词触发一段包含多个操作的自然语言指令，并严格按照口述顺序执行。

项目同时支持两种说法：

- `唤醒词 + 多条命令`
- `多条命令 + 唤醒词`

例如：

> 小助手，暂停音乐，然后降低音量百分之十

> 播放音乐，再增加音量 5%，小助手

## 当前状态

第一、二部分的 C++20 主链路已经接通：双路 WASAPI 采集、双设备时钟对齐、WebRTC
AEC3、sherpa-onnx KWS、Silero VAD、20 秒环形缓冲、三类候选话段提取、按唤醒创建的
在线 Paraformer 会话、严格命令解析、Windows 媒体/音量控制，以及诊断录制、确定性回放
和本地 Web 调试台。未编译 AEC3、模型缺失或参考流失效时会显式降级，KWS/VAD 仍可工作，
但不可信候选不会执行系统动作。

AEC 支持选择多通道麦克风中的指定通道，并用真实 loopback/mic 互相关自动估计外部延迟；
调试台可查看估计值、相关度并施加人工修正。benchmark 会对比录制时与当前重跑的派生事件、
候选边界和命令动作序列，同时汇总 ASR latency/RTF，所有动作仍强制 dry-run。

唤醒后会进入连续激活状态。默认以 900 ms 静音结束当前话段：间隔不超过 900 ms 的多条
命令合并成一个计划，超过 900 ms 则先提交前一计划，并在 6 秒空闲窗口内把下一段语音作为
独立 follow-up 话段处理。合法 follow-up 计划会刷新空闲窗口，单次激活从首次唤醒起最多
持续 20 秒。窗口和端点都以 16 kHz 音频样本时间为准。

当前可执行命令限定为播放、暂停、相对增加音量和相对降低音量；更广泛的桌面动作编排仍属
后续阶段。连续激活不会引入上下文补全：`再小点`、`继续`等省略说法仍会被严格解析器拒绝。
回放与基准模式始终为 dry-run，不会改变系统媒体状态或音量。

项目后续仍需解决三个高风险问题：

1. USB 音箱播放期间，软件 AEC 能否在目标房间和麦克风条件下稳定保留 2 米内的人声
2. 开放词表唤醒词在前置和后置两种位置能否低误触发地检出
3. 多命令识别结果能否生成确定、可审计并严格保序的执行计划

## 核心数据流

```text
USB 音箱 WASAPI Loopback ─┐
                          ├─> IAudioPreprocessor（AEC3 / Bypass）
麦克风 WASAPI Capture ────┘                │
                                           └─> 16 kHz / 10 ms 帧
                                                   │
                                     ┌─────────────┼─────────────┐
                                     │             │             │
                                   20 s Ring   sherpa KWS    Silero VAD
                                      │             │             │
                                      └──────> 话段状态机 <────────┘
                                                    │
                                prefix / suffix / embedded / follow-up
                                      （输出 PCM 已剔除唤醒区间）
                                                    │
                                         Online Paraformer worker
                                                    │
                              严格规则解析 → 保序 Windows 执行器
                                                    │
                                           非阻塞播报调度器
```

后置唤醒词要求系统在听到唤醒词之前就保留最近一段经过回声消除的音频，因此音频前端会持续维护内存环形缓冲区，但不会持续保存录音到磁盘，也不会持续运行完整语音识别。

## 构建与运行

要求 Windows 10/11、Visual Studio 2022（MSVC）、CMake 3.28+ 和 vcpkg。模型和 sherpa-onnx
二进制均固定版本并校验 SHA-256；模型不会提交到 Git。

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\tools\vcpkg
git -C C:\tools\vcpkg checkout f87344cac03158cbf1467264565f1fd36b382a24
C:\tools\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = 'C:\tools\vcpkg'

cmake --preset windows-x64
cmake --build --preset release --target fetch_models
cmake --build --preset release --parallel
ctest --preset all
```

可执行程序位于 `build/windows-x64/Release/voice_frontend.exe`：

```powershell
./build/windows-x64/Release/voice_frontend.exe list-devices
./build/windows-x64/Release/voice_frontend.exe live
./build/windows-x64/Release/voice_frontend.exe replay data/sessions/<session>
./build/windows-x64/Release/voice_frontend.exe benchmark data/sessions/<session>
```

`live` 会打印带临时访问令牌的本机地址并打开调试台。服务只绑定 `127.0.0.1:8765`。
命令卡会显示当前激活状态、按样本时间计算的空闲/总时长倒计时以及最近一次播报。当前
`announcements.backend="log"` 只发布结构化文字事件；接口和独立有界队列为后续 TTS 保留，
播报慢、失败或队列已满均不会阻塞或改变动作顺序。
停止录制时，调试台会冻结停止瞬间的波形快照，但 C++ 实时音频链路仍会继续运行。冻结后可以选择
2/5/10/20 秒显示窗口，拖动波形轨道或时间滑块检查 VAD/KWS 触发位置；点击“回到实时”或再次开始
录制即可恢复实时滚动。
载入录制会话后，时间轴会清空实时数据并切换为该会话的回放波形。点击“回到实时”会关闭回放、
重置音频队列和推理状态，然后重新启动麦克风与 loopback 采集；live/replay 遥测不会混入同一视图。
未点击“开始录制”时，原始 PCM 只存在于定长内存缓冲中。默认参数及合法范围见
[`config/default.toml`](config/default.toml)，界面保存的覆盖项写入已忽略版本控制的
`config/local.toml`。自定义关键词的原始文本保存在 `config/keywords_raw.txt`，运行时读取的
token 文件为 `config/keywords.txt`；两者需使用固定模型对应的 `tokens.txt`/`en.phone` 和
sherpa-onnx `text2token` 流程同步生成。

第一部分的音频/话段设计见 [`docs/PART1_FRONTEND.md`](docs/PART1_FRONTEND.md)，第二部分的
AEC、流式识别、命令语法和执行边界见
[`docs/PART2_AEC_ASR_COMMANDS.md`](docs/PART2_AEC_ASR_COMMANDS.md)。

## 初步技术选型

| 层次 | 首选方案 | 说明 |
| --- | --- | --- |
| 音频采集 | Windows WASAPI | 同时采集麦克风与 USB 音箱 loopback 参考信号 |
| 回声消除 | WebRTC AEC3 | 软件 AEC 优先，实测失败后才考虑带硬件 AEC 的麦克风阵列 |
| 音频前端 | C++ | 负责低延迟采集、时钟对齐、AEC、重采样和环形缓冲区 |
| 唤醒词 | sherpa-onnx 开放词表 KWS | 通过文字或拼音配置短语，无需为每个唤醒词重新训练模型 |
| 语音识别 | FunASR 来源的 Online Paraformer，由 sherpa-onnx 运行 | 复用同一套 ONNX Runtime，KWS 命中后才创建会话 |
| 编排层 | C++ 确定性规则 | 完整消费文本，先形成计划再严格保序执行 |
| Windows 控制 | GSMTC、IAudioEndpointVolume | 使用显式 Play/Pause 和相对音量百分点，不发送媒体键 |

## 唤醒词配置方式

用户在配置时输入唤醒短语及必要的发音信息，并录制覆盖不同距离、语速和背景声的校准样本。sherpa-onnx 根据文字、拼音或音素生成关键词 token；录音用于调整 `boosting score`、触发阈值和建立独立验收集，不用于生成声纹，也不限制只有录制者能够唤醒。

任何说话人只要正确说出配置的短语并达到声学阈值，都应能够触发系统。若将来需要无法用文字、拼音或音素稳定表达的生造发音，再单独评估 Query-by-Example 录音匹配，不作为 MVP 主链路。

## MVP 验收清单

以下条目全部通过后，MVP 才视为验收完成。唤醒召回率必须对前置和后置说法分别统计，不允许用合并平均值掩盖其中一种失败。

### 测试基线

- [ ] 记录电脑、麦克风、USB 音箱 endpoint、采样格式、音箱音量、房间布局、测试距离、模型版本和阈值
- [ ] 校准集与验收集严格分离，验收集同时包含录制者和至少一名非录制者的语音
- [ ] 同一批双路录音可离线重放，并产生可重复的指标报告

### 音频与 AEC

- [ ] 麦克风 WASAPI Capture 与 USB 音箱 WASAPI Loopback 可连续同步采集
- [ ] WebRTC AEC3 输出进入 20 秒内存环形缓冲区，回放声显著衰减且不会被解析成唤醒或命令
- [ ] USB 音频设备断开重连、系统休眠恢复后，音频链路能够自动恢复

### 唤醒检测

- [ ] 修改文字或拼音配置即可更换唤醒词，不重新训练 KWS 模型
- [ ] 非录制者能够正常唤醒，系统不进行说话人身份限制
- [ ] 0.5 米安静环境的唤醒召回率不低于 95%
- [ ] 2 米安静环境的唤醒召回率不低于 90%
- [ ] 2 米音乐播放环境的唤醒召回率不低于 80%
- [ ] 日常语音、音乐和视频负样本的误唤醒少于 1 次/8 小时
- [ ] 前置和后置唤醒均能正确定位，且唤醒词本身不会进入命令解析文本

### 命令与执行

- [ ] 安静环境下四条基础命令的计划准确率不低于 95%，扬声器播放时不低于 85%
- [ ] 播放、暂停、增加音量和降低音量均可执行；相对音量默认步长为 5 个百分点
- [ ] 一次口述最多 8 条命令，动作严格保序，并分别记录 `queued/started/succeeded/noop/failed`
- [ ] 900 ms 内的命令合并执行；更长停顿后的 follow-up 在激活窗口内独立识别并保持全局 FIFO 顺序
- [ ] 每个开始执行的计划只产生一次非阻塞文字播报，播报失败不会改变动作结果
- [ ] partial、不完整候选、未知残留和超范围数值不会执行；重复 final 不会重复控制系统
- [ ] 最终识别结果到计划生成的 P95 不超过 1.5 秒

### 稳定性与隐私

- [ ] 连续运行 8 小时无崩溃、无音频设备句柄或缓冲区持续增长
- [ ] 未触发的原始音频不上传、不写盘，环形缓冲区内容按容量自动覆盖
- [ ] 诊断录音只能由用户显式开启，并有可见的录音状态
- [ ] 审计日志不包含原始音频、访问令牌或完整环境变量

## 文档

- [项目计划](docs/PROJECT_PLAN.md)
- [系统架构](docs/ARCHITECTURE.md)
- [ADR-0001：软件 AEC 优先](docs/adr/0001-software-aec-first.md)
- [ADR-0002：开放词表 KWS 与录音校准](docs/adr/0002-open-vocabulary-kws.md)

## 设计原则

- 本地优先：默认不上传音频，环形缓冲区只驻留内存
- 触发后识别：常驻链路只运行音频前端、VAD 和 KWS，完整 ASR 按需启动
- 确定性执行：先生成完整动作计划，再逐项校验和执行，不让大模型直接操作桌面
- 严格保序：命令按口述顺序串行执行，并记录每一步的开始、结果和失败原因
- 默认安全：关闭软件、删除文件、发送消息等高风险动作必须有显式确认策略
