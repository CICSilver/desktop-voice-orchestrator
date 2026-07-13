# 第一部分实现与验证手册

## 已实现范围

- 两个独立的 WASAPI shared-mode、event-driven 采集线程：麦克风与 render endpoint loopback。
- `IAudioPreprocessor` 接口及 `BypassPreprocessor`：通道混合、流式线性重采样、16 kHz 单声道、10 ms 定帧；AEC 尚未实现。
- 绝对 `uint64_t` 样本时钟、QPC 映射和 20 秒覆盖式环形缓冲。
- sherpa-onnx v1.13.2 KWS 与 `silero_vad.int8.onnx`；KWS 不受 VAD 门控。
- prefix、suffix、embedded 状态机。候选源区间与扩展唤醒区间零重叠；embedded PCM 中插入配置化静音并保留左右映射。
- Boost.Beast 本机 HTTP/WebSocket 服务和无构建步骤的 HTML/CSS/JavaScript 调试台。
- 显式诊断录制、三路 float WAV、NDJSON 时间线/事件、候选 WAV、SHA-256/配置/格式/QPC manifest。
- 实时、0.5×、1×、2×、不限速回放；seek 会从会话起点不限速重算状态。
- `list-devices`、`live`、`replay`、`benchmark` 四个 C++ 入口。

## 线程与背压约束

采集线程只与各自的有界 SPSC 队列交互，不执行推理、JSON、磁盘或网络操作。处理线程消费麦克风，
loopback 只作为预处理器的可选 render 输入。录音工作线程、遥测 broker、HTTP acceptor 和每个 WebSocket
writer 都与处理线程分离。实时队列满时丢包并重置所有有状态组件；遥测发布使用 try-lock/try-push，
不会等待浏览器。录音队列竞争或溢出会把会话标记为不完整。不限速回放可在回放线程等待队列，保证
基准数据不丢失。

loopback endpoint 在完全空闲时可能不交付 WASAPI 包。采集层会按设备 mix format 生成带 QPC 的 10 ms
静音补包，并在时间线写入 `synthetic=true`，从而保证静音环境也有完整的 `loopback.wav`。真实 render
包恢复后立即以设备位置和 QPC 重新锚定。

## WebSocket 协议

每条服务端消息都是版本化 envelope：

```json
{
  "schema_version": 1,
  "seq": 42,
  "session_id": "runtime",
  "source": "live",
  "level": "info",
  "type": "telemetry",
  "timestamp_sample": 32000,
  "payload": {}
}
```

WebSocket 只承载服务端 JSON 流；控制命令使用同令牌保护的 `/api/command`。静态资源不加载 CDN。
KWS 面板只展示命中脉冲与 token 时间刻度，不伪造 sherpa-onnx 未提供的逐帧置信度。

## 会话目录

`data/sessions/<timestamp>-<id>/` 包含：

- `mic.wav`、`loopback.wav`、`processed.wav`
- `timeline.ndjson`、`events.ndjson`
- `candidates/<utterance_id>.wav`
- `manifest.json`

manifest 包含完整配置快照、实际模型/关键词 SHA-256、三路格式/帧数/QPC 范围、静音补包数、录音队列
丢弃数、完整性和停止时运行指标。原运行事件只用于对比；回放重新经过当前预处理、VAD、KWS、ring 和
状态机。

## 自动验证

```powershell
$env:VCPKG_ROOT = 'C:\tools\vcpkg'
cmake --preset windows-x64
cmake --build --preset release --target fetch_models
cmake --build --preset release --parallel
ctest --preset all --output-on-failure
```

core-only 快速循环可使用 `windows-x64-no-sherpa` / `core-release` / `ctest --preset core`。模型集成用例在
core-only 构建中跳过，在完整构建中使用官方参考 WAV 验证实际 KWS 模型、token 和 Windows DLL。

现场冒烟顺序：

1. `voice_frontend list-devices`，保存 endpoint ID 与默认标志。
2. `voice_frontend live --no-browser`，打开输出的带 token URL；确认状态为“在线”、三条时间轴同步且控制台无错误。
3. 开始录制 30 秒，覆盖静音、口述和系统播放；停止后确认三路 WAV 存在且 manifest `complete=true`。
4. 运行 `voice_frontend benchmark <session>`；要求 `audio_queue_drops=0`、`discontinuities=0`，并保存 JSON。
5. 在调试台载入会话，依次检查 0.5×/1×/2×/不限速、暂停、seek 和重放后的事件顺序。
6. 修改热参数并应用，确认 `config_applied`；修改冷参数但只点“应用”，确认 `config_rejected`；保存冷参数后确认 `restart_required`。

## 尚需人工完成的退出验证

仓库已经提供录制、回放、事件和 benchmark 基础设施，但以下结果必须在目标硬件、房间和独立数据集上
实际测量，不能由单元测试代替：0.5 m/2 m 召回率、8 小时负样本误唤醒、8 小时耐久、设备断连/休眠恢复、
磁盘与慢浏览器故障注入，以及至少一名非录制者的独立验收。音乐播放指标本阶段只记录；AEC 仍为旁路，
不作为退出门槛。
