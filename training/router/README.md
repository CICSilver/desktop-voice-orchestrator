# Desktop Voice Orchestrator — M4 本地 3B/4B 路由器精调与部署流水线 (`dvo-router`)

本目录包含 **M4 Slice C** 本地极速意图路由器（Local Command Router）的数据合成、教师模型蒸馏、A100 QLoRA 微调、离线评测与 GGUF 模型量化导出的完整工程实现。

---

## 目录结构

```text
training/router/
├── __init__.py                     # 模块导出定义
├── schema.py                       # 严格对齐 src/openai_router.cpp 的 14 动作 JSON Schema 与 Prompt 构造器
├── generate_router_dataset.py      # 涵盖 6 大场景、ASR 同音词噪声与动态白名单的 SFT 数据合成器
├── evaluate_router_offline.py      # 覆盖 Schema 合规率、精确匹配率、拒识率及延迟分位数的离线/接口评测器
├── colab_m4_router_a100.ipynb      # Google Colab A100 一键式微调与 GGUF 导出 Notebook
├── Modelfile.dvo-router            # 禁用思维链、零思考开销的 Ollama 部署描述文件
└── README.md                       # 本说明文档
```

---

## 核心设计与特性

1. **字节级对齐 C++ 运行时 (`src/openai_router.cpp`)**：
   - 严格限定 14 种受控意图动作（11 种无参媒体动作、`audio.volume.adjust`、`media.search_play`、`app.open`）。
   - 每条动作对象必须包含且仅包含 6 个属性（未使用的参数强制填写 `null`）。
   - 动态系统提示词根据白名单目录实时渲染，确保学生模型具备泛化读取白名单的能力，而非写死固定应用列表。

2. **ASR 鲁棒性与硬负样本拒识能力**：
   - 数据生成器内置 SenseVoice / Paraformer 常见拼音同音字/错别字噪声注入（如“死盯/斯蒂姆”对应 steam、“克拉什/挂梯子”对应 clash 等）。
   - 重点加固负样本（`{"actions": []}`）：涵盖日常闲聊、游戏连麦报点、歌词哼唱、提问查询、自我否定撤回（“算了别切了”）、未注册应用及越界音量请求。

3. **零思维链开销与极速响应**：
   - 针对 Qwen3 架构，显式配置 `enable_thinking=False` 并在 Modelfile 中屏蔽 `<think>` 标记，确保模型即刻输出紧凑 JSON，推理延迟压至 **80~110ms**。

---

## 一键式微调全流程 (Google Colab A100)

1. **打开并运行 Notebook**：
   - 将 [`colab_m4_router_a100.ipynb`](./colab_m4_router_a100.ipynb) 上传至 Google Colab，并将运行时类型设置为 **A100 GPU**。
   - （可选）在 Colab 左侧 `Secrets` 中配置 `GEMINI_API_KEY` 或 `OPENAI_API_KEY`。若未配置，脚本将自动采用内置离线增强引擎生成数据。
2. **流水线自动执行**：
   - **内置 ColabJobHook 状态监控与远程抢救**：全程自动向 Google Drive (`colab_runs/m4_router/`) 写入原子 `status.json`、追加 `events.jsonl` 与滚动 `tail.log`。本地执行 `python tools/watch_colab_job.py --watch --job m4_router` 即可享受零 Token 阻塞监控；若遭遇依赖或数据生成异常，Hook 会进入 `waiting_for_fix` 状态，本地修复后发送 `sync_and_retry` 即可原地重试，训练完成后自动断开 A100 运行时。
   - **Cell 1**：挂载 Drive、体检 GPU，拉取最新代码并初始化本地 NVMe 高速盘 `/content/local_router`。
   - **Cell 2**：配置 Unsloth / TRL / PEFT 及 GGUF 编译环境。
   - **Cell 3**：生成 3,000 条严格校验的 SFT 数据并备份至云盘。
   - **Cell 4**：Zero-Shot 摸底评测。
   - **Cell 5**：启动 bf16 QLoRA 精调（挂载 `make_hf_trainer_callback(hook)` 实时回传损失与步数），余弦学习率衰减，完成 Assistant-only 损失训练。
   - **Cell 6**：微调后效果评测，合并权重并导出 `dvo-qwen3-4b-router-q4_k_m.gguf` 与 `Modelfile.dvo-router`，自动同步至云盘 `desktop-voice-orchestrator-data/router_m4/export/` 并触发 `hook.finish_job(...)`。

---

## 本地主力机部署与验证 (RTX 4090)

### 1. 从 Google Drive 同步模型
从云盘 `desktop-voice-orchestrator-data/router_m4/export/` 将以下文件下载至本地仓库的 `models/router/` 目录：
- `dvo-qwen3-4b-router-q4_k_m.gguf`
- `Modelfile.dvo-router`

### 2. 通过 Ollama 构建本地服务
在命令行中执行：
```powershell
ollama create dvo-router -f training/router/Modelfile.dvo-router
```

### 3. 配置 `config/local.toml`
在本地配置文件 `config/local.toml` 的 `[commands]` 节中启用本地路由器：
```toml
[commands]
parser_mode = "hybrid"                 # 或 "router"
router_backend = "openai_compatible"
router_base_url = "http://127.0.0.1:11434/v1"
router_model = "dvo-router"
router_structured_outputs = true
router_timeout_ms = 2500
```

---

## 阅读模式 ⇄ 游戏模式 显存调度策略

RTX 4090 拥有 24GB 显存，针对不同使用场景可无缝切换：

1. **阅读 / 办公模式（Resident Low-Latency）**：
   - 让 `dvo-router` 常驻显存（仅占用约 **2.4GB VRAM**），首包响应延迟低至 **~80ms**，提供随叫随到的无感语音调度体验。
   - Ollama 默认保持 5 分钟热载，亦可在请求中指定 `keep_alive: -1` 实现永久常驻。
2. **3A 游戏模式（Instant VRAM Unload）**：
   - 当启动《黑神话：悟空》或《赛博朋克 2077》等极限压榨显存的 3A 大作时，可将 `keep_alive` 设为 `0`（即调即释）。
   - 推理完成后显存秒级归零，将整整 24GB 显存完全归还给游戏渲染与光追管线。

---

## 开发者本地离线自检

在推送代码或提交前，可在本地运行以下自检指令验证规范性：

```powershell
# 1. 验证 Schema 与数据生成器离线测试
python training/router/generate_router_dataset.py --self-test

# 2. 验证评测器指标计算与异常输入自检
python training/router/evaluate_router_offline.py --self-test

# 3. 验证 Notebook 代码单元格 AST 语法无误
python -c "import json, ast; nb=json.load(open('training/router/colab_m4_router_a100.ipynb', encoding='utf-8')); [ast.parse(''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code' and not ''.join(c['source']).lstrip().startswith(('!','%'))]; print('NB_AST_OK cells=', len(nb['cells']))"
```
