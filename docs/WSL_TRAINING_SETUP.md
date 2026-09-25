# WSL2 训练环境安装说明

用途：在这台 Windows 机器上搭建个性化训练环境（见 [PERSONALIZATION.md](PERSONALIZATION.md)），用 RTX 4090 微调
唤醒模型（icefall / k2）和识别模型（FunASR / Fun-ASR-Nano）。本文写给执行安装的人或代理，按顺序做即可。

## 0. 约束（必须遵守）

- **C 盘只留 WSL 系统组件（约 1–2 GB）。** Linux 发行版、交换文件、所有 Python 环境、缓存、模型和数据都放在 E 盘。
  C 盘当前只剩约 49 GB，E 盘约 580 GB。
- **不要在 WSL 里安装 NVIDIA 驱动**（`nvidia-driver-*`、`cuda-drivers` 都不要装）。GPU 驱动由 Windows 提供
  （当前 610.62，已支持 WSL CUDA），装了反而会冲突。PyTorch 的 pip 包自带 CUDA 运行库，不需要 CUDA Toolkit。
- 不要修改 `E:\workspace\desktop-voice-orchestrator` 里的代码、配置和 `data\sessions` 录音；WSL 里只读访问它们。
- 训练只在 conda 环境里进行，不改系统 Python。

已确认：BIOS 虚拟化已开启；内存 31 GB；尚未安装 WSL；`%USERPROFILE%\.wslconfig` 不存在。

## 1. Windows 侧：安装 WSL2 并把发行版放到 E 盘

以**管理员 PowerShell** 执行：

```powershell
wsl --install --no-distribution
```

**重启电脑。** 然后创建 `C:\Users\10385\.wslconfig`（UTF-8，无 BOM），把交换文件放到 E 盘并限制内存：

```ini
[wsl2]
memory=20GB
processors=12
swap=16GB
swapFile=E:\\WSL\\swap.vhdx
```

安装 Ubuntu 24.04 到 `E:\WSL\Ubuntu-24.04`：

```powershell
New-Item -ItemType Directory -Force E:\WSL | Out-Null
wsl --install -d Ubuntu-24.04 --location E:\WSL\Ubuntu-24.04
```

首次启动会要求设置 Linux 用户名和密码，**由机主本人输入**。

如果这版 WSL 不认识 `--location`，改用以下方式（结果相同）：先不带 `--location` 安装并完成首次启动，然后

```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --move E:\WSL\Ubuntu-24.04
```

若 `--move` 也不支持，用导出/导入：`wsl --export Ubuntu-24.04 E:\WSL\ubuntu.tar`、`wsl --unregister Ubuntu-24.04`、
`wsl --import Ubuntu-24.04 E:\WSL\Ubuntu-24.04 E:\WSL\ubuntu.tar`，再在 `/etc/wsl.conf` 写入
`[user]` `default=<用户名>`，最后删除 `ubuntu.tar`。

让虚拟磁盘在删除文件后能回收空间：

```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --set-sparse true
```

检查：`wsl -l -v` 显示 `Ubuntu-24.04` 的 VERSION 为 2；`E:\WSL\Ubuntu-24.04\ext4.vhdx` 存在；C 盘可用空间基本没变。

## 2. Linux 侧：系统包与 conda

在 WSL 里（以下都在 Ubuntu 中执行）：

```bash
sudo apt update && sudo apt install -y build-essential git git-lfs cmake ffmpeg sox libsndfile1 python3-dev wget curl
git lfs install
nvidia-smi
```

`nvidia-smi` 必须显示 RTX 4090；显示不出来就停下，不要装驱动，检查 Windows 驱动。

安装 Miniforge 到家目录（家目录在 E 盘的虚拟磁盘里）：

```bash
wget -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3 && rm /tmp/miniforge.sh
~/miniforge3/bin/conda init bash
```

工作目录统一放在 `~/dvo`：

```bash
mkdir -p ~/dvo/models ~/dvo/data ~/dvo/corpora
```

## 3. 环境 `kws`：唤醒模型微调（icefall + k2）

k2 的 wheel 与 PyTorch、CUDA 版本严格对应。选一组官方同时提供 wheel 的组合（例如 torch 2.4.1 + CUDA 12.1），
对照 <https://k2-fsa.github.io/k2/cuda.html> 选择 k2 版本。

```bash
conda create -y -n kws python=3.10
conda activate kws
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install "k2==<与 torch 2.4.1、cuda12.1 对应的版本>" -f https://k2-fsa.github.io/k2/cuda.html
pip install lhotse sentencepiece pypinyin kaldialign onnx onnxruntime sherpa-onnx
git clone https://github.com/k2-fsa/icefall ~/dvo/icefall
pip install -r ~/dvo/icefall/requirements.txt
conda env config vars set PYTHONPATH=$HOME/dvo/icefall
```

下载当前使用的唤醒模型的训练检查点（`pretrained.pt`，epoch-13-avg-2，以及训练用的 tokens/词表）：ModelScope 仓库
`pkufool/icefall-kws-zipformer-zh-en-3M-2025-12-20`，放到 `~/dvo/models/icefall-kws-zipformer-zh-en-3M-2025-12-20`。
找不到时在 Hugging Face 上搜同名仓库。训练代码见 k2-fsa/icefall 的 PR #1428。

## 4. 环境 `funasr`：识别模型微调（Fun-ASR-Nano）

```bash
conda create -y -n funasr python=3.10
conda activate funasr
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/modelscope/FunASR ~/dvo/FunASR && pip install -e ~/dvo/FunASR
git clone https://github.com/QwenAudio/Fun-ASR ~/dvo/Fun-ASR
pip install -U modelscope huggingface_hub transformers soundfile librosa onnx onnxruntime sherpa-onnx
```

如果 Fun-ASR / FunASR 仓库的 README 对 torch、transformers 版本有明确要求，以仓库为准。

下载模型与导出脚本：

```bash
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ~/dvo/models/Fun-ASR-Nano-2512
git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx ~/dvo/sherpa-onnx
```

ModelScope 不可用时改用 `huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir ~/dvo/models/Fun-ASR-Nano-2512`。
`sherpa-onnx` 仓库只用其中的模型导出脚本，不需要编译。

## 5. 通用中文语料（可在后台下载）

AISHELL-1（OpenSLR SLR33，约 15 GB），用于微调时混入通用语音，防止模型只会识别命令：

```bash
cd ~/dvo/corpora && wget -c https://www.openslr.org/resources/33/data_aishell.tgz && wget -c https://www.openslr.org/resources/33/resource_aishell.tgz
```

下载后先不解压，后续由训练脚本处理。

## 6. 验收

全部通过才算完成：

```bash
nvidia-smi
conda run -n kws python -c "import torch, k2, lhotse; print(torch.__version__, torch.cuda.is_available(), k2.__version__)"
conda run -n funasr python -c "import torch, funasr; print(torch.__version__, torch.cuda.is_available(), funasr.__version__)"
ls ~/dvo/models/icefall-kws-zipformer-zh-en-3M-2025-12-20 ~/dvo/models/Fun-ASR-Nano-2512
ls /mnt/e/workspace/desktop-voice-orchestrator/data/sessions | head
df -h ~
```

两条 `torch.cuda.is_available()` 都必须是 `True`。在 Windows 上确认 C 盘可用空间仍在 45 GB 以上。

预计占用：E 盘 60–80 GB（两个环境约 20 GB，模型约 5 GB，AISHELL-1 下载和解压约 35 GB，其余为训练数据和检查点）；
C 盘 1–2 GB。

完成后请记录：发行版名称、Linux 用户名、两个环境里实际装上的 torch / k2 / funasr 版本，以及任何偏离本文的地方。

## 7. 安装记录（2026-09-25）

- 发行版 `Ubuntu-24.04`（WSL2），虚拟磁盘 `E:\WSL\Ubuntu-24.04\ext4.vhdx`，交换文件 `E:\WSL\swap.vhdx`；Linux 用户 `silver`。
- 稀疏虚拟磁盘未启用：当前 WSL 拒绝 `--set-sparse`，提示强制启用有数据损坏风险。影响只是删除文件后虚拟磁盘不会自动
  缩小；需要时可在 `wsl --shutdown` 后手动压缩。
- `kws`：torch 2.4.1+cu121、k2 1.24.4.dev20250714+cuda12.1.torch2.4.1、lhotse 1.33.0、sherpa-onnx 1.13.8；
  `PYTHONPATH` 指向 `~/dvo/icefall`。
- `funasr`：torch 2.10.0+cu128、funasr 1.4.16、transformers 5.17.0（按仓库要求安装，比第 4 节示例的版本新）。
- 模型：`~/dvo/models/icefall-kws-zipformer-zh-en-3M-2025-12-20`（含 `checkpoint/epoch-13.pt`、
  `pretrained-epoch-13-avg-2.pt`、`data/lang_phone`），`~/dvo/models/Fun-ASR-Nano-2512`（`model.pt` 约 2 GB，含 Qwen3-0.6B）。
- 语料：`~/dvo/corpora/data_aishell.tgz`、`resource_aishell.tgz`（未解压）。
- 两个环境在 GPU 上的验收均通过；C 盘可用空间安装前后都是约 49 GB。
