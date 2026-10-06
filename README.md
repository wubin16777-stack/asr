# ASR / 字幕工作室

运行 `subtitle_studio_singing_only.py` 可启动歌曲字幕工作室。除基础依赖外，新增 ASR 引擎的 Python 运行库按需安装：

```bash
# B：FunASR Paraformer-zh
python -m pip install funasr modelscope

# C：Qwen3-ASR（含本地强制对齐时间戳）
python -m pip install -U qwen-asr
```

设置页的 ASR 区可选择引擎/模型并下载模型权重与配置。模型下载会显示百分比，并依次尝试 ModelScope、Hugging Face 与 hf-mirror 镜像。推荐：B 使用 Paraformer-zh（中文、CPU 可跑、字符时间戳）；C 默认 Qwen3-ASR-0.6B，并配套下载 Qwen3-ForcedAligner-0.6B；显存充足时可选 1.7B。A 是原有 Faster-Whisper（large-v3 或 large-v3-turbo）。

应用会检测 NVIDIA 显卡、CUDA 和可用显存，并为本地 ASR 选择合适的设备/精度，资源不足时回退 CPU。它不会自动安装或覆盖 PyTorch、CUDA 或显卡驱动；请根据本机驱动环境单独配置 PyTorch。

**D（剪映）是非官方云端接口，不是可下载的本地模型。** 使用 D 前，界面会再次要求确认；确认后会将分离出的人声音频上传至字节跳动相关云服务，并向 AsrTools 项目使用的第三方签名服务请求签名。接口可能变化或失效；请勿上传敏感或未获授权的音频。按首次使用规则，使用 D 前仍须至少下载一个 A/B/C 本地 ASR 模型。
