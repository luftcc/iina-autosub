# iina-autosub

> **打开电影一次，自动或一键根据电影声音生成适配 IINA 的中日双语字幕并热加载。**  
> 基于 Apple Silicon **MLX**（纯本地免费离线优先，支持在 IINA 设置里一键切换云端大模型 API）。

---

## 核心特性

- **零额外 App，全体验融入 IINA**：直接像平时一样用 IINA 打开日本电影，插件自动检测是否已有中文字幕；若无则自动或一键（`⌘⇧S` / `⌘⇧A` 打开「AI 字幕控制面板」）一次性生成全片字幕，**自动弹出进度与模型控制面板（含 5 步流水线状态、下载速度/体积、实时耗时）**，并在生成后**直接热挂载到当前画面**，同时支持在浏览器访问 `http://127.0.0.1:18421` 查看全局实时状态。
- **面板一键切模型**：直接在「AI 字幕控制面板」切换 ASR 模型、本地离线模型档位（实时标注 `✅ 已缓存` 与体积）或云端 API，无需层层翻找设置。

> 📖 **完整图文操作、进度查看、模型切换与排错指南请阅读：[详细使用指南 (docs/GUIDE.md)](docs/GUIDE.md)**
- **开箱即用，零环境折腾**：一条 `./install.sh` 命令自动通过 `uv` 隔离安装 Python 3.12 环境、自动准备静态 `ffmpeg/ffprobe` 与轻量 `Silero VAD`（~2MB）、注册静默后台常驻服务（空闲仅 ~40MB 内存）并安装 IINA 插件——**不需要预先安装 Homebrew、Node.js 或 Conda**。
- **纯本地免费离线优先 + 云端 API 随时切换**：
  - **本地模式（默认）**：`mlx-whisper` + `mlx-lm` 全程 Apple Silicon 硬件加速；按本机统一内存自动推荐合适模型档位（16GB → 7B / 24–36GB → 14B / 64GB+ → 32B），ASR 完成后显式释放显存再载入翻译模型，16GB Mac 也能稳跑不爆内存。
  - **云端模式（可选）**：兼容任意 OpenAI 格式接口（DeepSeek / OpenAI / Gemini / Claude 等），可在 IINA 插件偏好设置一键切换。
- **中日影视字幕专项优化**：
  - **已有内嵌日语字幕直译**：若 MKV/MP4 内嵌日语文本字幕，自动跳过语音识别直接上下文翻译，几十秒完成。
  - **5.1 / 7.1 环绕声人声增强**：自动选出日语原声轨（过滤评论音轨），下混时重点加权中置人声声道（FC），减少配乐/爆炸干扰。
  - **Silero VAD + 日语幻觉清洗**：过滤纯配乐/静音段并拦截常见日语 Whisper 幻觉（如“ご視聴ありがとうございました”）。
  - **双向上下文 + 术语表翻译**：全片先提取人名/专有名词表，再按滑动窗口携带前 8 句译文 + 后 2 句下文批量翻译，自然还原日语省略主语与称谓，严格 JSON ID 1:1 对齐校验。
  - **IINA 精调中日双语 `.ass`**：自动保存为 `<电影名>.zh-ja.ass`（同目录，下次打开 IINA 原生秒级自动加载），使用 macOS 原生 `PingFang SC`（中文主轨在上）+ `Hiragino Sans`（日文副轨在下），清晰无缺字。

---

## 系统要求

- **macOS**（推荐 Apple Silicon M1/M2/M3/M4；Intel Mac 可使用云端 API 模式）
- **IINA 1.4+**（[iina.io](https://iina.io)）

---

## 一键安装（GitHub 下载即用）

```bash
git clone https://github.com/luftcc/iina-autosub.git
cd iina-autosub
./install.sh
```

> **中国大陆网络环境**建议追加镜像参数：
> ```bash
> ./install.sh --hf-mirror
> ```

### 启用 IINA 插件（仅首次需要）

1. 打开 **IINA → 设置 (`⌘,`) → 插件**，勾选 **「启用插件系统」**（如果尚未启用），重启一次 IINA。
2. 在插件列表中确认 **「AutoSub AI 字幕」** 已启用（也可双击 `dist/iina-autosub.iinaplgz` 通过 IINA 弹窗安装）。
3. 确认 **IINA → 设置 → 字幕 → 自动加载** 保持默认的「由 IINA 智能检测」或「包含媒体文件名」。

完成！现在直接用 IINA 打开一部没有中文字幕的日本电影即可自动（或按 `⌘⇧S` / 打开右侧「AI 字幕」面板）一次性生成并热加载中日双语字幕。

---

## 本地模型内存分档（默认全离线免费）

| Mac 统一内存 | 默认自动推荐翻译模型 | 说明 |
| :--- | :--- | :--- |
| **16 GB** | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 显存占用约 4.5GB，速度快、结构稳定 |
| **24–36 GB** | `mlx-community/Qwen3-8B-4bit` | 新一代 Qwen3 8B（自动关闭思考标签），口语译文自然 |
| **48 GB+** | `mlx-community/Qwen3-14B-4bit` | 上下文与长句表现优秀（64GB/128GB 可在设置里手动升级 `Qwen3-32B-4bit`） |

你也可以在 **IINA → 设置 → 插件 → AutoSub AI 字幕 → 偏好设置** 中：
- 手动切换模型档位，或在「自定义本地 MLX 模型」填入任意 Hugging Face MLX 仓库（如 Sakura-LLM MLX 量化版）；
- 一键切换至 **云端 API**（填写 Base URL、API Key、Model，例如 DeepSeek `https://api.deepseek.com/v1` + `deepseek-chat`）；
- 在 `whisper-large-v3-turbo`（默认均衡）与 `kotoba-whisper-v2.1`（纯日语专精）间切换 ASR 模型；
- 切换 **中日双语 `.ass`** / **纯简体中文 `.ass`** / **`.srt`**。

---

## 命令行工具（可选）

除了 IINA 插件，也可以直接在终端调用核心引擎：

```bash
# 环境自检
./engine/.venv/bin/iina-autosub doctor

# 为单部日本电影生成中日双语 .ass 并直接用 IINA 打开
./engine/.venv/bin/iina-autosub generate "/path/to/movie.mkv" --open-iina

# 使用云端 DeepSeek API 翻译
./engine/.venv/bin/iina-autosub generate "/path/to/movie.mkv" \
  --mode cloud \
  --cloud-base-url "https://api.deepseek.com/v1" \
  --cloud-api-key "sk-..." \
  --cloud-model "deepseek-chat"
```

---

## 卸载

```bash
./uninstall.sh
# 若需连同本地缓存一并删除：
./uninstall.sh --purge-cache
```

---

## 项目结构

- `engine/`：Python 本地推理与字幕流水线引擎（FastAPI + `mlx-whisper` + `mlx-lm` + `Silero VAD ONNX` + `pysubs2`）
- `plugin/`：零构建依赖的 IINA 1.4+ 官方插件（事件监听、侧边栏 UI、偏好设置页、`sub-add` 热加载）
- `install.sh` / `uninstall.sh` / `scripts/pack_plugin.sh`：一键安装、后台 `LaunchAgent` 注册与 `.iinaplgz` 打包脚本

## 扩展规划

核心语言策略已解耦至 `engine/src/iina_autosub/pipeline/langs/`，后续增加「英→中」「韩→中」只需新增语言配置与翻译模板即可无缝接入。
