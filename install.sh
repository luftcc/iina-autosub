#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$ROOT_DIR/engine"
SHARE_DIR="$HOME/.local/share/iina-autosub"
LOG_DIR="$HOME/Library/Logs/iina-autosub"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_LABEL="com.iina-autosub.engine"
PLIST_PATH="$LAUNCH_AGENT_DIR/${PLIST_LABEL}.plist"
IINA_PLUGINS_DIR="$HOME/Library/Application Support/com.colliderli.iina/plugins"
PLUGIN_DEST="$IINA_PLUGINS_DIR/io.github.iina-autosub.iinaplugin"

HF_MIRROR_FLAG=""
SKIP_ASSETS=0
NO_OPEN_IINA=0

for arg in "$@"; do
  case "$arg" in
    --hf-mirror) HF_MIRROR_FLAG="--hf-mirror" ;;
    --skip-assets) SKIP_ASSETS=1 ;;
    --no-open-iina) NO_OPEN_IINA=1 ;;
    -h|--help)
      cat <<'EOF'
iina-autosub 一键安装脚本

用法:
  ./install.sh [选项]

选项:
  --hf-mirror     使用 Hugging Face 国内镜像 (hf-mirror.com) 下载基础资源
  --skip-assets   跳过预下载 static-ffmpeg 与 silero_vad.onnx
  --no-open-iina  仅直接安装插件文件，不弹出 IINA 窗口
EOF
      exit 0
      ;;
  esac
done

echo "==> [1/5] 检查 macOS 与硬件环境"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：iina-autosub 仅支持 macOS。" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "警告：检测到非 Apple Silicon (arm64)，本地 MLX 模型需要 Apple Silicon Mac，您仍可使用云端 API 模式。"
fi
if [[ ! -d "/Applications/IINA.app" ]]; then
  echo "提示：未在 /Applications/IINA.app 检测到 IINA，请确保已安装 IINA 1.4+（https://iina.io）。"
fi

echo "==> [2/5] 安装或检测 uv（自动管理隔离 Python 3.12 环境）"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "    未检测到 uv，正在安装到 ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "    使用 uv: $(command -v uv) ($(uv --version))"

echo "==> [3/5] 同步本地引擎依赖（纯 Apple MLX + ONNX）"
(
  cd "$ENGINE_DIR"
  uv sync --python 3.12
)

if [[ "$SKIP_ASSETS" -eq 0 ]]; then
  echo "==> [3.5/5] 准备 ffmpeg / ffprobe 与轻量 Silero VAD 模型 (~2MB)"
  "$ENGINE_DIR/.venv/bin/iina-autosub" fetch-assets $HF_MIRROR_FLAG
fi

echo "==> [4/5] 注册后台静默常驻服务 (LaunchAgent: 127.0.0.1:18421)"
mkdir -p "$SHARE_DIR" "$LOG_DIR" "$LAUNCH_AGENT_DIR"

cat > "$SHARE_DIR/start-engine.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
UID_NUM="\$(id -u)"
if launchctl print "gui/\${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1; then
  launchctl kickstart -k "gui/\${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1 || true
else
  launchctl bootstrap "gui/\${UID_NUM}" "${PLIST_PATH}" >/dev/null 2>&1 || \\
    nohup "${ENGINE_DIR}/.venv/bin/iina-autosub-server" >> "${LOG_DIR}/engine.log" 2>&1 &
fi
EOF
chmod +x "$SHARE_DIR/start-engine.sh"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${PLIST_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${ENGINE_DIR}/.venv/bin/iina-autosub-server</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ENGINE_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${ENGINE_DIR}/.venv/bin:/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:/usr/bin:/bin</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/engine.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/engine.err.log</string>
</dict>
</plist>
EOF

UID_NUM="$(id -u)"
launchctl bootout "gui/${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1 || true
# Wait for previous LaunchAgent teardown to settle cleanly on re-install / upgrade
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ! launchctl print "gui/${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
launchctl bootstrap "gui/${UID_NUM}" "$PLIST_PATH"
launchctl kickstart -k "gui/${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1 || true

# Verify engine health
for _ in 1 2 3 4 5 6 7 8; do
  if curl -fsS "http://127.0.0.1:18421/health" >/dev/null 2>&1; then
    echo "    本地引擎启动成功：http://127.0.0.1:18421"
    break
  fi
  sleep 0.5
done

echo "==> [5/5] 打包并安装 IINA 插件"
bash "$ROOT_DIR/scripts/pack_plugin.sh"

mkdir -p "$IINA_PLUGINS_DIR"
rm -rf "$PLUGIN_DEST"
cp -R "$ROOT_DIR/plugin" "$PLUGIN_DEST"
echo "    已安装插件至：$PLUGIN_DEST"

if [[ "$NO_OPEN_IINA" -eq 0 && -d "/Applications/IINA.app" ]]; then
  echo "    你也可以双击 dist/iina-autosub.iinaplgz 通过 IINA 原生弹窗安装。"
fi

cat <<'EOF'

============================================================
🎉 iina-autosub 安装完成！

接下来只需两步即可开箱使用：
1. 打开 IINA → 设置 (⌘,) → 插件 → 勾选「启用插件系统」（如尚未启用），
   重启 IINA 后即可在插件列表看到「AutoSub AI 字幕」。
2. 直接用 IINA 打开一部日本电影：
   - 默认自动检测无中字后一次性生成「中日双语 .ass」并热加载到当前画面；
   - 也可按 ⌘⇧S 或点击右侧「AI 字幕」侧边栏一键触发、选择音轨或切换本地/云端模型；
   - 首次运行会自动下载本地离线模型（国内网络可在插件设置中勾选 HF 国内镜像）。

诊断命令：engine/.venv/bin/iina-autosub doctor
卸载命令：./uninstall.sh
============================================================
EOF
