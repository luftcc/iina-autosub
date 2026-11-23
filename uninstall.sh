#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_LABEL="com.iina-autosub.engine"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
SHARE_DIR="$HOME/.local/share/iina-autosub"
LOG_DIR="$HOME/Library/Logs/iina-autosub"
PLUGIN_DEST="$HOME/Library/Application Support/com.colliderli.iina/plugins/io.github.iina-autosub.iinaplugin"
CACHE_DIR="$HOME/.cache/iina-autosub"

PURGE_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --purge-cache) PURGE_CACHE=1 ;;
  esac
done

echo "==> 停止并移除后台服务 LaunchAgent"
UID_NUM="$(id -u)"
launchctl bootout "gui/${UID_NUM}/${PLIST_LABEL}" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"
rm -rf "$SHARE_DIR" "$LOG_DIR"

echo "==> 移除 IINA 插件"
rm -rf "$PLUGIN_DEST"

echo "==> 清理本地虚拟环境"
rm -rf "$ROOT_DIR/engine/.venv" "$ROOT_DIR/dist"

if [[ "$PURGE_CACHE" -eq 1 ]]; then
  echo "==> 清理 VAD 与转录缓存：$CACHE_DIR"
  rm -rf "$CACHE_DIR"
else
  echo "保留缓存目录 $CACHE_DIR（如需一并删除请运行 ./uninstall.sh --purge-cache）"
fi

echo "✅ iina-autosub 已卸载完成。"
