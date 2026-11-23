const {
  core,
  mpv,
  event,
  http,
  sidebar,
  standaloneWindow,
  menu,
  preferences,
  console,
  utils,
} = iina;

const POLL_INTERVAL_MS = 900;
const OSD_MIN_INTERVAL_MS = 6000;
const ZH_LANG_RE = /^(zh|zho|chi|chs|cht|cmn|zh[-_].*)$/i;
const ZH_TITLE_RE = /(中文|简体|繁體|简中|繁中|双语|雙語|chinese|zh[-_.]?ja|zh[-_.]?hans|chs|cht)/i;
const REMOTE_SCHEME_RE = /^(https?|rtmp|rtsp|mms|edl|ytdl|smb|ftp|sftp|davs?):\/\//i;

let currentVideoPath = null;
let currentProbe = null;
let currentHealth = null;
let currentJob = null;
let pollTimer = null;
let lastOsdStage = "";
let lastOsdMsg = "";
let lastOsdTime = 0;
let sidebarReady = false;
let windowPanelReady = false;
let windowPanelLoaded = false;
let sidebarLoaded = false;
let engineStarting = false;

function pref(key, fallback) {
  const val = preferences.get(key);
  return val === undefined || val === null ? fallback : val;
}

function getEngineBaseUrl() {
  const host = String(pref("engine_host", "127.0.0.1") || "127.0.0.1").trim();
  const port = Number(pref("engine_port", 18421)) || 18421;
  return `http://${host}:${port}`;
}

function getSnapshotPrefs() {
  return {
    auto_generate: Boolean(pref("auto_generate", true)),
    auto_open_sidebar: Boolean(pref("auto_open_sidebar", true)),
    subtitle_style: String(pref("subtitle_style", "bilingual") || "bilingual"),
    output_format: String(pref("output_format", "ass") || "ass"),
    source_lang: String(pref("source_lang", "ja") || "ja"),
    translator_mode: String(pref("translator_mode", "local") || "local"),
    local_model: String(pref("local_model", "auto") || "auto"),
    local_model_custom: String(pref("local_model_custom", "") || ""),
    asr_model: String(pref("asr_model", "whisper-large-v3-turbo") || "whisper-large-v3-turbo"),
    cloud_base_url: String(pref("cloud_base_url", "https://api.deepseek.com/v1") || ""),
    cloud_api_key: String(pref("cloud_api_key", "") || ""),
    cloud_model: String(pref("cloud_model", "deepseek-chat") || "deepseek-chat"),
    use_embedded_source_sub: Boolean(pref("use_embedded_source_sub", true)),
    center_channel_boost: Boolean(pref("center_channel_boost", true)),
    hf_mirror: Boolean(pref("hf_mirror", false)),
  };
}

function resolveLocalVideoPath() {
  let raw = "";
  try {
    if (typeof mpv.getString === "function") {
      raw = mpv.getString("path") || "";
    }
  } catch (_e) {}

  if (!raw && core.status && core.status.url) {
    raw = String(core.status.url);
  }
  raw = String(raw || "").trim();
  if (!raw || REMOTE_SCHEME_RE.test(raw)) {
    return null;
  }
  if (raw.startsWith("file://")) {
    raw = raw.slice(7);
    try {
      raw = decodeURIComponent(raw);
    } catch (_e) {}
  }
  if (!raw.startsWith("/")) {
    return null;
  }
  return raw;
}

function hasChineseTrackInMpv() {
  try {
    const tracks = mpv.getNative("track-list");
    if (!Array.isArray(tracks)) return false;
    for (const t of tracks) {
      if (!t || t.type !== "sub") continue;
      const lang = String(t.lang || "");
      const title = String(t.title || "");
      const extFilename = String(t["external-filename"] || "");
      if (ZH_LANG_RE.test(lang) || ZH_TITLE_RE.test(title) || ZH_TITLE_RE.test(extFilename)) {
        return true;
      }
    }
  } catch (err) {
    console.warn(`[iina-autosub] track-list inspect failed: ${err}`);
  }
  return false;
}

function parseHttpJson(res) {
  if (res && res.data && typeof res.data === "object") {
    return res.data;
  }
  if (res && typeof res.text === "string" && res.text.trim()) {
    try {
      return JSON.parse(res.text);
    } catch (_e) {
      return null;
    }
  }
  return null;
}

async function httpGetJson(path, params) {
  const url = `${getEngineBaseUrl()}${path}`;
  const options = params ? { params } : {};
  const res = await http.get(url, options);
  const data = parseHttpJson(res);
  if (res.statusCode < 200 || res.statusCode >= 300) {
    const detail = (data && data.detail) || res.text || `HTTP ${res.statusCode}`;
    throw new Error(String(detail));
  }
  return data;
}

async function httpPostJson(path, payload) {
  const url = `${getEngineBaseUrl()}${path}`;
  const res = await http.post(url, {
    headers: { "Content-Type": "application/json" },
    data: payload || {},
  });
  const data = parseHttpJson(res);
  if (res.statusCode < 200 || res.statusCode >= 300) {
    const detail = (data && data.detail) || res.text || `HTTP ${res.statusCode}`;
    throw new Error(String(detail));
  }
  return data;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkEngineHealth() {
  try {
    const data = await httpGetJson("/health");
    currentHealth = data;
    return data;
  } catch (_err) {
    currentHealth = null;
    return null;
  }
}

async function ensureEngineRunning() {
  const ok = await checkEngineHealth();
  if (ok) return ok;
  if (engineStarting) {
    for (let i = 0; i < 12; i++) {
      await delay(500);
      const retry = await checkEngineHealth();
      if (retry) return retry;
    }
    return null;
  }

  engineStarting = true;
  try {
    if (utils && typeof utils.exec === "function") {
      await utils.exec("/bin/sh", [
        "-c",
        'if [ -x "$HOME/.local/share/iina-autosub/start-engine.sh" ]; then "$HOME/.local/share/iina-autosub/start-engine.sh" >/dev/null 2>&1; else launchctl kickstart -k "gui/$(id -u)/com.iina-autosub.engine" >/dev/null 2>&1 || true; fi',
      ]);
    }
    for (let i = 0; i < 14; i++) {
      await delay(500);
      const ready = await checkEngineHealth();
      if (ready) return ready;
    }
    return null;
  } catch (err) {
    console.error(`[iina-autosub] Failed to wake local engine: ${err}`);
    return null;
  } finally {
    engineStarting = false;
  }
}

function buildUiState() {
  return {
    videoPath: currentVideoPath,
    videoName: currentVideoPath ? currentVideoPath.split("/").pop() : "",
    health: currentHealth,
    probe: currentProbe,
    job: currentJob,
    prefs: getSnapshotPrefs(),
    dashboardUrl: getEngineBaseUrl(),
  };
}

function pushUiState() {
  const payload = buildUiState();
  if (sidebarReady && sidebar && typeof sidebar.postMessage === "function") {
    try {
      sidebar.postMessage("state", payload);
    } catch (err) {
      console.warn(`[iina-autosub] sidebar.postMessage failed: ${err}`);
    }
  }
  if (windowPanelReady && standaloneWindow && typeof standaloneWindow.postMessage === "function") {
    try {
      standaloneWindow.postMessage("state", payload);
    } catch (err) {
      console.warn(`[iina-autosub] standaloneWindow.postMessage failed: ${err}`);
    }
  }
}

function bindPanelMessages(target, onReadyFlag) {
  if (!target || typeof target.onMessage !== "function") return;

  target.onMessage("ui-ready", async () => {
    onReadyFlag();
    await checkEngineHealth();
    pushUiState();
  });

  target.onMessage("start-generate", (data) => {
    startGeneration(data || {});
  });

  target.onMessage("cancel-job", () => {
    cancelActiveJob();
  });

  target.onMessage("refresh-probe", () => {
    inspectCurrentMovie(false);
  });

  target.onMessage("wake-engine", async () => {
    core.osd("AI 字幕：正在唤醒本地引擎...");
    await ensureEngineRunning();
    pushUiState();
  });

  target.onMessage("open-dashboard", () => {
    if (utils && typeof utils.open === "function") {
      utils.open(getEngineBaseUrl());
    }
  });

  target.onMessage("update-quick-pref", (data) => {
    if (!data || typeof data.key !== "string") return;
    preferences.set(data.key, data.value);
    preferences.sync();
    pushUiState();
  });
}

function ensureControlWindowLoaded() {
  if (windowPanelLoaded || !standaloneWindow || typeof standaloneWindow.loadFile !== "function") {
    return;
  }
  try {
    standaloneWindow.loadFile("ui/sidebar.html");
    if (typeof standaloneWindow.setProperty === "function") {
      standaloneWindow.setProperty({ title: "AI 字幕控制面板", resizable: true });
    }
    if (typeof standaloneWindow.setFrame === "function") {
      standaloneWindow.setFrame(400, 700);
    }
    bindPanelMessages(standaloneWindow, () => {
      windowPanelReady = true;
    });
    windowPanelLoaded = true;
  } catch (err) {
    console.warn(`[iina-autosub] standaloneWindow init failed: ${err}`);
  }
}

function ensureSidebarLoaded() {
  if (sidebarLoaded || !sidebar || typeof sidebar.loadFile !== "function") {
    return;
  }
  try {
    sidebar.loadFile("ui/sidebar.html");
    bindPanelMessages(sidebar, () => {
      sidebarReady = true;
    });
    sidebarLoaded = true;
  } catch (err) {
    console.warn(`[iina-autosub] sidebar init failed: ${err}`);
  }
}

function openControlPanel() {
  // Primary UI: standaloneWindow uses WKWebView.loadFileURL(allowingReadAccessTo:)
  // which works reliably on IINA 1.4.2 + macOS (avoids IINA's blank sidebar WKWebView bug).
  ensureControlWindowLoaded();
  if (standaloneWindow && typeof standaloneWindow.open === "function") {
    try {
      standaloneWindow.open();
      checkEngineHealth().then(() => pushUiState());
      return;
    } catch (err) {
      console.warn(`[iina-autosub] standaloneWindow.open failed: ${err}`);
    }
  }
  ensureSidebarLoaded();
  if (sidebar && typeof sidebar.show === "function") {
    try {
      sidebar.show();
      checkEngineHealth().then(() => pushUiState());
    } catch (_e) {}
  }
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function hotLoadSubtitle(subPath) {
  if (!subPath) return false;
  try {
    mpv.command("sub-add", [subPath, "select"]);
    console.log(`[iina-autosub] Hot-loaded subtitle via mpv sub-add: ${subPath}`);
    return true;
  } catch (err) {
    console.warn(`[iina-autosub] mpv sub-add failed (${err}), falling back to loadTrack`);
    try {
      core.subtitle.loadTrack(subPath);
      return true;
    } catch (err2) {
      console.error(`[iina-autosub] core.subtitle.loadTrack failed: ${err2}`);
      return false;
    }
  }
}

function maybeShowProgressOsd(job) {
  if (!job) return;
  const now = Date.now();
  const pct = Math.round((job.progress || 0) * 100);
  const stageChanged = job.stage !== lastOsdStage;
  const msgChanged = job.message !== lastOsdMsg && now - lastOsdTime >= OSD_MIN_INTERVAL_MS;
  if (stageChanged || msgChanged) {
    lastOsdStage = job.stage;
    lastOsdMsg = job.message || "";
    lastOsdTime = now;
    core.osd(`AI 字幕 [${pct}%] ${job.message}`);
  }
}

function startPolling(jobId, expectedVideoPath) {
  stopPolling();
  lastOsdStage = "";
  lastOsdMsg = "";
  lastOsdTime = 0;

  pollTimer = setInterval(async () => {
    try {
      const job = await httpGetJson(`/jobs/${jobId}`);
      if (expectedVideoPath !== currentVideoPath) {
        stopPolling();
        return;
      }
      currentJob = job;
      pushUiState();

      if (job.stage === "done") {
        stopPolling();
        if (job.subtitle_path) {
          hotLoadSubtitle(job.subtitle_path);
        }
        checkEngineHealth().then(() => pushUiState());
        const styleTag = pref("subtitle_style", "bilingual") === "bilingual" ? "中日双语" : "简体中文";
        if (job.reused_existing) {
          core.osd("AI 字幕：已加载同目录中文字幕");
        } else {
          core.osd(`AI 字幕已加载（${styleTag} · 共 ${job.segments_count || 0} 句）`);
        }
        return;
      }

      if (job.stage === "error") {
        stopPolling();
        core.osd(`AI 字幕生成失败：${job.error || job.message}`);
        return;
      }

      if (job.stage === "cancelled") {
        stopPolling();
        core.osd("AI 字幕：已取消生成");
        return;
      }

      maybeShowProgressOsd(job);
    } catch (err) {
      console.warn(`[iina-autosub] Poll error: ${err}`);
    }
  }, POLL_INTERVAL_MS);
}

async function startGeneration(opts) {
  const options = opts || {};
  const videoPath = options.videoPath || currentVideoPath || resolveLocalVideoPath();
  if (!videoPath) {
    core.osd("AI 字幕：当前没有本地电影文件");
    return;
  }
  currentVideoPath = videoPath;

  const health = await ensureEngineRunning();
  if (!health) {
    core.osd("AI 字幕：本地引擎未启动，请先运行 ./install.sh");
    openControlPanel();
    pushUiState();
    return;
  }

  const p = getSnapshotPrefs();
  const resolvedLocalModel = p.local_model_custom.trim() || p.local_model;

  if (p.translator_mode === "cloud" && !p.cloud_api_key.trim()) {
    core.osd("AI 字幕：请在插件偏好设置中填写云端 API Key，或切换为本地离线模式");
    openControlPanel();
    pushUiState();
    return;
  }

  if (p.auto_open_sidebar) {
    openControlPanel();
  }

  const body = {
    video_path: videoPath,
    source_lang: p.source_lang,
    target_lang: "zh-Hans",
    audio_track_index:
      options.audioTrackIndex !== undefined && options.audioTrackIndex !== null
        ? Number(options.audioTrackIndex)
        : null,
    subtitle_style: options.subtitleStyle || p.subtitle_style,
    output_format: p.output_format,
    asr_model: p.asr_model,
    use_embedded_source_sub: p.use_embedded_source_sub,
    force_regenerate: Boolean(options.forceRegenerate),
    ignore_cache: Boolean(options.ignoreCache),
    hf_mirror: p.hf_mirror,
    center_channel_boost: p.center_channel_boost,
    enable_vad: true,
    translator: {
      mode: p.translator_mode,
      local_model: resolvedLocalModel,
      cloud_base_url: p.cloud_base_url,
      cloud_api_key: p.cloud_api_key,
      cloud_model: p.cloud_model,
    },
  };

  try {
    core.osd("AI 字幕：开始一次性生成中文字幕（已在面板显示实时进度）");
    const job = await httpPostJson("/jobs", body);
    currentJob = job;
    pushUiState();
    startPolling(job.job_id, videoPath);
  } catch (err) {
    console.error(`[iina-autosub] Failed to submit job: ${err}`);
    core.osd(`AI 字幕提交失败：${err.message || err}`);
    currentJob = {
      job_id: "",
      video_path: videoPath,
      stage: "error",
      progress: 0,
      message: String(err.message || err),
      error: String(err.message || err),
    };
    pushUiState();
  }
}

async function cancelActiveJob() {
  if (!currentJob || !currentJob.job_id) return;
  try {
    const job = await httpPostJson(`/jobs/${currentJob.job_id}/cancel`, {});
    currentJob = job;
    stopPolling();
    pushUiState();
    core.osd("AI 字幕：已取消任务");
  } catch (err) {
    console.warn(`[iina-autosub] Cancel failed: ${err}`);
  }
}

async function inspectCurrentMovie(allowAutoTrigger) {
  stopPolling();
  currentJob = null;
  currentProbe = null;
  currentVideoPath = resolveLocalVideoPath();
  pushUiState();

  if (!currentVideoPath) {
    return;
  }

  const health = await ensureEngineRunning();
  if (!health) {
    pushUiState();
    if (allowAutoTrigger && pref("auto_generate", true)) {
      core.osd("AI 字幕：本地引擎未响应，请检查后台服务");
    }
    return;
  }

  try {
    currentProbe = await httpGetJson("/probe", {
      path: currentVideoPath,
      source_lang: String(pref("source_lang", "ja") || "ja"),
    });
  } catch (err) {
    console.warn(`[iina-autosub] Probe failed: ${err}`);
    currentProbe = null;
  }
  pushUiState();

  const alreadyHasZh =
    hasChineseTrackInMpv() ||
    (currentProbe &&
      ((currentProbe.existing_zh_subtitles && currentProbe.existing_zh_subtitles.length > 0) ||
        (currentProbe.embedded_zh_streams && currentProbe.embedded_zh_streams.length > 0)));

  if (alreadyHasZh) {
    console.log("[iina-autosub] Existing Chinese subtitle detected, skipping auto-generation.");
    return;
  }

  if (allowAutoTrigger) {
    if (pref("auto_generate", true)) {
      await startGeneration({ videoPath: currentVideoPath, forceRegenerate: false });
    } else {
      core.osd("AI 字幕：未检测到中文字幕，按 ⌘⇧S 一键生成");
    }
  }
}

// Preload standalone control window immediately (does not require iina.window-loaded)
ensureControlWindowLoaded();

// Register Plugin Menu Items
menu.addItem(
  menu.item(
    "一键生成中文字幕",
    () => {
      startGeneration({ forceRegenerate: false });
    },
    { keyBinding: "Meta+Shift+s" },
  ),
);

menu.addItem(
  menu.item("强制重新生成中文字幕", () => {
    startGeneration({ forceRegenerate: true });
  }),
);

menu.addItem(
  menu.item(
    "打开 AI 字幕面板（查看进度/切模型）",
    () => {
      openControlPanel();
    },
    { keyBinding: "Meta+Shift+a" },
  ),
);

menu.addItem(
  menu.item("在浏览器打开实时状态页", () => {
    if (utils && typeof utils.open === "function") {
      utils.open(getEngineBaseUrl());
    }
  }),
);

menu.addItem(
  menu.item("取消当前字幕生成", () => {
    cancelActiveJob();
  }),
);

// Lifecycle hooks
event.on("iina.window-loaded", () => {
  ensureSidebarLoaded();
  ensureControlWindowLoaded();
});

if (core.window && core.window.loaded) {
  ensureSidebarLoaded();
}

event.on("iina.file-loaded", () => {
  setTimeout(() => {
    inspectCurrentMovie(true);
  }, 550);
});
