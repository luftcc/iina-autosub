// End-to-end JavaScriptCore harness test for plugin/main.js & plugin/global.js
// Run via: /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc scripts/test_plugin.js

function assert(cond, msg) {
  if (!cond) {
    throw new Error("ASSERTION FAILED: " + msg);
  }
}

// Polyfill timers for synchronous/deterministic JSC testing
const scheduledTimeouts = [];
const activeIntervals = new Map();
let nextTimerId = 1;

globalThis.setTimeout = function (fn, _ms) {
  const id = nextTimerId++;
  scheduledTimeouts.push({ id, fn });
  return id;
};
globalThis.clearTimeout = function (id) {
  const idx = scheduledTimeouts.findIndex((t) => t.id === id);
  if (idx >= 0) scheduledTimeouts.splice(idx, 1);
};
globalThis.setInterval = function (fn, _ms) {
  const id = nextTimerId++;
  activeIntervals.set(id, fn);
  return id;
};
globalThis.clearInterval = function (id) {
  activeIntervals.delete(id);
};

// Parse-check global.js and sidebar inline script first
const globalSrc = read("plugin/global.js");
new Function(globalSrc);

const sidebarHtml = read("plugin/ui/sidebar.html");
const scriptMatch = sidebarHtml.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch && scriptMatch[1], "sidebar.html missing inline <script>");
new Function(scriptMatch[1]);

// Mock IINA 1.4+ runtime
const events = {};
const sidebarListeners = {};
const postedSidebarStates = [];
const osdMessages = [];
const mpvCommands = [];
const httpLog = [];
let mockMpvPath = "/Movies/Your.Name.2016.mkv";
let mockTrackList = [{ type: "audio", lang: "jpn" }];
let mockJobStageIndex = 0;

const prefStore = {
  auto_generate: true,
  auto_open_sidebar: true,
  subtitle_style: "bilingual",
  output_format: "ass",
  source_lang: "ja",
  translator_mode: "local",
  local_model: "auto",
  local_model_custom: "",
  asr_model: "whisper-large-v3-turbo",
  cloud_base_url: "https://api.deepseek.com/v1",
  cloud_api_key: "",
  cloud_model: "deepseek-chat",
  use_embedded_source_sub: true,
  center_channel_boost: true,
  hf_mirror: false,
  engine_host: "127.0.0.1",
  engine_port: 18421,
};

let sidebarShowCount = 0;
let windowOpenCount = 0;
const windowListeners = {};
const postedWindowStates = [];

globalThis.iina = {
  core: {
    status: { url: "" },
    window: { loaded: false },
    osd(msg) {
      osdMessages.push(msg);
    },
    subtitle: {
      loadTrack(p) {
        mpvCommands.push(["loadTrack", p]);
      },
    },
  },
  mpv: {
    getString(prop) {
      return prop === "path" ? mockMpvPath : "";
    },
    getNative(prop) {
      return prop === "track-list" ? mockTrackList : null;
    },
    command(name, args) {
      mpvCommands.push([name, args]);
    },
  },
  event: {
    on(name, cb) {
      events[name] = cb;
    },
  },
  http: {
    async get(url, opts) {
      httpLog.push({ method: "GET", url, opts });
      if (url.endsWith("/health")) {
        return {
          statusCode: 200,
          data: { status: "ok", memory_gb: 128, recommended_local_model: "mlx-community/Qwen3-32B-4bit" },
        };
      }
      if (url.endsWith("/probe")) {
        return {
          statusCode: 200,
          data: {
            video_path: mockMpvPath,
            audio_tracks: [{ stream_index: 1, language: "jpn", channels: 6, codec: "aac" }],
            recommended_audio_stream: 1,
            existing_zh_subtitles: [],
            embedded_zh_streams: [],
            embedded_source_text_streams: [],
          },
        };
      }
      if (url.includes("/jobs/job123")) {
        mockJobStageIndex += 1;
        if (mockJobStageIndex === 1) {
          return {
            statusCode: 200,
            data: { job_id: "job123", stage: "asr", progress: 0.4, message: "语音识别中" },
          };
        }
        return {
          statusCode: 200,
          data: {
            job_id: "job123",
            stage: "done",
            progress: 1.0,
            message: "完成",
            subtitle_path: "/Movies/Your.Name.2016.zh-ja.ass",
            segments_count: 120,
            reused_existing: false,
          },
        };
      }
      return { statusCode: 404, text: "not found" };
    },
    async post(url, opts) {
      httpLog.push({ method: "POST", url, opts });
      if (url.endsWith("/jobs")) {
        return {
          statusCode: 200,
          data: { job_id: "job123", stage: "queued", progress: 0.0, message: "已加入队列" },
        };
      }
      return { statusCode: 200, data: { job_id: "job123", stage: "cancelled" } };
    },
  },
  sidebar: {
    loadFile(p) {
      assert(p === "ui/sidebar.html", "unexpected sidebar path: " + p);
    },
    onMessage(name, cb) {
      sidebarListeners[name] = cb;
    },
    postMessage(name, data) {
      if (name === "state") postedSidebarStates.push(data);
    },
    show() {
      sidebarShowCount += 1;
    },
  },
  standaloneWindow: {
    loadFile(p) {
      assert(p === "ui/sidebar.html", "unexpected standaloneWindow path: " + p);
    },
    setProperty() {},
    setFrame() {},
    open() {
      windowOpenCount += 1;
    },
    onMessage(name, cb) {
      windowListeners[name] = cb;
    },
    postMessage(name, data) {
      if (name === "state") postedWindowStates.push(data);
    },
  },
  menu: {
    items: [],
    item(title, cb, opts) {
      return { title, cb, opts };
    },
    addItem(it) {
      this.items.push(it);
    },
  },
  preferences: {
    get(k) {
      return prefStore[k];
    },
    set(k, v) {
      prefStore[k] = v;
    },
    sync() {},
  },
  console: {
    log() {},
    warn() {},
    error(msg) {
      print("PLUGIN ERROR:", msg);
    },
  },
  utils: {
    async exec() {
      return { status: 0, stdout: "", stderr: "" };
    },
  },
};

// Evaluate main.js inside mocked IINA runtime
const mainSrc = read("plugin/main.js");
new Function(mainSrc)();

async function flushMicrotasks(rounds = 30) {
  for (let i = 0; i < rounds; i++) {
    await Promise.resolve();
  }
}

async function runScenario() {
  assert(iina.menu.items.length === 5, "expected 5 menu items");
  events["iina.window-loaded"]();
  await sidebarListeners["ui-ready"]();
  assert(postedSidebarStates.length >= 1, "sidebar should receive initial state");

  // Scenario 1: Open Japanese movie without Chinese subtitle -> auto-open panel, auto-generate & hot-load
  await windowListeners["ui-ready"]();
  assert(postedWindowStates.length >= 1, "standalone window should receive initial state");
  events["iina.file-loaded"]();
  assert(scheduledTimeouts.length === 1, "file-loaded should schedule delayed inspect");
  scheduledTimeouts.shift().fn();
  await flushMicrotasks();
  assert(windowOpenCount >= 1 || sidebarShowCount >= 1, "should auto-open control panel when generation starts");

  // Drain 2 poll ticks -> in-progress -> done -> hot-load
  const intervalFns = Array.from(activeIntervals.values());
  assert(intervalFns.length === 1, "expected 1 active poll interval");
  await intervalFns[0]();
  await intervalFns[0]();

  assert(activeIntervals.size === 0, "poll interval should stop once done");
  const subAdd = mpvCommands.find((c) => c[0] === "sub-add");
  assert(subAdd && subAdd[1][0] === "/Movies/Your.Name.2016.zh-ja.ass" && subAdd[1][1] === "select", "should hot-load bilingual ass via mpv sub-add select");

  const postJobReq = httpLog.find((r) => r.method === "POST" && r.url.endsWith("/jobs"));
  assert(postJobReq.opts.data.subtitle_style === "bilingual", "default style should be bilingual");
  assert(postJobReq.opts.data.translator.mode === "local", "default translator should be local");

  // Scenario 2: Movie already has embedded Chinese sub -> should NOT submit a new job
  const prevPostCount = httpLog.filter((r) => r.method === "POST").length;
  mockMpvPath = "/Movies/AlreadySubbed.mkv";
  mockTrackList = [{ type: "sub", lang: "chi", title: "简体中文" }];
  events["iina.file-loaded"]();
  scheduledTimeouts.shift().fn();
  await flushMicrotasks();
  const afterPostCount = httpLog.filter((r) => r.method === "POST").length;
  assert(afterPostCount === prevPostCount, "should skip auto-generate when Chinese track already exists");

  // Scenario 3: Remote HTTP stream -> should ignore completely
  const prevHttpCount = httpLog.length;
  mockMpvPath = "https://example.com/stream.m3u8";
  mockTrackList = [];
  events["iina.file-loaded"]();
  scheduledTimeouts.shift().fn();
  await flushMicrotasks();
  assert(httpLog.length === prevHttpCount, "should ignore remote http streams");

  print("ALL PLUGIN JSC TESTS PASSED!");
}

runScenario().catch((err) => {
  print("TEST FAILED:", err, err && err.stack);
  if (typeof quit === "function") quit(1);
});
