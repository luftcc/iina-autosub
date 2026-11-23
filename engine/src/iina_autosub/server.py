from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from iina_autosub import __version__
from iina_autosub.config import (
    ASR_MODEL_PRESETS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOCAL_LLM_PRESETS,
    detect_memory_gb,
    is_apple_silicon,
    recommend_local_llm_repo,
    sync_macos_system_proxy,
)
from iina_autosub.ffmpeg_bin import get_ffmpeg_binaries
from iina_autosub.hf_download import is_hf_repo_cached
from iina_autosub.job_manager import ACTIVE_STAGES, JobManager
from iina_autosub.models import GenerateRequest, JobState, ProbeResult
from iina_autosub.pipeline.probe import probe_video

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>iina-autosub 实时状态面板</title>
<style>
  :root { color-scheme: dark light; --bg: #0f1218; --card: #171c26; --border: rgba(255,255,255,0.1); --text: #edf2f7; --muted: #94a3b8; --accent: #3b82f6; --ok: #22c55e; --err: #ef4444; }
  @media (prefers-color-scheme: light) { :root { --bg: #f5f7fb; --card: #ffffff; --border: rgba(0,0,0,0.09); --text: #111827; --muted: #64748b; } }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px 20px; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 820px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 22px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; }
  .row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
  .pill { font-size: 12px; padding: 3px 10px; border-radius: 999px; background: rgba(59,130,246,0.15); color: var(--accent); }
  .pill.ok { background: rgba(34,197,94,0.16); color: var(--ok); }
  .pill.err { background: rgba(239,68,68,0.16); color: var(--err); }
  .bar { width: 100%; height: 8px; border-radius: 999px; background: rgba(148,163,184,0.2); overflow: hidden; margin: 10px 0 6px; }
  .fill { height: 100%; background: var(--accent); transition: width 0.3s ease; }
  .meta { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; margin-top: 10px; }
  .mcard { border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-size: 12.5px; }
  .btn { padding: 5px 11px; border-radius: 7px; border: 1px solid rgba(239,68,68,0.45); background: rgba(239,68,68,0.15); color: #f87171; cursor: pointer; font-size: 12px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>iina-autosub 本地字幕引擎</h1>
  <div class="sub" id="sysInfo">连接中...</div>

  <div class="card">
    <div class="row"><strong>当前与历史任务（每秒自动刷新）</strong><span class="sub" style="margin:0" id="jobCount"></span></div>
    <div id="jobsList" style="margin-top:12px">加载中...</div>
  </div>

  <div class="card">
    <strong>本地模型缓存状态</strong>
    <div class="sub" style="margin:4px 0 0">标记「✅ 已缓存」的模型可秒级加载，无需再等待下载；在 IINA 侧边栏或插件偏好设置中可一键切换。</div>
    <div class="grid" id="modelsList"></div>
  </div>
</div>
<script>
const STAGE_MAP = { queued:"排队中", probing:"分析音轨", downloading_model:"首次下载模型", extracting_audio:"提取音轨", vad:"检测人声", asr:"日语语音识别", segmenting:"优化断句", translating:"翻译中", rendering:"生成字幕", done:"已完成", cancelled:"已取消", error:"失败" };
const ACTIVE = new Set(["queued","probing","downloading_model","extracting_audio","vad","asr","segmenting","translating","rendering"]);
function fmtDur(sec) { const s = Math.max(0, Math.floor(sec)); const m = Math.floor(s/60); return `${m}:${String(s%60).padStart(2,"0")}`; }
async function cancelJob(id) { await fetch(`/jobs/${id}/cancel`, {method:"POST"}); refresh(); }
async function refresh() {
  try {
    const [h, jobs] = await Promise.all([fetch("/health").then(r=>r.json()), fetch("/jobs").then(r=>r.json())]);
    document.getElementById("sysInfo").textContent = `引擎 v${h.version} 在线 · Apple Silicon · ${h.memory_gb}GB 统一内存 · 默认推荐 ${h.recommended_local_model.split("/").pop()}`;
    document.getElementById("jobCount").textContent = `共 ${jobs.length} 条记录`;
    const jl = document.getElementById("jobsList");
    if (!jobs.length) {
      jl.innerHTML = `<div class="sub">暂无任务。用 IINA 打开一部日本电影后，这里会实时展示进度。</div>`;
    } else {
      jl.innerHTML = jobs.map(j => {
        const pct = Math.round((j.progress||0)*100);
        const name = (j.video_path||"").split("/").pop();
        const elapsed = fmtDur((ACTIVE.has(j.stage) ? Date.now()/1000 : j.updated_at) - j.created_at);
        const cls = j.stage==="done" ? "ok" : (j.stage==="error" ? "err" : "");
        const cancelBtn = ACTIVE.has(j.stage) ? `<button class="btn" onclick="cancelJob('${j.job_id}')">取消任务</button>` : "";
        return `<div style="border-top:1px solid var(--border);padding:12px 0">
          <div class="row"><strong style="word-break:break-all">${name}</strong><div style="display:flex;gap:8px;align-items:center"><span class="pill ${cls}">${STAGE_MAP[j.stage]||j.stage} · ${pct}%</span>${cancelBtn}</div></div>
          <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
          <div class="meta"><span>${j.message||""} ${j.detail?`(${j.detail})`:""}</span><span>已用 ${elapsed}</span></div>
          ${j.subtitle_path ? `<div class="sub" style="margin:6px 0 0">输出：<code>${j.subtitle_path}</code></div>` : ""}
          ${j.error ? `<div style="color:#f87171;font-size:12px;margin-top:6px">${j.error}</div>` : ""}
        </div>`;
      }).join("");
    }
    const allModels = [...h.asr_presets.map(m=>({...m, kind:"ASR 语音识别"})), ...h.local_llm_presets.map(m=>({...m, kind:"本地翻译 LLM"}))];
    document.getElementById("modelsList").innerHTML = allModels.map(m => `
      <div class="mcard">
        <div class="row"><strong>${m.label}</strong><span class="pill ${m.cached?"ok":""}">${m.cached?"✅ 已缓存":m.size_hint}</span></div>
        <div class="sub" style="margin:4px 0 0">${m.kind} · <code>${m.repo_id.split("/").pop()}</code></div>
      </div>`).join("");
  } catch (e) {
    document.getElementById("sysInfo").textContent = "无法连接到本地引擎：" + e;
  }
}
refresh();
setInterval(refresh, 1200);
</script>
</body>
</html>
"""


def create_app() -> FastAPI:
    manager = JobManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sync_macos_system_proxy()
        yield
        manager.shutdown()

    app = FastAPI(title="iina-autosub engine", version=__version__, lifespan=lifespan)
    app.state.job_manager = manager

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/health")
    def health() -> dict[str, object]:
        ram_gb = detect_memory_gb()
        ffmpeg_ok = False
        try:
            get_ffmpeg_binaries()
            ffmpeg_ok = True
        except Exception:
            ffmpeg_ok = False
        return {
            "status": "ok",
            "version": __version__,
            "apple_silicon": is_apple_silicon(),
            "memory_gb": ram_gb,
            "recommended_local_model": recommend_local_llm_repo(ram_gb),
            "ffmpeg_ready": ffmpeg_ok,
            "asr_presets": [
                {**asdict(p), "cached": is_hf_repo_cached(p.repo_id)} for p in ASR_MODEL_PRESETS
            ],
            "local_llm_presets": [
                {**asdict(p), "cached": is_hf_repo_cached(p.repo_id)} for p in LOCAL_LLM_PRESETS
            ],
        }

    @app.get("/probe", response_model=ProbeResult)
    def probe(
        path: str = Query(..., description="Absolute local path to video file"),
        source_lang: str = Query("ja"),
    ) -> ProbeResult:
        try:
            return probe_video(path, source_lang=source_lang)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/jobs", response_model=JobState)
    def create_job(req: GenerateRequest) -> JobState:
        vpath = Path(req.video_path).expanduser()
        if not vpath.is_file():
            raise HTTPException(status_code=404, detail=f"找不到视频文件: {vpath}")
        return manager.submit(req)

    @app.get("/jobs", response_model=list[JobState])
    def list_jobs() -> list[JobState]:
        return manager.list_jobs()

    @app.get("/jobs/{job_id}", response_model=JobState)
    def get_job(job_id: str) -> JobState:
        state = manager.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="找不到对应任务")
        return state

    @app.post("/jobs/{job_id}/cancel", response_model=JobState)
    def cancel_job(job_id: str) -> JobState:
        state = manager.cancel(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="找不到对应任务")
        return state

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="找不到对应任务")

        loop = asyncio.get_running_loop()
        queue = manager.subscribe(job_id, loop)

        async def event_stream() -> AsyncGenerator[str, None]:
            try:
                while True:
                    try:
                        state = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {state.model_dump_json()}\n\n"
                        if state.stage not in ACTIVE_STAGES:
                            break
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        cur = manager.get(job_id)
                        if cur is None or cur.stage not in ACTIVE_STAGES:
                            break
            finally:
                manager.unsubscribe(job_id, queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
