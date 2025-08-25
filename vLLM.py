# app.py — Mobile-first AI Web Chat on Modal (vLLM, no external API)
# -------------------------------------------------------------------
# 单应用（单文件）在 Modal 上部署：同一容器内同时运行 vLLM 推理服务 + FastAPI 前端。
# - 前端移动端优先，流式输出（SSE→纯文本 token 转发），支持停止生成
# - 不依赖任何外部 API（直接调本机 vLLM 的 OpenAI 兼容接口）
# - 启动时自动拉模型并常驻；挂载 HuggingFace / vLLM 缓存卷
# - 仅需：`modal app.py` 部署（本文件 __main__ 里会调用 app.deploy()）
#
# 若你的 vLLM 夜ly 轮子对 Python 3.13 暂不完全兼容，可将 add_python 改为 3.11。
# -------------------------------------------------------------------

import os
import json
import time
import subprocess
from typing import AsyncGenerator, Dict, Any, List

import modal

# -------------------- 配置区 --------------------
APP_NAME = "mobile-chat-vllm"
WEB_PORT = 8000             # 对外暴露的 Web 界面端口（FastAPI）
VLLM_PORT = 4321            # vLLM OpenAI 兼容服务端口（仅容器内访问）

# 模型设置（可改成你想要的 HF 模型）
MODEL_NAME = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "6f42c8b5bf8f3f687bd6fb28833da03a19867ce8"

# 资源与并行
N_GPU = 1                   # 使用 GPU 数量（H200:1）
MAX_NUM_SEQS = 32           # vLLM 同时排队/并发序列数（按需调优）
GPU_MEM_UTIL = 0.9          # vLLM GPU 显存利用率

# 缓存卷（首次拉取模型较大，强烈建议使用缓存卷节约后续冷启动时间）
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# 运行与缩容策略
MINUTES = 60
SCALEDOWN_IDLE = 15 * MINUTES   # 空闲多久后缩容
STARTUP_TIMEOUT = 30 * MINUTES  # 冷启动超时时间

# ------------------------------------------------

app = modal.App(APP_NAME)

# 构建镜像：CUDA 基础镜像 + Python + 依赖
# 如遇 vLLM 与 Python 3.13 兼容问题，可改 add_python="3.11"
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.13"
    )
    .entrypoint([])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # 更快的权重下载
    })
    .apt_install("git", "curl")
    # 先安装 web 依赖（FastAPI / aiohttp / uvicorn 等）
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "pydantic==2.7.1",
        "aiohttp==3.9.5",
        "python-multipart==0.0.9",
    )
    # 再安装 vLLM（nightly 轮子）与 HF 传输加速
    .uv_pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/nightly",
    )
)


# 单函数：同一容器内启动 vLLM + 返回 FastAPI 应用
@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=SCALEDOWN_IDLE,
    timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=WEB_PORT, startup_timeout=STARTUP_TIMEOUT)
def web_app():
    """启动 vLLM 作为子进程，并返回 FastAPI Web 应用（移动端 UI + 流式代理）。"""
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse

    # 1) 启动 vLLM serve（OpenAI 兼容服务）
    vllm_cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "auto",
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--tensor-parallel-size", str(N_GPU),
        # 可按需添加更多参数：--kv-cache-dtype, --enable-chunked-prefill, --chat-template 等
    ]

    print("[boot] starting vLLM:", " ".join(vllm_cmd), flush=True)
    # 注意：保持子进程存活
    _vllm = subprocess.Popen(" ".join(vllm_cmd), shell=True)

    VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"

    async def wait_vllm_ready(timeout_s: int = 1200) -> None:
        """等待 vLLM /v1/models 就绪。"""
        import aiohttp
        start = time.time()
        url = f"{VLLM_BASE}/v1/models"
        async with aiohttp.ClientSession(raise_for_status=False) as sess:
            while True:
                try:
                    async with sess.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("data"):
                                print("[boot] vLLM ready.", flush=True)
                                return
                except Exception as e:
                    pass
                await asyncio.sleep(2)
                if time.time() - start > timeout_s:
                    raise RuntimeError("vLLM failed to become ready in time")

    # 等 vLLM 初次可用（不强制等到权重完全加载完，但至少 API 可连）
    try:
        import asyncio as _asyncio
        _asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    try:
        # 不阻塞太久：先等接口可连，再由前端显示加载状态
        import anyio
    except Exception:
        pass

    # 简单等待可用（不致死阻塞）
    try:
        import asyncio
        asyncio.get_event_loop().run_until_complete(wait_vllm_ready(600))
    except Exception as e:
        print("[warn] wait_vllm_ready error:", e, flush=True)

    # 2) 定义 FastAPI 应用 + 路由
    app = FastAPI(title="Mobile Chat · vLLM on Modal", version="1.0")

    MOBILE_HTML = f"""
    <!doctype html>
    <html lang=zh-CN>
    <head>
      <meta charset=utf-8 />
      <meta name=viewport content="width=device-width, initial-scale=1, maximum-scale=1" />
      <title>AI Chat · Modal (vLLM)</title>
      <style>
        :root{{--bg:#0b1020;--panel:#121836;--muted:#8b93b7;--text:#e8ecff;--acc:#6ea8fe}}
        *{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#0b1020,#0d1430);font-family:system-ui,Inter,Roboto,Helvetica,Arial;color:var(--text)}}
        header{{position:sticky;top:0;background:rgba(10,14,30,.7);backdrop-filter: blur(8px);border-bottom:1px solid #1f2a5a;z-index:10}}
        .wrap{{max-width:760px;margin:0 auto;padding:12px}}
        .card{{background:var(--panel);border:1px solid #1f2a5a;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}}
        #chat{{padding:14px;display:flex;flex-direction:column;gap:10px;margin-bottom:86px}}
        .msg{{max-width:86%;padding:10px 12px;border-radius:12px;word-wrap:break-word;line-height:1.55;}}
        .user{{align-self:flex-end;background:#1a244f}}
        .bot{{align-self:flex-start;background:#0f1533;border:1px solid #28336b}}
        footer{{position:fixed;bottom:0;left:0;right:0;background:#121836;border-top:1px solid #1f2a5a}}
        .row{{display:flex;gap:8px;align-items:flex-end;padding:10px}}
        textarea{{flex:1;min-height:50px;max-height:200px;border:1px solid #28336b;border-radius:12px;padding:10px 12px;background:#0f1533;color:var(--text);resize:vertical}}
        button{{appearance:none;border:1px solid #3552a3;background:linear-gradient(180deg,#3a60c9,#27479a);color:white;padding:10px 12px;border-radius:12px;font-weight:600;}}
        button:disabled{{opacity:.6;}}
        .muted{{color:var(--muted);font-size:12px;padding:6px 12px}}
        .pill{{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border:1px solid #28336b;border-radius:999px;color:#a7b3ff;background:#0f1533;margin-left:auto}}
        .mono{{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}}
        .toolbar{{display:flex;gap:8px;align-items:center;padding:10px 12px}}
      </style>
    </head>
    <body>
      <header>
        <div class=wrap style="display:flex;gap:10px;align-items:center;justify-content:space-between">
          <div style=font-weight:700>AI Chat · <span style=color:var(--acc)>Modal</span></div>
          <div class="pill mono" id=status>准备中…</div>
        </div>
      </header>

      <main class=wrap>
        <div class=card>
          <div class=toolbar>
            <div class=mono>Model: <span id=model>{MODEL_NAME}</span></div>
            <button id=clear>清空</button>
          </div>
          <div id=chat></div>
          <div class=muted>此页面直接调用容器内 vLLM（OpenAI 兼容）接口，无需外部 API Key。</div>
        </div>
      </main>

      <footer>
        <div class=row>
          <textarea id=input placeholder="输入消息… 按 Enter 发送，Shift+Enter 换行"></textarea>
          <button id=send>发送</button>
          <button id=stop disabled>停止</button>
        </div>
      </footer>

      <script>
        const chatEl = document.getElementById('chat');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const stopBtn = document.getElementById('stop');
        const clearBtn = document.getElementById('clear');
        const statusEl = document.getElementById('status');
        const messages = [];
        let aborter = null;

        function add(role, text){
          const d = document.createElement('div');
          d.className = 'msg ' + (role==='user'?'user':'bot');
          d.textContent = text || '';
          chatEl.appendChild(d);
          chatEl.scrollTop = chatEl.scrollHeight;
          return d;
        }

        async function health(){
          try{
            const r = await fetch('/health');
            const t = await r.text();
            statusEl.textContent = t.includes('ok') ? '就绪' : '等待 vLLM…';
          }catch{ statusEl.textContent = '等待 vLLM…'; }
        }
        setInterval(health, 4000); health();

        async function send(){
          const text = inputEl.value.trim();
          if(!text) return;
          add('user', text);
          messages.push({role:'user', content:text});
          inputEl.value = '';

          const bot = add('assistant', '');
          sendBtn.disabled = true; stopBtn.disabled = false;
          aborter = new AbortController();

          try{
            const resp = await fetch('/api/chat/stream', {
              method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({messages}), signal: aborter.signal
            });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            while(true){
              const {value, done} = await reader.read();
              if(done) break;
              buf += decoder.decode(value, {stream:true});
              // 服务器发送的是纯文本 token 流，直接追加
              bot.textContent += buf;
              buf='';
              chatEl.scrollTop = chatEl.scrollHeight;
            }
            // 把完整回复存入对话
            messages.push({role:'assistant', content: bot.textContent});
          }catch(e){
            if(e.name !== 'AbortError') bot.textContent = '❌ 出错：' + (e.message||e);
          }finally{
            sendBtn.disabled = false; stopBtn.disabled = true; aborter = null;
          }
        }

        function stop(){ if(aborter){ aborter.abort(); } }
        function clear(){ chatEl.innerHTML=''; messages.length=0; }

        sendBtn.onclick = send;
        stopBtn.onclick = stop;
        clearBtn.onclick = clear;
        inputEl.addEventListener('keydown', e=>{
          if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
        });
      </script>
    </body>
    </html>
    """

    @app.get("/", response_class=HTMLResponse)
    async def index(_: Request):
        return HTMLResponse(MOBILE_HTML)

    @app.get("/health")
    async def health():
        # 简单健康检查：探测 vLLM /v1/models
        import aiohttp
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{VLLM_BASE}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        return PlainTextResponse("ok")
        except Exception:
            pass
        return PlainTextResponse("starting", status_code=503)

    @app.get("/api/models")
    async def list_models():
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{VLLM_BASE}/v1/models", timeout=30) as r:
                data = await r.json()
                return JSONResponse(data, status_code=r.status)

    @app.post("/api/chat")
    async def chat(body: Dict[str, Any]):
        """非流式：转发到 vLLM /v1/chat/completions，并返回完整回复。"""
        import aiohttp
        payload = {
            "model": MODEL_NAME,
            "messages": body.get("messages", []),
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 512),
            "stream": False,
        }
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{VLLM_BASE}/v1/chat/completions", json=payload, timeout=600) as r:
                data = await r.json()
                if r.status != 200:
                    return JSONResponse({"error": data}, status_code=r.status)
                # 取第一条 choice
                reply = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                return JSONResponse({"reply": reply, "raw": data})

    @app.post("/api/chat/stream")
    async def chat_stream(body: Dict[str, Any]):
        """流式：从 vLLM 的 SSE 流中提取 token 文本，按纯文本 chunk 转发到前端。"""
        import aiohttp

        async def gen() -> AsyncGenerator[bytes, None]:
            payload = {
                "model": MODEL_NAME,
                "messages": body.get("messages", []),
                "temperature": body.get("temperature", 0.7),
                "max_tokens": body.get("max_tokens", 512),
                "stream": True,
            }
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(
                        f"{VLLM_BASE}/v1/chat/completions",
                        json=payload,
                        timeout=0,  # 由流控制
                    ) as resp:
                        async for raw, _ in resp.content.iter_chunks():
                            if not raw:
                                continue
                            text = raw.decode("utf-8", errors="ignore")
                            # 典型 SSE 行："data: {json...}\n\n" 或 "data: [DONE]"
                            for line in text.splitlines():
                                if not line.startswith("data:"):
                                    continue
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    return
                                try:
                                    evt = json.loads(data)
                                except Exception:
                                    continue
                                # vLLM(OpenAI兼容) delta 结构
                                choice = (evt.get("choices") or [{}])[0]
                                delta = choice.get("delta") or {}
                                token = delta.get("content")
                                if token:
                                    yield token.encode("utf-8")
            except asyncio.CancelledError:
                return
            except Exception as e:
                err = f"[stream-error] {e}"
                yield err.encode("utf-8")

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    return app


# -------------------- 部署入口 --------------------
if __name__ == "__main__":
    # 一键部署到 Modal：
    #   modal app.py
    app.deploy()
