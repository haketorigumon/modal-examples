# app.py — Mobile-first AI Web Chat on Modal (vLLM, HF download, hot-reload, SSE streaming, auth + full UI + session/templates)
# -----------------------------------------------------------------------------------------------------------------
# 一键部署：
#   modal app.py
# 功能（完整版）：
# - 在单个 Modal 实例内运行 vLLM serve + FastAPI 前端（移动端优先）
# - 直接从 Hugging Face 下载模型（snapshot_download），支持私有仓库（设置 HF_TOKEN）
# - 模型管理：下载 / 上传 / 列表 / 热载 / 删除（热载为无缝切换到备用端口）
# - 会话管理：新建/列出/删除/重命名/导出/导入；消息持久化（SQLite）
# - Prompt 模板管理：新增/列出/删除/应用
# - 流式输出采用 SSE（JSON event），前端逐 token 渲染并支持停止生成
# - 管理接口可用 ADMIN_TOKEN 保护（在 Modal Secrets 中设置），前端可保存 token
# - 日志 tail、上传模型解压（zip/tar）
# 注意：部署需要适当的 Modal Volumes（huggingface-cache、vllm-cache、models-store、chat-store）
# -----------------------------------------------------------------------------------------------------------------

import os
import json
import time
import uuid
import sqlite3
import shutil
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional

import modal

# -------------------- 配置 --------------------
APP_NAME = "mobile-chat-vllm-pro-full"
WEB_PORT = 8000
BASE_VLLM_PORT = 4321

DEFAULT_MODEL_NAME = "ByteDance-Seed/Seed-OSS-36B-Instruct"
DEFAULT_MODEL_REV = "6f42c8b5bf8f3f687bd6fb28833da03a19867ce8"

N_GPU = 1
MAX_NUM_SEQS = 32
GPU_MEM_UTIL = 0.9

# Volumes (Modal named volumes) — create via Modal dashboard or allow create_if_missing as used here
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
chat_store_vol = modal.Volume.from_name("chat-store", create_if_missing=True)
models_vol = modal.Volume.from_name("models-store", create_if_missing=True)

CHAT_DB_DIR = "/root/chat_store"
CHAT_DB_PATH = f"{CHAT_DB_DIR}/chat.db"
MODELS_DIR = "/root/models"
LOG_PATH = "/root/vllm_run.log"

MINUTES = 60
SCALEDOWN_IDLE = 15 * MINUTES
STARTUP_TIMEOUT = 30 * MINUTES

# Admin token (optional). Set in Modal Secrets to protect management endpoints.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# ------------------------------------------------
app = modal.App(APP_NAME)

# Build image: CUDA base + Python + dependencies
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .apt_install("git", "curl")
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "pydantic==2.7.1",
        "aiohttp==3.9.5",
        "python-multipart==0.0.9",
        "huggingface_hub>=0.18.0",
    )
    .uv_pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/nightly",
    )
)


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=SCALEDOWN_IDLE,
    timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        CHAT_DB_DIR: chat_store_vol,
        MODELS_DIR: models_vol,
    },
)
@modal.web_server(port=WEB_PORT, startup_timeout=STARTUP_TIMEOUT)
def web_app():
    """主函数：管理 vLLM 子进程（支持热更）并提供完整 FastAPI 后端 + 前端页面。"""
    import asyncio
    import aiohttp
    from fastapi import FastAPI, Request, UploadFile, File, Form, Header
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse, FileResponse
    from huggingface_hub import snapshot_download

    # 运行时状态
    vllm_procs: Dict[int, subprocess.Popen] = {}
    active_port = BASE_VLLM_PORT
    current_model = DEFAULT_MODEL_NAME

    # ensure directories
    os.makedirs(CHAT_DB_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # sqlite helpers
    def db_conn():
        conn = sqlite3.connect(CHAT_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    with db_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, model TEXT, system TEXT, created_at INTEGER);
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts INTEGER);
            CREATE TABLE IF NOT EXISTS templates (id TEXT PRIMARY KEY, name TEXT, prompt TEXT, created_at INTEGER);
            """
        )
        c.commit()

    # vLLM lifecycle: start/stop/wait
    def start_vllm_on_port(port: int, model_ref: str, revision: Optional[str] = None) -> subprocess.Popen:
        model_arg = model_ref
        cmd = [
            "vllm", "serve", model_arg,
            "--served-model-name", model_arg,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--dtype", "auto",
            "--gpu-memory-utilization", str(GPU_MEM_UTIL),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--tensor-parallel-size", str(N_GPU),
        ]
        if revision:
            cmd.extend(["--revision", revision])
        f = open(LOG_PATH, "ab")
        proc = subprocess.Popen(" ".join(cmd), shell=True, stdout=f, stderr=f)
        vllm_procs[port] = proc
        print(f"[vllm:start] port={port} model={model_ref} rev={revision}", flush=True)
        return proc

    def stop_vllm_port(port: int):
        p = vllm_procs.get(port)
        if p and p.poll() is None:
            try:
                p.terminate(); p.wait(timeout=10)
            except Exception:
                try: p.kill()
                except Exception: pass
        vllm_procs.pop(port, None)

    async def wait_vllm_ready_port(port: int, timeout_s: int = 600):
        url = f"http://127.0.0.1:{port}/v1/models"
        start = time.time()
        async with aiohttp.ClientSession(raise_for_status=False) as sess:
            while True:
                try:
                    async with sess.get(url, timeout=10) as r:
                        if r.status == 200:
                            j = await r.json()
                            if j.get('data'):
                                return
                except Exception:
                    pass
                await asyncio.sleep(1)
                if time.time() - start > timeout_s:
                    raise RuntimeError(f"vLLM on port {port} not ready in time")

    # start initial model on BASE_VLLM_PORT
    start_vllm_on_port(BASE_VLLM_PORT, DEFAULT_MODEL_NAME, DEFAULT_MODEL_REV)
    try:
        asyncio.get_event_loop().run_until_complete(wait_vllm_ready_port(BASE_VLLM_PORT, 600))
    except Exception as e:
        print('[warn] initial vllm not ready:', e, flush=True)

    VLLM_BASE = lambda port=None: f"http://127.0.0.1:{port or active_port}"

    # DB functions for sessions/templates
    def now_ts(): return int(time.time())

    def new_session(title: str = "新对话", model: Optional[str] = None, system: str = "") -> str:
        sid = uuid.uuid4().hex[:12]
        with db_conn() as c:
            c.execute("INSERT INTO sessions (id,title,model,system,created_at) VALUES (?,?,?,?,?)", (sid, title, model or current_model, system, now_ts()))
            c.commit()
        return sid

    def session_get(sid: str):
        with db_conn() as c:
            cur = c.execute("SELECT * FROM sessions WHERE id=?", (sid,))
            return cur.fetchone()

    def session_list():
        with db_conn() as c:
            return [dict(r) for r in c.execute("SELECT id,title,model,system,created_at FROM sessions ORDER BY created_at DESC").fetchall()]

    def session_delete(sid: str):
        with db_conn() as c:
            c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))
            c.commit()

    def messages_get(sid: str):
        with db_conn() as c:
            return [dict(r) for r in c.execute("SELECT role,content,ts FROM messages WHERE session_id=? ORDER BY id ASC", (sid,)).fetchall()]

    def message_add(sid: str, role: str, content: str):
        with db_conn() as c:
            c.execute("INSERT INTO messages (session_id,role,content,ts) VALUES (?,?,?,?)", (sid, role, content, now_ts()))
            c.commit()

    # template helpers
    def save_template(name: str, prompt: str) -> str:
        tid = uuid.uuid4().hex[:10]
        with db_conn() as c:
            c.execute("INSERT INTO templates (id,name,prompt,created_at) VALUES (?,?,?,?)", (tid, name, prompt, now_ts()))
            c.commit()
        return tid

    def list_templates():
        with db_conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM templates ORDER BY created_at DESC").fetchall()]

    def delete_template(tid: str):
        with db_conn() as c:
            c.execute("DELETE FROM templates WHERE id=?", (tid,))
            c.commit()

    # HF download helper
    def hf_download_to_models_dir(repo_id: str, revision: Optional[str] = None) -> str:
        token = os.environ.get('HF_TOKEN')
        cache_dir = MODELS_DIR
        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, use_auth_token=token)
        safe = repo_id.replace('/', '_')
        dest = Path(MODELS_DIR) / safe
        if dest.exists():
            return str(dest)
        try:
            shutil.copytree(path, dest)
        except Exception:
            return str(path)
        return str(dest)

    # admin token check
    def check_admin_token(header_token: Optional[str]) -> bool:
        if ADMIN_TOKEN is None:
            return True
        if not header_token:
            return False
        if header_token.startswith('Bearer '):
            header_token = header_token.split(' ',1)[1]
        return header_token == ADMIN_TOKEN

    # FastAPI app (routes)
    app = FastAPI(title="Mobile Chat · vLLM (Full)")

    # Full front-end HTML (mobile-first, sessions sidebar, templates, settings, admin modal)
    FRONTEND_HTML = """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
      <title>AI Chat · Modal (vLLM)</title>
      <style>
        :root{--bg:#07102a;--card:#0f1730;--muted:#8b93b7;--accent:#6ea8fe;--text:#eaf0ff}
        *{box-sizing:border-box}
        body{margin:0;background:linear-gradient(180deg,var(--bg),#05102a);color:var(--text);font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial}
        header{padding:12px;background:linear-gradient(90deg,#07102a,#0b1636);display:flex;align-items:center;justify-content:space-between}
        .wrap{max-width:1100px;margin:0 auto;padding:10px}
        .main{display:flex;gap:12px}
        .sidebar{width:260px}
        .card{background:var(--card);border:1px solid #1f2a5a;border-radius:10px;padding:8px}
        .sessions{display:flex;flex-direction:column;gap:6px;max-height:60vh;overflow:auto}
        .sess{padding:8px;border-radius:8px;background:transparent;display:flex;justify-content:space-between;align-items:center}
        .sess.active{background:#11213d}
        #chat{flex:1;display:flex;flex-direction:column;gap:8px;max-height:64vh;overflow:auto;padding:10px}
        .msg{max-width:80%;padding:10px;border-radius:12px;word-break:break-word;line-height:1.5}
        .user{align-self:flex-end;background:#244070}
        .bot{align-self:flex-start;background:#0b1530;border:1px solid #22335a}
        footer{position:fixed;left:0;right:0;bottom:0;background:linear-gradient(90deg,#07102a,#0b1636);padding:10px;border-top:1px solid #1f2a5a}
        .row{display:flex;gap:8px;align-items:center}
        textarea{flex:1;min-height:56px;max-height:200px;border-radius:10px;padding:10px;border:1px solid #263155;background:#07102a;color:var(--text);resize:vertical}
        button{padding:8px 10px;border-radius:8px;border:none;background:var(--accent);color:#fff;font-weight:600}
        .muted{color:var(--muted);font-size:13px}
        .small{font-size:12px;color:#9fb0ff}
        .toolbar{display:flex;gap:8px;align-items:center;justify-content:space-between}
        .modal{position:fixed;left:0;right:0;top:0;bottom:0;background:rgba(0,0,0,0.6);display:none;align-items:center;justify-content:center}
        .modal .inner{background:#0b1636;padding:16px;border-radius:10px;max-width:90%;width:720px}
        input[type=text],select,input[type=password]{padding:8px;border-radius:8px;border:1px solid #263155;background:#07102a;color:var(--text)}
        .flexcol{display:flex;flex-direction:column;gap:8px}
        @media(max-width:900px){ .main{flex-direction:column} .sidebar{width:100%;order:2} #chat{order:1} }
      </style>
    </head>
    <body>
      <header>
        <div style="font-weight:700">AI Chat · <span style="color:var(--accent)">Modal</span></div>
        <div class=small id=status>初始化…</div>
      </header>

      <div class=wrap>
        <div class=main>
          <div class=sidebar>
            <div class=card>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <div style="font-weight:700">会话</div>
                <div><button id=new_session>新建</button></div>
              </div>
              <div class="sessions" id=session_list></div>
              <div class=muted style="margin-top:8px">提示：会话保存在容器的持久化卷中。</div>
            </div>

            <div style="height:8px"></div>

            <div class=card>
              <div style="font-weight:700;margin-bottom:6px">模板</div>
              <div id=template_list class=flexcol style="max-height:26vh;overflow:auto"></div>
              <div style="margin-top:8px;display:flex;gap:6px"><input id=tpl_name placeholder="模板名"/><input id=tpl_prompt placeholder="Prompt 内容"/><button id=tpl_save>保存模板</button></div>
            </div>

            <div style="height:8px"></div>

            <div class=card>
              <div style="font-weight:700;margin-bottom:6px">模型 / 设置</div>
              <div class="flexcol">
                <select id=model_select></select>
                <div style="display:flex;gap:6px"><input id=temp type=text placeholder="温度 (0.0-1.0)" value="0.7"/><input id=max_tokens type=text placeholder="max tokens" value="512"/></div>
                <div style="display:flex;gap:6px"><input id=system_prompt placeholder="System 提示（可选）"/></div>
                <div style="display:flex;gap:6px"><button id=set_model>设置为默认并加载</button><button id=export_session>导出会话</button><button id=import_session>导入会话</button></div>
              </div>
            </div>
          </div>

          <div style="flex:1;display:flex;flex-direction:column">
            <div class=card style="flex:1;display:flex;flex-direction:column">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><div style="font-weight:700">聊天</div><div class=muted id=current_model>Model: (loading)</div></div>
              <div id=chat></div>
            </div>
          </div>
        </div>
      </div>

      <footer>
        <div class=row style="max-width:1100px;margin:0 auto;">
          <textarea id=input placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
          <button id=send>发送</button>
          <button id=stop disabled>停止</button>
        </div>
      </footer>

      <!-- Admin modal -->
      <div id=admin_modal class=modal>
        <div class=inner>
          <div style="display:flex;justify-content:space-between;align-items:center"><div style="font-weight:700">管理面板</div><div><button id=close_admin>关闭</button></div></div>
          <div class=flexcol style="margin-top:10px">
            <div><label>管理员令牌（如启用）：</label><input id=admin_token type=password placeholder="管理员令牌" /><button id=save_token>保存</button></div>
            <div><label>从 Hugging Face 下载（repo id）</label><input id=hf_repo type=text placeholder="repo_id (eg. facebook/galactica)"/><input id=hf_rev type=text placeholder="revision (optional)"/><button id=hf_download>下载并添加</button></div>
            <div><label>上传模型归档（zip/tar）</label><input id=model_file type=file/><button id=upload_model>上传</button></div>
            <div><label>本地模型</label><div id=model_list class=flexcol style="max-height:160px;overflow:auto"></div></div>
            <div><label>日志</label><pre id=log_tail style="height:160px;overflow:auto;background:#07142a;padding:8px;border-radius:6px"></pre><button id=refresh_log>刷新日志</button></div>
          </div>
        </div>
      </div>

      <script>
        // Frontend logic: sessions, templates, models, chat (SSE streaming)
        const sessionListEl = document.getElementById('session_list');
        const chatEl = document.getElementById('chat');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const stopBtn = document.getElementById('stop');
        const newSessionBtn = document.getElementById('new_session');
        const tplSaveBtn = document.getElementById('tpl_save');
        const tplName = document.getElementById('tpl_name');
        const tplPrompt = document.getElementById('tpl_prompt');
        const tplListEl = document.getElementById('template_list');
        const modelSelect = document.getElementById('model_select');
        const setModelBtn = document.getElementById('set_model');
        const tempEl = document.getElementById('temp');
        const maxTokensEl = document.getElementById('max_tokens');
        const systemPromptEl = document.getElementById('system_prompt');
        const currentModelEl = document.getElementById('current_model');

        const adminBtn = document.getElementById('open_admin');
        const adminModal = document.getElementById('admin_modal');
        const closeAdmin = document.getElementById('close_admin');
        const saveTokenBtn = document.getElementById('save_token');
        const adminTokenInput = document.getElementById('admin_token');
        const hfRepo = document.getElementById('hf_repo');
        const hfRev = document.getElementById('hf_rev');
        const hfDownloadBtn = document.getElementById('hf_download');
        const modelFileInput = document.getElementById('model_file');
        const uploadModelBtn = document.getElementById('upload_model');
        const modelListDiv = document.getElementById('model_list');
        const logTail = document.getElementById('log_tail');
        const refreshLogBtn = document.getElementById('refresh_log');

        let currentSession = null;
        let sseController = null;

        function authHeaders(){ const t = localStorage.getItem('admin_token'); return t ? {'Authorization':'Bearer '+t} : {}; }

        async function api(path, opts={}){
          opts.headers = Object.assign({'Content-Type':'application/json'}, opts.headers||{}, authHeaders());
          if(opts.body && typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);
          const r = await fetch(path, opts);
          return r;
        }

        // Sessions
        async function loadSessions(){
          const r = await fetch('/api/sessions');
          const j = await r.json();
          sessionListEl.innerHTML = '';
          for(const s of j){
            const el = document.createElement('div'); el.className='sess';
            el.innerHTML = `<div style="flex:1">${s.title || s.id}</div><div style="display:flex;gap:6px"><button class=s_switch data-id="${s.id}">切换</button><button class=s_rename data-id="${s.id}">重命名</button><button class=s_export data-id="${s.id}">导出</button><button class=s_del data-id="${s.id}">删除</button></div>`;
            sessionListEl.appendChild(el);
          }
          // handlers
          sessionListEl.querySelectorAll('.s_switch').forEach(b=>b.onclick=()=>{ openSession(b.dataset.id); });
          sessionListEl.querySelectorAll('.s_rename').forEach(b=>b.onclick=async ()=>{ const id=b.dataset.id; const t=prompt('新会话名'); if(t){ await fetch('/api/sessions/'+id+'/rename',{method:'POST',headers:Object.assign({'Content-Type':'application/json'}, authHeaders()),body:JSON.stringify({title:t})}); loadSessions(); }});
          sessionListEl.querySelectorAll('.s_export').forEach(b=>b.onclick=()=>{ window.location='/api/sessions/'+b.dataset.id+'/export'; });
          sessionListEl.querySelectorAll('.s_del').forEach(b=>b.onclick=async ()=>{ if(!confirm('删除会话？')) return; const r=await fetch('/api/sessions/'+b.dataset.id,{method:'DELETE'}); if(r.ok){ loadSessions(); if(currentSession===b.dataset.id){ currentSession=null; chatEl.innerHTML=''; } } });
        }

        async function createSession(){
          const r = await fetch('/api/sessions',{method:'POST',headers: {'Content-Type':'application/json'}, body: JSON.stringify({title:'新对话'})});
          const j = await r.json(); if(j.id){ loadSessions(); openSession(j.id); }
        }

        newSessionBtn.onclick = createSession;

        async function openSession(sid){
          currentSession = sid;
          chatEl.innerHTML = '';
          const r = await fetch('/api/sessions/'+sid+'/messages');
          if(r.ok){ const msgs = await r.json(); for(const m of msgs){ addLocalMsg(m.role, m.content); } }
        }

        function addLocalMsg(role,text){ const d=document.createElement('div'); d.className='msg '+(role==='user'?'user':'bot'); d.textContent=text; chatEl.appendChild(d); chatEl.scrollTop=chatEl.scrollHeight; }

        // Templates
        async function loadTemplates(){ const r=await fetch('/api/templates'); const j=await r.json(); tplListEl.innerHTML=''; for(const t of j){ const el=document.createElement('div'); el.className='row'; el.innerHTML=`<div style="flex:1">${t.name}</div><div><button class=tpl_apply data-id="${t.id}">应用</button><button class=tpl_del data-id="${t.id}">删除</button></div>`; tplListEl.appendChild(el); }
          tplListEl.querySelectorAll('.tpl_apply').forEach(b=>b.onclick=async ()=>{ const id=b.dataset.id; const rr=await fetch('/api/templates'); const list=await rr.json(); const tpl=list.find(x=>x.id===id); if(tpl) inputEl.value = tpl.prompt; });
          tplListEl.querySelectorAll('.tpl_del').forEach(b=>b.onclick=async ()=>{ if(!confirm('删除模板？')) return; await fetch('/api/templates/'+b.dataset.id,{method:'DELETE'}); loadTemplates(); });
        }
        tplSaveBtn.onclick = async ()=>{ const name=tplName.value.trim(); const prompt=tplPrompt.value.trim(); if(!name||!prompt) return alert('请输入名和内容'); const r=await fetch('/api/templates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,prompt})}); if(r.ok){ tplName.value=''; tplPrompt.value=''; loadTemplates(); } };

        // Models: list and load
        async function loadModels(){ const r = await fetch('/api/models/list'); if(!r.ok) return; const j = await r.json(); modelSelect.innerHTML=''; currentModelEl.textContent = 'Model: ' + (j.online?.[0]?.id || 'unknown');
          // online
          (j.online||[]).forEach(m=>{ const opt=document.createElement('option'); opt.value=m.id||m.name; opt.textContent = m.id||m.name; modelSelect.appendChild(opt); });
          (j.local||[]).forEach(m=>{ const opt=document.createElement('option'); opt.value=m.path; opt.textContent = m.id + ' (local)'; modelSelect.appendChild(opt); });
        }

        setModelBtn.onclick = async ()=>{
          const m = modelSelect.value; if(!m) return alert('请选择模型'); const r = await fetch('/api/models/reload',{method:'POST',headers:Object.assign({'Content-Type':'application/json'}, authHeaders()),body:JSON.stringify({model:m})}); if(r.ok){ alert('模型正在重载（请等待就绪）'); loadModels(); } else { const t = await r.text(); alert('错误: '+t); }
        }

        // Admin modal
        document.getElementById('open_admin').onclick = ()=>{ adminModal.style.display='flex'; adminTokenInput.value = localStorage.getItem('admin_token')||''; loadModelList(); refreshLog(); }
        closeAdmin.onclick = ()=>{ adminModal.style.display='none'; }
        saveTokenBtn.onclick = ()=>{ localStorage.setItem('admin_token', adminTokenInput.value); alert('已保存'); }

        async function loadModelList(){ const r = await fetch('/api/models/list'); if(!r.ok) return; const j = await r.json(); modelListDiv.innerHTML=''; (j.local||[]).forEach(m=>{ const el=document.createElement('div'); el.className='row'; el.innerHTML=`<div style="flex:1">${m.id}</div><div><button class=load data-path="${m.path}">加载</button><button class=del data-name="${m.id}">删除</button></div>`; modelListDiv.appendChild(el); }); modelListDiv.querySelectorAll('.load').forEach(b=>b.onclick=async ()=>{ const path=b.dataset.path; const r = await fetch('/api/models/reload',{method:'POST',headers:Object.assign({'Content-Type':'application/json'}, authHeaders()),body:JSON.stringify({model:path})}); if(r.ok){ alert('加载成功'); loadModels(); } else { alert('加载失败'); } }); modelListDiv.querySelectorAll('.del').forEach(b=>b.onclick=async ()=>{ if(!confirm('删除模型目录？')) return; const r = await fetch('/api/models/delete',{method:'POST',headers:Object.assign({'Content-Type':'application/json'}, authHeaders()),body:JSON.stringify({model:b.dataset.name})}); if(r.ok){ alert('删除成功'); loadModelList(); } else { alert('删除失败'); } }); }

        hfDownloadBtn.onclick = async ()=>{ const repo = hfRepo.value.trim(); if(!repo) return alert('请输入 repo id'); hfDownloadBtn.disabled=true; const r = await fetch('/api/models/download',{method:'POST',headers:Object.assign({'Content-Type':'application/json'}, authHeaders()),body:JSON.stringify({repo_id:repo,revision: hfRev.value.trim()||null})}); hfDownloadBtn.disabled=false; if(r.ok){ alert('下载完成'); loadModelList(); loadModels(); } else { const t = await r.text(); alert('错误: '+t); } }
        uploadModelBtn.onclick = async ()=>{ const f = modelFileInput.files[0]; if(!f) return alert('请选择文件'); const fd = new FormData(); fd.append('file', f); uploadModelBtn.disabled=true; const r = await fetch('/api/models/upload', {method:'POST', body: fd, headers: authHeaders()}); uploadModelBtn.disabled=false; if(r.ok){ alert('上传并解压完成'); loadModelList(); loadModels(); } else { const t = await r.text(); alert('上传失败:'+t); } }

        refreshLogBtn.onclick = refreshLog;
        async function refreshLog(){ const r = await fetch('/api/vllm/log', {headers: authHeaders()}); if(r.ok){ logTail.textContent = await r.text(); logTail.scrollTop = logTail.scrollHeight; } }

        // Templates and sessions initial load
        loadSessions(); loadTemplates(); loadModels();
        setInterval(loadSessions, 8000);

        // Chat (SSE streaming)
        async function send(){
          const text = inputEl.value.trim(); if(!text) return; addLocalMsg('user', text); inputEl.value=''; sendBtn.disabled=true; stopBtn.disabled=false;
          // prepare payload; server will append messages to session
          const payload = { session_id: currentSession, messages: [{role:'user', content:text}], temperature: parseFloat(tempEl.value)||0.7, max_tokens: parseInt(maxTokensEl.value)||512, system: systemPromptEl.value||'' };
          const headers = Object.assign({'Content-Type':'application/json'}, authHeaders());
          sseController = new AbortController();
          try{
            const resp = await fetch('/api/chat/stream', {method:'POST', headers, body: JSON.stringify(payload), signal: sseController.signal});
            if(!resp.ok){ const t=await resp.text(); addLocalMsg('assistant','错误: '+t); return }
            const reader = resp.body.getReader(); const decoder = new TextDecoder(); let buf=''; let botDiv=null;
            while(true){ const {value, done} = await reader.read(); if(done) break; buf += decoder.decode(value, {stream:true});
              // SSE parse
              let parts = buf.split('

');
              for(let i=0;i<parts.length-1;i++){
                const part = parts[i].trim(); if(!part.startsWith('data:')) continue; const jsonStr = part.slice(5).trim(); try{ const ev = JSON.parse(jsonStr); if(ev.type==='token'){ if(!botDiv) botDiv = addLocalMsg('assistant',''); botDiv.textContent += ev.token; chatEl.scrollTop = chatEl.scrollHeight; } else if(ev.type==='tool'){ addLocalMsg('assistant','[TOOL] '+JSON.stringify(ev.tool)); } else if(ev.type==='error'){ addLocalMsg('assistant','错误: '+ev.error); } }
                catch(e){ console.warn('sse json parse', e); }
              }
              buf = parts[parts.length-1];
            }
          }catch(e){ if(e.name!=='AbortError') addLocalMsg('assistant','请求失败: '+(e.message||e)); }
          finally{ sendBtn.disabled=false; stopBtn.disabled=true; sseController=null; }
        }

        function addLocalMsg(role,text){ const d=document.createElement('div'); d.className='msg '+(role==='user'?'user':'bot'); d.textContent=text; chatEl.appendChild(d); chatEl.scrollTop = chatEl.scrollHeight; }

        stopBtn.onclick = ()=>{ if(sseController) sseController.abort(); }
        sendBtn.onclick = send;
        inputEl.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });

        // session export/import endpoints
        document.getElementById('export_session').onclick = ()=>{ if(!currentSession) return alert('先选择会话'); window.location = '/api/sessions/'+currentSession+'/export'; };
        document.getElementById('import_session').onclick = async ()=>{ const f = await new Promise(r=>{ const inp = document.createElement('input'); inp.type='file'; inp.accept='application/json'; inp.onchange=()=>r(inp.files[0]); inp.click(); }); if(!f) return; const fd = new FormData(); fd.append('file', f); const r = await fetch('/api/sessions/import', {method:'POST', body: fd}); if(r.ok){ alert('导入完成'); loadSessions(); } else { alert('导入失败'); } };

        // template load at start
        async function loadTemplates(){ const r=await fetch('/api/templates'); if(r.ok){ const j=await r.json(); tplListEl.innerHTML=''; for(const t of j){ const el=document.createElement('div'); el.className='row'; el.innerHTML = `<div style="flex:1">${t.name}</div><div><button class=tpl_apply data-id="${t.id}">应用</button><button class=tpl_del data-id="${t.id}">删除</button></div>`; tplListEl.appendChild(el); } tplListEl.querySelectorAll('.tpl_apply').forEach(b=>b.onclick=async ()=>{ const id=b.dataset.id; const rr=await fetch('/api/templates'); const list=await rr.json(); const tpl=list.find(x=>x.id===id); if(tpl) inputEl.value = tpl.prompt; }); tplListEl.querySelectorAll('.tpl_del').forEach(b=>b.onclick=async ()=>{ if(!confirm('删除模板？')) return; await fetch('/api/templates/'+b.dataset.id,{method:'DELETE'}); loadTemplates(); }); } }

      </script>
    </body>
    </html>
    """

    # ---- 后端 API：会话 & 模板 ----
    @app.get('/api/sessions')
    async def api_sessions_list():
        return JSONResponse(session_list())

    @app.post('/api/sessions')
    async def api_sessions_create(body: Dict[str, Any]):
        title = body.get('title','新对话')
        model = body.get('model') or current_model
        system = body.get('system','')
        sid = new_session(title, model, system)
        return JSONResponse({'id': sid})

    @app.get('/api/sessions/{sid}/messages')
    async def api_sessions_messages(sid: str):
        return JSONResponse(messages_get(sid))

    @app.delete('/api/sessions/{sid}')
    async def api_sessions_delete(sid: str):
        session_delete(sid)
        return JSONResponse({'ok': True})

    @app.post('/api/sessions/{sid}/rename')
    async def api_sessions_rename(sid: str, body: Dict[str, Any]):
        new_title = body.get('title')
        if not new_title:
            return JSONResponse({'error': 'missing title'}, status_code=400)
        with db_conn() as c:
            c.execute('UPDATE sessions SET title=? WHERE id=?', (new_title, sid)); c.commit()
        return JSONResponse({'ok': True})

    @app.get('/api/sessions/{sid}/export')
    async def api_sessions_export(sid: str):
        s = session_get(sid)
        if not s: return JSONResponse({'error':'not found'}, status_code=404)
        payload = {'session': dict(s), 'messages': messages_get(sid)}
        fn = f'/tmp/session_{sid}.json'
        Path(fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return FileResponse(fn, media_type='application/json', filename=f'session_{sid}.json')

    @app.post('/api/sessions/import')
    async def api_sessions_import(file: UploadFile = File(...)):
        data = await file.read()
        try:
            payload = json.loads(data)
            s = payload.get('session')
            msgs = payload.get('messages', [])
            sid = s.get('id') if s and s.get('id') else uuid.uuid4().hex[:12]
            with db_conn() as c:
                c.execute("INSERT OR REPLACE INTO sessions (id,title,model,system,created_at) VALUES (?,?,?,?,?)", (sid, s.get('title','imported'), s.get('model', current_model), s.get('system',''), int(s.get('created_at', now_ts()))))
                for m in msgs:
                    c.execute("INSERT INTO messages (session_id,role,content,ts) VALUES (?,?,?,?)", (sid, m.get('role'), m.get('content'), int(m.get('ts', now_ts()))))
                c.commit()
            return JSONResponse({'ok': True, 'id': sid})
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=400)

    # templates endpoints
    @app.post('/api/templates')
    async def api_templates_create(body: Dict[str, Any]):
        name = body.get('name') or 'tpl'
        prompt = body.get('prompt') or ''
        tid = save_template(name, prompt)
        return JSONResponse({'id': tid})

    @app.get('/api/templates')
    async def api_templates_list():
        return JSONResponse(list_templates())

    @app.delete('/api/templates/{tid}')
    async def api_templates_delete(tid: str):
        delete_template(tid)
        return JSONResponse({'ok': True})

    # ---- 其他已有路由： models download/reload/upload/delete/list, chat endpoints, logs ----
    @app.post('/api/models/download')
    async def api_models_download(body: Dict[str, Any], authorization: Optional[str] = Header(None)):
        if ADMIN_TOKEN and not check_admin_token(authorization): return JSONResponse({'error':'unauthorized'}, status_code=401)
        repo = body.get('repo_id')
        rev = body.get('revision')
        if not repo: return JSONResponse({'error':'missing repo_id'}, status_code=400)
        try:
            local = hf_download_to_models_dir(repo, rev)
            return JSONResponse({'ok':True,'local_path':local})
        except Exception as e:
            return JSONResponse({'error':str(e)}, status_code=500)

    @app.post('/api/models/upload')
    async def api_models_upload(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
        if ADMIN_TOKEN and not check_admin_token(authorization): return JSONResponse({'error':'unauthorized'}, status_code=401)
        dest = Path(MODELS_DIR)
        name = Path(file.filename).stem
        target = dest / name
        if target.exists(): return JSONResponse({'error':'model exists'}, status_code=400)
        target.mkdir(parents=True, exist_ok=False)
        data = await file.read()
        tmp = dest / (file.filename)
        tmp.write_bytes(data)
        try:
            if str(tmp).endswith('.zip'):
                shutil.unpack_archive(str(tmp), str(target))
            else:
                import tarfile
                with tarfile.open(str(tmp)) as tf:
                    tf.extractall(path=str(target))
            tmp.unlink()
        except Exception as e:
            return JSONResponse({'error':f'extract error: {e}'}, status_code=500)
        return JSONResponse({'ok':True,'path':str(target)})

    @app.get('/api/models/list')
    async def api_models_list():
        local=[]
        for p in Path(MODELS_DIR).iterdir():
            if p.is_dir(): local.append({'id':p.name,'path':str(p)})
        online=[]
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{VLLM_BASE()}/v1/models", timeout=5) as r:
                    if r.status==200:
                        j=await r.json(); online=j.get('data',[])
        except Exception:
            pass
        return JSONResponse({'local':local,'online':online})

    @app.post('/api/models/reload')
    async def api_models_reload(body: Dict[str, Any], authorization: Optional[str] = Header(None)):
        if ADMIN_TOKEN and not check_admin_token(authorization): return JSONResponse({'error':'unauthorized'}, status_code=401)
        model = body.get('model')
        rev = body.get('revision')
        if not model: return JSONResponse({'error':'missing model'}, status_code=400)
        new_port = max(list(vllm_procs.keys())+[BASE_VLLM_PORT]) + 1
        start_vllm_on_port(new_port, model, rev)
        try:
            await wait_vllm_ready_port(new_port, timeout_s=600)
        except Exception as e:
            stop_vllm_port(new_port)
            return JSONResponse({'error':f'new vllm not ready: {e}'}, status_code=500)
        nonlocal active_port
        old_port = active_port
        active_port = new_port
        stop_vllm_port(old_port)
        return JSONResponse({'ok':True,'active_port':active_port})

    @app.post('/api/models/delete')
    async def api_models_delete(body: Dict[str, Any], authorization: Optional[str] = Header(None)):
        if ADMIN_TOKEN and not check_admin_token(authorization): return JSONResponse({'error':'unauthorized'}, status_code=401)
        mid = body.get('model')
        if not mid: return JSONResponse({'error':'missing model'}, status_code=400)
        target = Path(MODELS_DIR)/mid
        if not target.exists(): return JSONResponse({'error':'not found'}, status_code=404)
        try:
            shutil.rmtree(target)
            return JSONResponse({'ok':True})
        except Exception as e:
            return JSONResponse({'error':str(e)}, status_code=500)

    @app.get('/api/vllm/log')
    async def tail_log(lines: int = 200, authorization: Optional[str] = Header(None)):
        if ADMIN_TOKEN and not check_admin_token(authorization): return JSONResponse({'error':'unauthorized'}, status_code=401)
        p = Path(LOG_PATH)
        if not p.exists(): return PlainTextResponse('')
        data = p.read_text(encoding='utf-8', errors='ignore').splitlines()[-lines:]
        return PlainTextResponse('
'.join(data))

    # Chat streaming (SSE JSON events)
    @app.post('/api/chat/stream')
    async def chat_stream(body: Dict[str, Any]):
        sid = body.get('session_id') or new_session()
        # append user messages
        for m in body.get('messages', []):
            if m.get('role')=='user': message_add(sid,'user',m.get('content',''))
        model = body.get('model') or current_model
        async def sse_gen() -> AsyncGenerator[str, None]:
            payload = {'model':model,'messages':messages_get(sid),'temperature':body.get('temperature',0.7),'max_tokens':body.get('max_tokens',512),'stream':True}
            url = f"{VLLM_BASE()}/v1/chat/completions"
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(url,json=payload,timeout=0) as resp:
                        async for raw,_ in resp.content.iter_chunks():
                            if not raw: continue
                            text = raw.decode('utf-8',errors='ignore')
                            for line in text.splitlines():
                                if not line.startswith('data:'): continue
                                data = line[5:].strip()
                                if data == '[DONE]':
                                    yield f'data: {json.dumps({"type":"done"}, ensure_ascii=False)}

'
                                    return
                                try:
                                    evt = json.loads(data)
                                except Exception:
                                    continue
                                choice = (evt.get('choices') or [{}])[0]
                                delta = choice.get('delta') or {}
                                token = delta.get('content')
                                if token:
                                    yield f'data: {json.dumps({"type":"token","token":token}, ensure_ascii=False)}

'
                                tc = delta.get('tool_calls')
                                if tc:
                                    yield f'data: {json.dumps({"type":"tool","tool":tc}, ensure_ascii=False)}

'
            except asyncio.CancelledError:
                return
            except Exception as e:
                yield f'data: {json.dumps({"type":"error","error":str(e)}, ensure_ascii=False)}

'
        return StreamingResponse(sse_gen(), media_type='text/event-stream; charset=utf-8')

    @app.post('/api/chat')
    async def chat_once(body: Dict[str, Any]):
        sid = body.get('session_id') or new_session()
        for m in body.get('messages', []):
            if m.get('role') == 'user': message_add(sid,'user',m.get('content',''))
        payload = {'model': body.get('model') or current_model,'messages': messages_get(sid),'temperature': body.get('temperature',0.7),'max_tokens': body.get('max_tokens',512),'stream': False}
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{VLLM_BASE()}/v1/chat/completions", json=payload, timeout=1200) as r:
                data = await r.json()
                if r.status!=200: return JSONResponse({'error':data}, status_code=r.status)
                reply = (data.get('choices') or [{}])[0].get('message',{}).get('content','')
                if reply: message_add(sid,'assistant',reply)
                return JSONResponse({'reply':reply,'session_id':sid,'raw':data})

    # models list local
    @app.get('/api/models/list_local')
    async def api_models_list_local():
        res=[]
        for p in Path(MODELS_DIR).iterdir():
            if p.is_dir(): res.append({'id':p.name,'path':str(p)})
        return JSONResponse(res)

    return app


# -------------------- 部署入口 --------------------
if __name__ == '__main__':
    app.deploy()
