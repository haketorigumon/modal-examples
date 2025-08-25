import json
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/nightly",
    )
)


MODEL_NAME = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "6f42c8b5bf8f3f687bd6fb28833da03a19867ce8"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


MAX_INPUTS = 32  # how many requests can one replica handle? tune carefully!
CUDA_GRAPH_CAPTURE_SIZES = [  # 1, 2, 4, ... MAX_INPUTS
    1 << i for i in range((MAX_INPUTS).bit_length())
]


app = modal.App("inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 4321


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=30 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)

if __name__ == "__main__":
    app.deploy()
