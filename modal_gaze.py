import os
import sys
import subprocess
from pathlib import Path

import modal

app = modal.App("sibem-gaze")

vol = modal.Volume.from_name("sibem-gaze-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsndfile1",
        "build-essential",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "pydantic",
        "python-multipart",
        "opencv-python-headless",
        "numpy<2.2",
        "scipy",
        "librosa",
        "soundfile",
        "scikit-image",
        "filetype",
        "imageio",
        "imageio-ffmpeg",
        "mediapipe==0.10.14",
        "onnxruntime-gpu==1.20.2",
        "huggingface_hub",
        "cython",
    )
    .pip_install("einops", "tqdm")
    .env(
        {
            "DITTO_ENGINE_DIR": "/app/ditto_engine",
            "CHECKPOINTS_DIR": "/cache/checkpoints",
            "CFG_PKL": "/cache/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
            "DATA_ROOT": "/cache/checkpoints/ditto_pytorch",
            "DEFAULT_PORTRAIT": "/app/public/sibem.jpg",
            "GAZE_FPS_TARGET": "20",
            "STREAM_MAX_DIM": "576",
            "STREAM_JPEG_QUALITY": "68",
            "DITTO_SAMPLING_STEPS": "4",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir("ditto_engine", remote_path="/app/ditto_engine")
    .add_local_file("gaze_worker.py", remote_path="/app/gaze_worker.py")
    .add_local_file("public/sibem.jpg", remote_path="/app/public/sibem.jpg")
    .add_local_dir("public/avatars", remote_path="/app/public/avatars")
)


web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
    )
    .add_local_file("package.json", "/app/package.json", copy=True)
    .add_local_file("package-lock.json", "/app/package-lock.json", copy=True)
    .run_commands("cd /app && npm ci --omit=dev")
    .add_local_dir("dist", "/app/dist")
    .add_local_dir("knowledge", "/app/knowledge")
    .add_local_file("public/sibem.jpg", "/app/public/sibem.jpg")
    .add_local_file("public/avatars/sibem.jpg", "/app/public/avatars/sibem.jpg")
    .add_local_file("server.js", "/app/server.js")
)


def _ensure_checkpoints():
    ckpt_marker = Path("/cache/checkpoints/ditto_pytorch/models/warp_network.pth")
    cfg_marker = Path("/cache/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    if ckpt_marker.is_file() and cfg_marker.is_file():
        return
    from huggingface_hub import snapshot_download

    Path("/cache/checkpoints").mkdir(parents=True, exist_ok=True)
    print("[modal_gaze] Downloading digital-avatar/ditto-talkinghead checkpoints...")
    snapshot_download(
        repo_id="digital-avatar/ditto-talkinghead",
        local_dir="/cache/checkpoints",
        allow_patterns=["ditto_cfg/*", "ditto_pytorch/*"],
    )
    vol.commit()
    print("[modal_gaze] Checkpoints committed to Modal Volume.")


@app.function(
    image=image,
    gpu="L40S",
    memory=32768,
    timeout=3600,
    scaledown_window=35,
    min_containers=0,
    max_containers=4,
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(8766, startup_timeout=600, label="sibem-gaze")
def serve():
    _ensure_checkpoints()

    # Expose PyTorch & NVIDIA CUDA 12 libs so onnxruntime-gpu accelerates HuBERT & landmarks cleanly.
    import torch

    site_pkg = Path(torch.__file__).resolve().parent.parent
    lib_dirs = [str(Path(torch.__file__).resolve().parent / "lib")]
    nvidia_root = site_pkg / "nvidia"
    if nvidia_root.is_dir():
        for lib_sub in nvidia_root.glob("*/lib"):
            lib_dirs.append(str(lib_sub))

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONPATH"] = "/app"

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gaze_worker:app",
            "--app-dir",
            "/app",
            "--host",
            "0.0.0.0",
            "--port",
            "8766",
            "--ws-ping-interval",
            "10",
            "--ws-ping-timeout",
            "15",
            "--log-level",
            "info",
        ],
        env=env,
        cwd="/app",
    )


@app.function(
    image=web_image,
    cpu=2.0,
    memory=2048,
    timeout=3600,
    scaledown_window=180,
    min_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("sibem-nebius")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(3005, startup_timeout=60, label="sibem")
def web():
    env = os.environ.copy()
    env["PORT"] = "3005"
    env.setdefault("GAZE_WORKER_URL", "https://nextmarte--sibem-gaze.us-east.modal.run")
    subprocess.Popen(
        ["node", "/app/server.js"],
        env=env,
        cwd="/app",
    )
