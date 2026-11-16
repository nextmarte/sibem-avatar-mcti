import os
import sys
import io
import time
import copy
import math
import uuid
import queue
import base64
import asyncio
import threading
import subprocess
import contextlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent
DITTO_DIR = Path(os.environ.get("DITTO_DIR", str(ROOT_DIR.parent / "talking-avatar-ditto")))
_default_engine_dir = (ROOT_DIR / "ditto_engine") if (ROOT_DIR / "ditto_engine").is_dir() else (DITTO_DIR / "ditto_engine")
DITTO_ENGINE_DIR = Path(os.environ.get("DITTO_ENGINE_DIR", str(_default_engine_dir)))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", str(DITTO_DIR / "checkpoints")))
CFG_PKL = Path(os.environ.get("CFG_PKL", str(CHECKPOINTS_DIR / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CHECKPOINTS_DIR / "ditto_pytorch")))
DEFAULT_PORTRAIT = Path(os.environ.get("DEFAULT_PORTRAIT", str(ROOT_DIR / "public" / "sibem.jpg")))

if str(DITTO_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(DITTO_ENGINE_DIR))

from stream_pipeline_offline import StreamSDK
from core.atomic_components.motion_stitch import (
    _mix_s_d_info,
    _eye_delta,
    ctrl_motion,
    fade,
    transform_keypoint,
)
from core.utils.get_mask import get_mask


def cv2_to_torch_theta(M_2x3_or_3x3: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Convert OpenCV affine matrix (src -> dst) into PyTorch affine_grid theta (align_corners=False)."""
    M3 = np.eye(3, dtype=np.float64)
    M3[:2, :] = M_2x3_or_3x3[:2, :]
    M_inv = np.linalg.inv(M3)
    sh, sw = src_hw
    dh, dw = dst_hw
    N_dst = np.array(
        [[dw / 2.0, 0.0, (dw - 1.0) / 2.0], [0.0, dh / 2.0, (dh - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    N_src_inv = np.linalg.inv(
        np.array(
            [[sw / 2.0, 0.0, (sw - 1.0) / 2.0], [0.0, sh / 2.0, (sh - 1.0) / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    )
    T = N_src_inv @ M_inv @ N_dst
    return T[:2, :].astype(np.float32)


def decode_audio_bytes_to_16k_f32(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary MP3/WAV/OGG audio bytes into 16kHz mono float32 waveform via ffmpeg pipe."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return wav


class GazeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.target_x = 0.0  # -1 (left on screen) .. +1 (right on screen)
        self.target_y = 0.0  # -1 (top) .. +1 (bottom)
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.detected = False
        self.source = "idle"
        self.last_update = 0.0
        self.last_t = 0.0

    def update(self, x: float, y: float, detected: bool, source: str = "webcam"):
        with self.lock:
            self.target_x = float(np.clip(x, -1.0, 1.0))
            self.target_y = float(np.clip(y, -1.0, 1.0))
            self.detected = bool(detected)
            self.source = source
            self.last_update = time.perf_counter()

    def reset(self):
        with self.lock:
            self.target_x = 0.0
            self.target_y = 0.0
            self.detected = False
            self.source = "reset"
            self.last_update = time.perf_counter()

    def step(self, t_now: float) -> tuple[float, float, bool]:
        with self.lock:
            if self.detected and (t_now - self.last_update) > 2.2:
                self.detected = False
                self.target_x = 0.0
                self.target_y = 0.0
            elif not self.detected:
                self.target_x = 0.0
                self.target_y = 0.0

            # Scale per-frame alpha so physical gaze velocity matches 10 FPS baseline
            dt = max(0.01, min(0.20, t_now - self.last_t)) if self.last_t > 0.0 else 0.05
            self.last_t = t_now
            base_alpha = 0.24 if self.detected else 0.12
            alpha = 1.0 - math.pow(1.0 - base_alpha, dt / 0.10)
            self.cur_x += (self.target_x - self.cur_x) * alpha
            self.cur_y += (self.target_y - self.cur_y) * alpha
            return self.cur_x, self.cur_y, self.detected


def gaze_to_motion_params(nx: float, ny: float, t_sec: float, tracked: bool) -> dict:
    """
    Map normalized viewer coordinates (nx in [-1,1], ny in [-1,1]) to safe Ditto pose & pupil offsets.
    For Sibem portrait:
      - Negative nx (viewer-left) -> positive delta_yaw (up to +11.5 deg), eye_dx > 0
      - Positive nx (viewer-right) -> negative delta_yaw (down to -9.0 deg), eye_dx < 0
      - Negative ny (top) -> negative delta_pitch (chin up -7.5 deg), eye_dy < 0
      - Positive ny (bottom) -> positive delta_pitch (chin down +6.5 deg), eye_dy > 0
    """
    micro_yaw = math.sin(t_sec * 0.75) * (0.25 if tracked else 0.7)
    micro_pitch = math.cos(t_sec * 0.95) * (0.2 if tracked else 0.5)
    micro_roll = math.sin(t_sec * 0.55) * (0.3 if tracked else 0.65)

    yaw_head = (-nx * 11.5 if nx < 0 else -nx * 9.0) + micro_yaw
    pitch_head = (ny * 7.5 if ny < 0 else ny * 6.5) + micro_pitch
    roll_head = (nx * 2.2) + micro_roll

    # Eye pupils lead slightly more strongly than head turn
    eye_dx = -nx * 12.0
    eye_dy = ny * 7.5

    return {
        "delta_yaw": float(np.clip(yaw_head, -9.5, 12.0)),
        "delta_pitch": float(np.clip(pitch_head, -8.0, 7.0)),
        "delta_roll": float(np.clip(roll_head, -3.5, 3.5)),
        "eye_dx": float(eye_dx),
        "eye_dy": float(eye_dy),
    }


class ReactionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.kind = "none"
        self.start_at = 0.0
        self.duration = 0.0
        self.strength = 1.0
        self.smile_level = 0.0
        self.smile_target = 0.0
        self.smile_until = 0.0
        self.last_t = 0.0

    def trigger(self, kind: str, strength: float = 1.0, duration: float = 2.0):
        t_now = time.perf_counter()
        with self.lock:
            self.kind = str(kind or "none")
            self.start_at = t_now
            self.duration = float(duration)
            self.strength = float(np.clip(strength, 0.2, 1.5))
            if self.kind in ("smile", "thumbs_up", "wave", "victory", "head_yes", "nod"):
                self.smile_target = min(1.0, 0.75 * self.strength)
                self.smile_until = t_now + duration

    def set_mirror_smile(self, level: float):
        t_now = time.perf_counter()
        with self.lock:
            val = float(np.clip(level, 0.0, 1.0))
            if val > 0.05:
                self.smile_target = val
                self.smile_until = t_now + 0.85

    def step(self, t_now: float) -> dict:
        with self.lock:
            dt = max(0.01, min(0.20, t_now - self.last_t)) if self.last_t > 0.0 else 0.05
            self.last_t = t_now
            if t_now > self.smile_until:
                self.smile_target = 0.0
            smile_alpha = 1.0 - math.pow(1.0 - 0.18, dt / 0.10)
            self.smile_level += (self.smile_target - self.smile_level) * smile_alpha

            active_kind = "none"
            nod_pitch = 0.0
            nod_yaw = 0.0
            nod_roll = 0.0
            if self.kind != "none" and self.duration > 0.0:
                u = (t_now - self.start_at) / self.duration
                if 0.0 <= u <= 1.0:
                    active_kind = self.kind
                    env = math.sin(math.pi * u)
                    if self.kind in ("head_yes", "thumbs_up", "nod", "victory"):
                        # Clear vertical affirmative double/triple nod (pitch up & down)
                        amp = 8.5 if self.kind in ("head_yes", "nod") else 5.2
                        nod_pitch = math.sin(u * math.pi * 4.5) * amp * env * self.strength
                        nod_roll = math.sin(u * math.pi * 2.0) * 1.6 * env
                    elif self.kind == "wave":
                        nod_yaw = math.sin(u * math.pi * 3.0) * 4.5 * env
                        nod_pitch = math.sin(u * math.pi * 2.0) * 3.0 * env
                        nod_roll = 3.2 * env
                    elif self.kind in ("head_no", "thumbs_down", "shake"):
                        # Clear horizontal negative head shake (yaw left & right)
                        amp = 11.0 if self.kind in ("head_no", "shake") else 6.5
                        nod_yaw = math.sin(u * math.pi * 5.0) * amp * env * self.strength
                        nod_roll = math.sin(u * math.pi * 2.5) * 2.2 * env
                        nod_pitch = 1.8 * env
                else:
                    self.kind = "none"

            return {
                "kind": active_kind,
                "smile": float(self.smile_level),
                "nod_pitch": float(nod_pitch),
                "nod_yaw": float(nod_yaw),
                "nod_roll": float(nod_roll),
            }


class BlinkState:
    def __init__(self):
        self.started_at = 0.0
        self.next_at = time.perf_counter() + float(np.random.uniform(1.5, 3.0))
        self.double_pending = False

    def trigger(self):
        t_now = time.perf_counter()
        if t_now - self.started_at > 0.55:
            self.started_at = t_now
            self.next_at = t_now + float(np.random.uniform(2.6, 4.4))


class SessionState:
    """Isolated per-visitor avatar state so multiple simultaneous users never cross-talk."""

    def __init__(self):
        self.gaze = GazeState()
        self.reaction = ReactionState()
        self.blink = BlinkState()
        self.speech_queue: queue.Queue = queue.Queue()
        self.active_speech_cancel = threading.Event()
        self.speech_gen = 0


class SibemGazeEngine:
    def __init__(self):
        self.gpu_lock = threading.Lock()
        self.motion_lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.gaze = GazeState()
        self.reaction = ReactionState()
        self.speech_queue: queue.Queue = queue.Queue()
        self.active_speech_cancel = threading.Event()
        self.ready = False
        self.portrait_path = str(DEFAULT_PORTRAIT)
        self.avatar_id = "sibem"
        self.portrait_cache: dict[str, dict] = {}
        self.width = 768
        self.height = 768
        self.fps_target = float(os.environ.get("GAZE_FPS_TARGET", "24.0"))
        self.device_name = "CPU"
        self.render_stream = None
        self._blink_started_at = 0.0
        self._next_blink_at = 0.0
        self._double_blink_pending = False
        self.eye_blink_mask = None
        self.smile_exp_delta = None

    def load(self):
        self.sdk = StreamSDK(str(CFG_PKL), str(DATA_ROOT))
        self.dk = self.sdk.default_kwargs
        self.warp_mod = self.sdk.warp_f3d.warp_net.model.half().eval()
        self.decode_mod = self.sdk.decode_f3d.decoder.model.half().eval()
        self.stitch_net = self.sdk.motion_stitch.stitch_net
        self.delta_eye_arr = self.dk.get("delta_eye_arr")

        # Build explicit eyelid keypoint mask ([11, 13, 15, 16, 18]); dk does not populate fix_exp_a3 on its own.
        eye_mask = np.zeros((21, 3), dtype=np.float32)
        eye_mask[[11, 13, 15, 16, 18]] = 1.0
        self.eye_blink_mask = eye_mask.reshape(1, -1)

        # Gentle closed-mouth warm smile deformation across mouth corners [6, 12] and cheeks [14, 17]
        smile = np.zeros((21, 3), dtype=np.float32)
        smile[6, 0] = 0.0045
        smile[6, 1] = -0.0065
        smile[12, 0] = -0.0045
        smile[12, 1] = -0.0065
        smile[14, 1] = -0.0025
        smile[17, 1] = -0.0025
        self.smile_exp_delta = smile.reshape(1, -1)

        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)
            self.render_stream = torch.cuda.Stream()

        # Configure Audio2Motion for fast low-latency diffusion
        self.sampling_steps = int(os.environ.get("DITTO_SAMPLING_STEPS", "4"))
        self.sdk.audio2motion.lmdm.setup(self.sampling_steps)

        mask_np = get_mask(512, 512, 0.9, 0.9)  # [512, 512, 1] float32
        self.mask_512_half = torch.from_numpy(mask_np).permute(2, 0, 1)[None].cuda().half()

        self._next_blink_at = time.perf_counter() + 1.8
        self.set_portrait(str(DEFAULT_PORTRAIT), avatar_id="sibem")
        if self.render_stream is not None:
            torch.cuda.synchronize()
        # Warmup CUDA graphs/kernels
        self._render_one_frame(copy.deepcopy(self.x_s_info), None, gaze_to_motion_params(0.0, 0.0, 0.0, False), 0.0)
        self.ready = True
        print(f"[gaze_worker] Ready on {self.device_name} | Portrait {self.width}x{self.height} | delta_eye={None if self.delta_eye_arr is None else len(self.delta_eye_arr)}")

    def _apply_cached_portrait(self, entry: dict):
        self.source_info = entry["source_info"]
        self.x_s_info = entry["x_s_info"]
        self.x_s_kp = entry["x_s_kp"]
        self.x_s_t = entry["x_s_t"]
        self.f_s_t = entry["f_s_t"]
        self.grid_c2o_half = entry["grid_c2o_half"]
        self.mask_warped_half = entry["mask_warped_half"]
        self.inv_mask_bg_half = entry["inv_mask_bg_half"]
        self.width = entry["width"]
        self.height = entry["height"]
        self.portrait_path = entry["portrait_path"]
        self.avatar_id = entry["avatar_id"]

    def set_portrait(self, image_path: str, avatar_id: Optional[str] = None, use_cache: bool = True):
        cache_key = avatar_id or image_path
        with self.motion_lock, self.render_lock:
            if use_cache and cache_key in self.portrait_cache:
                self._apply_cached_portrait(self.portrait_cache[cache_key])
                return

            info = self.sdk.avatar_registrar(image_path, max_dim=768, crop_scale=2.3, crop_vx_ratio=0, crop_vy_ratio=-0.125)
            rgb = info["img_rgb_lst"][0]
            h, w = rgb.shape[:2]
            M = info["M_c2o_lst"][0]

            theta = cv2_to_torch_theta(M, (512, 512), (h, w))
            theta_t = torch.from_numpy(theta)[None].cuda()
            grid = F.affine_grid(theta_t, size=(1, 3, h, w), align_corners=False)

            bg_half = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda().half()
            mask_warped = F.grid_sample(
                self.mask_512_half, grid.to(dtype=self.mask_512_half.dtype), mode="bilinear", padding_mode="zeros", align_corners=False
            ).clamp_(0.0, 1.0)
            inv_mask_bg = (1.0 - mask_warped) * bg_half

            x_s_info = info["x_s_info_lst"][0]
            x_s_kp = transform_keypoint(x_s_info)
            entry = {
                "source_info": info,
                "x_s_info": x_s_info,
                "x_s_kp": x_s_kp,
                "x_s_t": torch.from_numpy(x_s_kp).cuda().half(),
                "f_s_t": torch.from_numpy(info["f_s_lst"][0]).cuda().half(),
                "grid_c2o_half": grid.half(),
                "mask_warped_half": mask_warped,
                "inv_mask_bg_half": inv_mask_bg,
                "width": w,
                "height": h,
                "portrait_path": image_path,
                "avatar_id": cache_key,
            }
            if use_cache:
                self.portrait_cache[cache_key] = entry
            self._apply_cached_portrait(entry)

    def prepare_speech(self, audio_bytes: bytes, speech_id: Optional[str] = None) -> dict:
        wav = decode_audio_bytes_to_16k_f32(audio_bytes)
        n_samples = len(wav)
        n_frames_audio = max(1, int(round(n_samples / 640.0)))
        aud_feat = self.sdk.wav2feat.wav2feat(wav, sr=16000)

        with self.motion_lock:
            self.sdk.condition_handler.setup(self.source_info, emo=4, eye_f0_mode=False)
            self.sdk.audio2motion.setup(
                self.x_s_info,
                overlap_v2=10,
                fix_kp_cond=1,
                fix_kp_cond_dim=(0, 202),
                sampling_timesteps=self.sampling_steps,
                online_mode=False,
                v_min_max_for_clip=self.dk.get("v_min_max_for_clip"),
                smo_k_d=3,
            )

            seq_frames = self.sdk.audio2motion.seq_frames  # 80
            valid_clip_len = self.sdk.audio2motion.valid_clip_len  # 70

            res_kp_seq = None
            res_start = 0
            local_idx = 0
            out_info_list = []

            if len(aud_feat) == 0:
                aud_feat = np.zeros((1, self.sdk.wav2feat.feat_dim), dtype=np.float32)

            while local_idx < len(aud_feat):
                chunk = aud_feat[local_idx : local_idx + seq_frames]
                real_len = valid_clip_len
                if len(chunk) < seq_frames:
                    real_len = len(chunk)
                    pad = np.repeat(chunk[-1:], seq_frames - len(chunk), axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)

                aud_cond = self.sdk.condition_handler(chunk, local_idx)
                res_kp_seq = self.sdk.audio2motion(aud_cond, res_kp_seq)
                valid_seq = res_kp_seq[:, res_start : res_start + real_len]
                out_info_list.extend(self.sdk.audio2motion.cvt_fmt(valid_seq))
                res_start += real_len
                local_idx += real_len

            if len(out_info_list) > n_frames_audio:
                out_info_list = out_info_list[:n_frames_audio]

            d0 = copy.deepcopy(out_info_list[0]) if out_info_list else copy.deepcopy(self.x_s_info)

        return {
            "speechId": speech_id or str(uuid.uuid4())[:8],
            "x_d_info_list": out_info_list,
            "d0": d0,
            "durationSec": n_samples / 16000.0,
            "audioBase64": base64.b64encode(audio_bytes).decode("ascii"),
        }

    def trigger_blink(self, blink_state: Optional[BlinkState] = None):
        if blink_state is not None:
            blink_state.trigger()
            return
        t_now = time.perf_counter()
        if t_now - self._blink_started_at > 0.55:
            self._blink_started_at = t_now
            self._next_blink_at = t_now + float(np.random.uniform(2.6, 4.4))

    def _blink_delta_exp(self, t_sec: float, blink_state: Optional[BlinkState] = None):
        if self.delta_eye_arr is None or len(self.delta_eye_arr) == 0:
            return None
        if blink_state is not None:
            if t_sec >= blink_state.next_at:
                blink_state.started_at = t_sec
                if blink_state.double_pending:
                    blink_state.double_pending = False
                    blink_state.next_at = t_sec + float(np.random.uniform(2.8, 4.6))
                else:
                    do_double = np.random.random() < 0.22
                    blink_state.double_pending = do_double
                    blink_state.next_at = t_sec + (0.48 if do_double else float(np.random.uniform(2.5, 4.5)))
            blink_dur = 0.36
            elapsed = t_sec - blink_state.started_at
        else:
            if t_sec >= self._next_blink_at:
                self._blink_started_at = t_sec
                if self._double_blink_pending:
                    self._double_blink_pending = False
                    self._next_blink_at = t_sec + float(np.random.uniform(2.8, 4.6))
                else:
                    do_double = np.random.random() < 0.22
                    self._double_blink_pending = do_double
                    self._next_blink_at = t_sec + (0.48 if do_double else float(np.random.uniform(2.5, 4.5)))

            blink_dur = 0.36
            elapsed = t_sec - self._blink_started_at
        if elapsed < 0.0 or elapsed >= blink_dur:
            return None
        # Focus on active blink frames [4..12] (peak eyelid closure at 6..7) with slight boost for crisp visibility
        active_slice = self.delta_eye_arr[4:13]
        idx = min(len(active_slice) - 1, int((elapsed / blink_dur) * len(active_slice)))
        return active_slice[idx][None] * 1.15

    def _render_one_frame(
        self,
        x_d_base: dict,
        d0: Optional[dict],
        gaze_params: dict,
        t_sec: float,
        fade_alpha: float = 1.0,
        react: Optional[dict] = None,
        blink_state: Optional[BlinkState] = None,
    ) -> bytes:
        with self.render_lock:
            if d0 is not None:
                use_keys = self.dk.get("use_d_keys") or {"exp": 1.0, "pitch": 0.4, "yaw": 0.2, "roll": 0.4, "t": 0.3}
                x_d = _mix_s_d_info(
                    self.x_s_info,
                    x_d_base,
                    use_keys,
                    d0,
                )
            else:
                x_d = copy.deepcopy(x_d_base)

            if fade_alpha < 1.0:
                x_d = fade(x_d, self.x_s_info, fade_alpha, keys=("exp", "pitch", "yaw", "roll", "t"))

            # Natural blinking on eye keypoints [11, 13, 15, 16, 18]
            delta_eye = self._blink_delta_exp(t_sec, blink_state=blink_state)
            if delta_eye is not None and self.eye_blink_mask is not None:
                x_d["exp"] = x_d["exp"] + delta_eye * self.eye_blink_mask

            if react and react.get("smile", 0.0) > 0.02 and self.smile_exp_delta is not None:
                # Apply gentler mouth-corner warmth while speaking so speech articulation stays clear
                smile_scale = react["smile"] * (0.45 if d0 is not None else 1.0)
                x_d["exp"] = x_d["exp"] + self.smile_exp_delta * smile_scale

            nod_p = react["nod_pitch"] if react else 0.0
            nod_y = react["nod_yaw"] if react else 0.0
            nod_r = react["nod_roll"] if react else 0.0

            x_d = ctrl_motion(
                x_d,
                delta_yaw=float(np.clip(gaze_params["delta_yaw"] + nod_y, -13.0, 14.0)),
                delta_pitch=float(np.clip(gaze_params["delta_pitch"] + nod_p, -11.0, 10.5)),
                delta_roll=float(np.clip(gaze_params["delta_roll"] + nod_r, -7.5, 7.5)),
            )
            x_d["exp"] = _eye_delta(
                x_d["exp"].copy(),
                dx=gaze_params["eye_dx"],
                dy=gaze_params["eye_dy"],
            )

            x_d_kp = transform_keypoint(x_d)
            x_d_kp = self.stitch_net(self.x_s_kp, x_d_kp)

            stream_ctx = (
                torch.cuda.stream(self.render_stream)
                if self.render_stream is not None
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), stream_ctx:
                x_d_t = torch.from_numpy(x_d_kp).cuda().half()
                f3d = self.warp_mod(self.f_s_t, self.x_s_t, x_d_t)
                pred = self.decode_mod(f3d).clamp_(0.0, 1.0)
                warped = F.grid_sample(
                    pred, self.grid_c2o_half, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                out_t = (warped * self.mask_warped_half + self.inv_mask_bg_half).clamp_(0.0, 1.0)
                max_dim = int(os.environ.get("STREAM_MAX_DIM", "540"))
                if max_dim > 0 and max(out_t.shape[2], out_t.shape[3]) > max_dim:
                    out_t = F.interpolate(out_t, size=(max_dim, max_dim), mode="bilinear", align_corners=False)
                out_u8 = (out_t[0].permute(1, 2, 0) * 255.0).byte().cpu().numpy()

        bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        jpeg_quality = int(os.environ.get("STREAM_JPEG_QUALITY", "65"))
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def render_frame_snapshot(self) -> tuple[bytes, dict]:
        t_now = time.perf_counter()
        nx, ny, tracked = self.gaze.step(t_now)
        gp = gaze_to_motion_params(nx, ny, t_now, tracked)
        jpg = self._render_one_frame(self.x_s_info, None, gp, t_now, 1.0)
        return jpg, gp


engine = SibemGazeEngine()
ACTIVE_SESSIONS: dict[str, dict] = {}
app = FastAPI(title="Sibem Gaze Worker (Ditto GPU)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GazePayload(BaseModel):
    x: float = 0.0
    y: float = 0.0
    detected: bool = True
    source: str = "api"


class SpeakPayload(BaseModel):
    audioBase64: str
    speechId: Optional[str] = None
    sessionId: Optional[str] = None


class ReactionPayload(BaseModel):
    kind: str = "smile"
    strength: float = 1.0
    duration: float = 2.0
    sessionId: Optional[str] = None


class InterruptPayload(BaseModel):
    sessionId: Optional[str] = None


@app.on_event("startup")
def on_startup():
    engine.load()


@app.get("/health")
def health():
    return {
        "ok": engine.ready,
        "device": engine.device_name,
        "width": engine.width,
        "height": engine.height,
        "fpsTarget": engine.fps_target,
        "portrait": Path(engine.portrait_path).name,
        "avatarId": engine.avatar_id,
    }


@app.post("/gaze")
def post_gaze(payload: GazePayload):
    engine.gaze.update(payload.x, payload.y, payload.detected, payload.source)
    return {"ok": True}


@app.post("/reaction")
def post_reaction(payload: ReactionPayload):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Worker warming up")
    if payload.sessionId:
        entry = ACTIVE_SESSIONS.get(payload.sessionId)
        if not entry:
            return {"ok": True, "routed": False, "kind": payload.kind, "sessionId": payload.sessionId}
        target_sess = entry["sess"]
        if payload.kind == "blink":
            engine.trigger_blink(target_sess.blink)
        else:
            target_sess.reaction.trigger(payload.kind, payload.strength, payload.duration)
            if payload.kind in ("thumbs_up", "wave", "victory", "head_yes", "nod"):
                engine.trigger_blink(target_sess.blink)
        return {"ok": True, "routed": True, "kind": payload.kind, "sessionId": payload.sessionId}
    if payload.kind == "blink":
        engine.trigger_blink()
    else:
        engine.reaction.trigger(payload.kind, payload.strength, payload.duration)
        if payload.kind in ("thumbs_up", "wave", "victory", "head_yes", "nod"):
            engine.trigger_blink()
    return {"ok": True, "routed": True, "kind": payload.kind}


@app.post("/avatar/{avatar_id}")
def post_avatar(avatar_id: str):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Worker warming up")
    safe_id = "".join(ch for ch in avatar_id.lower() if ch.isalnum() or ch in ("-", "_"))
    candidate = ROOT_DIR / "public" / "avatars" / f"{safe_id}.jpg"
    if not candidate.is_file() and safe_id == "sibem":
        candidate = DEFAULT_PORTRAIT
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Avatar '{safe_id}' not found")
    engine.set_portrait(str(candidate), avatar_id=safe_id, use_cache=True)
    return {"ok": True, "avatarId": engine.avatar_id, "width": engine.width, "height": engine.height}


def _clear_session_speech(entry: dict):
    target_sess = entry["sess"]
    raw_q = entry["raw_speak_queue"]
    target_sess.speech_gen += 1
    target_sess.active_speech_cancel.set()
    while not raw_q.empty():
        try:
            raw_q.get_nowait()
            raw_q.task_done()
        except asyncio.QueueEmpty:
            break
    while not target_sess.speech_queue.empty():
        try:
            target_sess.speech_queue.get_nowait()
        except queue.Empty:
            break


@app.post("/speak")
def post_speak(payload: SpeakPayload):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Worker warming up")
    if payload.sessionId:
        entry = ACTIVE_SESSIONS.get(payload.sessionId)
        if not entry:
            return {"ok": True, "routed": False, "sessionId": payload.sessionId}
        target_sess = entry["sess"]
        target_sess.active_speech_cancel.clear()
        gen = target_sess.speech_gen
        entry["loop"].call_soon_threadsafe(
            entry["raw_speak_queue"].put_nowait,
            (payload.audioBase64, payload.speechId, gen),
        )
        return {"ok": True, "routed": True, "sessionId": payload.sessionId}
    audio_bytes = base64.b64decode(payload.audioBase64)
    job = engine.prepare_speech(audio_bytes, payload.speechId)
    engine.speech_queue.put(job)
    return {"ok": True, "routed": True, "speechId": job["speechId"], "durationSec": job["durationSec"]}


@app.post("/interrupt")
def post_interrupt(payload: Optional[InterruptPayload] = None):
    if payload and payload.sessionId:
        entry = ACTIVE_SESSIONS.get(payload.sessionId)
        if not entry:
            return {"ok": True, "routed": False, "sessionId": payload.sessionId}
        entry["loop"].call_soon_threadsafe(_clear_session_speech, entry)
        return {"ok": True, "routed": True, "sessionId": payload.sessionId}
    engine.active_speech_cancel.set()
    while not engine.speech_queue.empty():
        try:
            engine.speech_queue.get_nowait()
        except queue.Empty:
            break
    return {"ok": True, "routed": True}


@app.post("/portrait")
async def post_portrait(file: UploadFile = File(...)):
    data = await file.read()
    tmp_path = Path("/tmp") / "_custom_portrait.jpg" if os.name != "nt" else (ROOT_DIR / "public" / "_custom_portrait.jpg")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    cv2.imwrite(str(tmp_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    engine.set_portrait(str(tmp_path), avatar_id="custom", use_cache=False)
    return {"ok": True, "avatarId": "custom", "width": engine.width, "height": engine.height}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("sessionId") or ""
    sess = SessionState()
    stop_flag = asyncio.Event()
    raw_speak_queue: asyncio.Queue[tuple[str, Optional[str], int]] = asyncio.Queue()
    session_entry = {"sess": sess, "loop": asyncio.get_running_loop(), "raw_speak_queue": raw_speak_queue}
    if session_id:
        ACTIVE_SESSIONS[session_id] = session_entry
    await websocket.send_json(
        {
            "type": "ready",
            "width": engine.width,
            "height": engine.height,
            "fps": engine.fps_target,
            "device": engine.device_name,
            "avatarId": engine.avatar_id,
        }
    )

    async def speak_worker_loop():
        while not stop_flag.is_set():
            try:
                audio_b64, speech_id, gen = await raw_speak_queue.get()
                try:
                    if stop_flag.is_set() or gen != sess.speech_gen or sess.active_speech_cancel.is_set():
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    job = await asyncio.to_thread(engine.prepare_speech, audio_bytes, speech_id)
                    if not stop_flag.is_set() and gen == sess.speech_gen and not sess.active_speech_cancel.is_set():
                        sess.speech_queue.put(job)
                except Exception as exc:
                    print(f"[ws_stream speak error] {exc}")
                finally:
                    raw_speak_queue.task_done()
            except asyncio.CancelledError:
                break

    async def rx_loop():
        try:
            while not stop_flag.is_set():
                msg = await websocket.receive_json()
                mtype = msg.get("type")
                if mtype == "gaze":
                    sess.gaze.update(
                        float(msg.get("x", 0.0)),
                        float(msg.get("y", 0.0)),
                        bool(msg.get("detected", True)),
                        str(msg.get("source", "webcam")),
                    )
                elif mtype == "smile":
                    sess.reaction.set_mirror_smile(float(msg.get("level", 0.0)))
                elif mtype == "reaction":
                    kind = str(msg.get("kind", "smile"))
                    if kind == "blink":
                        engine.trigger_blink(sess.blink)
                    else:
                        sess.reaction.trigger(kind, float(msg.get("strength", 1.0)), float(msg.get("duration", 2.0)))
                        if kind in ("thumbs_up", "wave", "victory", "head_yes", "nod"):
                            engine.trigger_blink(sess.blink)
                elif mtype == "reset_gaze":
                    sess.gaze.reset()
                elif mtype == "interrupt":
                    sess.speech_gen += 1
                    sess.active_speech_cancel.set()
                    while not raw_speak_queue.empty():
                        try:
                            raw_speak_queue.get_nowait()
                            raw_speak_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    while not sess.speech_queue.empty():
                        try:
                            sess.speech_queue.get_nowait()
                        except queue.Empty:
                            break
                elif mtype == "speak":
                    audio_b64 = msg.get("audioBase64") or ""
                    if audio_b64:
                        sess.active_speech_cancel.clear()
                        await raw_speak_queue.put((audio_b64, msg.get("speechId"), sess.speech_gen))
        except (WebSocketDisconnect, Exception):
            stop_flag.set()

    async def tx_loop():
        active_job = None
        speech_t0 = 0.0
        last_gaze_ack = 0.0

        while not stop_flag.is_set():
            try:
                frame_t0 = time.perf_counter()

                # Check if a new speech job arrived when idle
                if active_job is None and not sess.speech_queue.empty():
                    try:
                        active_job = sess.speech_queue.get_nowait()
                        sess.active_speech_cancel.clear()
                        speech_t0 = frame_t0
                        await websocket.send_json(
                            {
                                "type": "speech_start",
                                "speechId": active_job["speechId"],
                                "durationSec": active_job["durationSec"],
                                "audioBase64": active_job["audioBase64"],
                                "mime": "audio/mpeg",
                            }
                        )
                    except queue.Empty:
                        active_job = None

                nx, ny, tracked = sess.gaze.step(frame_t0)
                gp = gaze_to_motion_params(nx, ny, frame_t0, tracked)
                react = sess.reaction.step(frame_t0)

                if active_job is not None:
                    if sess.active_speech_cancel.is_set():
                        sid = active_job["speechId"]
                        active_job = None
                        await websocket.send_json({"type": "speech_end", "speechId": sid, "interrupted": True})
                        jpg = await asyncio.to_thread(engine._render_one_frame, engine.x_s_info, None, gp, frame_t0, 1.0, react, sess.blink)
                    else:
                        elapsed = time.perf_counter() - speech_t0
                        frames_list = active_job["x_d_info_list"]
                        fidx = int(elapsed * 25.0)
                        if fidx >= len(frames_list) or elapsed >= active_job["durationSec"]:
                            sid = active_job["speechId"]
                            has_next = not sess.speech_queue.empty()
                            active_job = None
                            await websocket.send_json({"type": "speech_end", "speechId": sid, "interrupted": False, "chained": has_next})
                            if has_next:
                                try:
                                    active_job = sess.speech_queue.get_nowait()
                                    sess.active_speech_cancel.clear()
                                    speech_t0 = time.perf_counter()
                                    await websocket.send_json(
                                        {
                                            "type": "speech_start",
                                            "speechId": active_job["speechId"],
                                            "durationSec": active_job["durationSec"],
                                            "audioBase64": active_job["audioBase64"],
                                            "mime": "audio/mpeg",
                                        }
                                    )
                                    jpg = await asyncio.to_thread(
                                        engine._render_one_frame,
                                        active_job["x_d_info_list"][0],
                                        active_job["d0"],
                                        gp,
                                        frame_t0,
                                        0.75,
                                        react,
                                        sess.blink,
                                    )
                                except queue.Empty:
                                    jpg = await asyncio.to_thread(engine._render_one_frame, engine.x_s_info, None, gp, frame_t0, 1.0, react, sess.blink)
                            else:
                                jpg = await asyncio.to_thread(engine._render_one_frame, engine.x_s_info, None, gp, frame_t0, 1.0, react, sess.blink)
                        else:
                            fade_in = min(1.0, (fidx + 1) / 3.0)
                            fade_out = 1.0 if not sess.speech_queue.empty() else min(1.0, max(0.0, (len(frames_list) - fidx) / 4.0))
                            f_alpha = min(fade_in, fade_out)
                            jpg = await asyncio.to_thread(
                                engine._render_one_frame,
                                frames_list[fidx],
                                active_job["d0"],
                                gp,
                                frame_t0,
                                f_alpha,
                                react,
                                sess.blink,
                            )
                else:
                    jpg = await asyncio.to_thread(engine._render_one_frame, engine.x_s_info, None, gp, frame_t0, 1.0, react, sess.blink)

                await websocket.send_bytes(jpg)
                if frame_t0 - last_gaze_ack > 1.0:
                    last_gaze_ack = frame_t0
                    await websocket.send_json(
                        {
                            "type": "telemetry",
                            "yaw": round(gp["delta_yaw"] + react["nod_yaw"], 2),
                            "pitch": round(gp["delta_pitch"] + react["nod_pitch"], 2),
                            "tracked": tracked,
                            "speaking": active_job is not None,
                            "blinking": (0.06 <= (frame_t0 - sess.blink.started_at) <= 0.24),
                            "reaction": react["kind"],
                            "avatarId": engine.avatar_id,
                            "renderMs": round((time.perf_counter() - frame_t0) * 1000.0, 1),
                        }
                    )

                dt = time.perf_counter() - frame_t0
                target_fps = engine.fps_target if active_job is not None else min(engine.fps_target, 20.0)
                sleep_for = max(0.002, (1.0 / target_fps) - dt)
                await asyncio.sleep(sleep_for)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                stop_flag.set()
                break

    speak_task = asyncio.create_task(speak_worker_loop())
    rx_task = asyncio.create_task(rx_loop())
    tx_task = asyncio.create_task(tx_loop())
    try:
        await stop_flag.wait()
    finally:
        if session_id and ACTIVE_SESSIONS.get(session_id) is session_entry:
            ACTIVE_SESSIONS.pop(session_id, None)
        speak_task.cancel()
        rx_task.cancel()
        tx_task.cancel()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GAZE_WORKER_PORT", "8766"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
