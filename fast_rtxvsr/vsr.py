"""RTX Video Super Resolution via nvidia-vfx (NVIDIA's CUDA Video Effects SDK).

Video mode (the default): decode -> GPU VSR -> encode in one pass.
Prefers NVDEC + NVENC (PyNvVideoCodec) so pixels stay on the GPU.
Falls back to PyAV host frames if that wheel is missing or init fails.
Legacy frame mode: PNG/JPG folders for debugging only.

Run with the fast-rtxvsr venv interpreter (provisioned by ``fast-rtxvsr
setup``), or any Python that already has the stack installed::

    fast-rtxvsr run input.mp4 --width 1920 --height 1080
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from .media import ffmpeg_exe, probe_fps


def _log_event(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


def _normalize_codec(raw: str | None) -> str:
    text = str(raw or "h264").strip().lower()
    if text in {"nvenc", "nvidia", "cuda", "h264", "h264_nvenc", "avc"}:
        return "h264"
    if text in {"hevc", "hevc_nvenc", "h265", "h265_nvenc"}:
        return "hevc"
    if text in {"av1", "av1_nvenc"}:
        return "av1"
    return "h264"


def _normalize_preset(raw: str | None) -> str:
    text = str(raw or "P7").strip().upper()
    if text.startswith("P") and text[1:].isdigit():
        number = int(text[1:])
        if 1 <= number <= 7:
            return f"P{number}"
    if text.isdigit() and 1 <= int(text) <= 7:
        return f"P{int(text)}"
    return "P7"


def _prepare_pynv_dlls() -> None:
    """Windows: PyNvVideoCodec_130.pyd links cudart64_12.dll (CUDA 12 runtime)."""
    import os
    import sys

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    roots: list[Path] = []
    for item in getattr(sys, "path", []):
        try:
            roots.append(Path(item))
        except TypeError:
            continue
    candidates: list[Path] = []
    cuda_bin = Path(os.environ.get("CUDA_PATH") or "") / "bin"
    if cuda_bin.is_dir():
        candidates.append(cuda_bin)
    for root in roots:
        candidates.append(root / "PyNvVideoCodec")
        candidates.append(root / "nvidia" / "cuda_runtime" / "bin")
    seen: set[str] = set()
    for folder in candidates:
        try:
            resolved = str(folder.resolve())
        except OSError:
            continue
        if resolved in seen or not folder.is_dir():
            continue
        seen.add(resolved)
        os.add_dll_directory(resolved)


def _pynv_available() -> bool:
    try:
        _prepare_pynv_dlls()
        import PyNvVideoCodec  # noqa: F401
    except Exception:
        return False
    return True


def _avframe_to_rgb_float(frame, gpu: int):
    import torch

    arr = frame.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu}")
    return tensor.permute(2, 0, 1).float().div_(255.0).contiguous()


def _pick_video_encoder(container, width: int, height: int, fps: float, bitrate: int, codec: str):
    import av

    frame_rate = Fraction(fps if fps else 24).limit_denominator(10000)
    wanted = {
        "h264": ("h264_nvenc", "hevc_nvenc", "libx264"),
        "hevc": ("hevc_nvenc", "h264_nvenc", "libx264"),
        "av1": ("av1_nvenc", "hevc_nvenc", "h264_nvenc", "libx264"),
    }.get(codec, ("h264_nvenc", "hevc_nvenc", "libx264"))
    for name in wanted:
        try:
            stream = container.add_stream(name, rate=frame_rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = bitrate
            if name.endswith("_nvenc"):
                stream.options = {"preset": "p7", "rc": "constqp", "qp": "16"}
            stream.codec_context.open()
            return stream, name
        except Exception:
            continue
    raise RuntimeError("no usable H.264/H.265 encoder (tried h264_nvenc, hevc_nvenc, libx264)")


class _GpuSurface:
    """NV12 planes for PyNvVideoCodec GPU-buffer Encode().

    Encode() 2.x dispatches on __dlpack__ first, so a raw torch tensor trips
    torch 2.13's keyword-only stream argument. GPU buffer mode instead wants a
    cuda() method returning one CUDA Array Interface object per plane: luma
    (H, W, 1) then chroma (H/2, W/2, 2), all uint8 on the same device buffer.
    """

    def __init__(self, nv12) -> None:
        import torch

        if not isinstance(nv12, torch.Tensor) or not nv12.is_cuda:
            raise ValueError("NVENC surface must be a CUDA tensor")
        if nv12.dim() != 2 or nv12.dtype != torch.uint8:
            raise ValueError("NVENC surface must be uint8 (H*3/2, W)")
        height, width = nv12.shape
        full = int(height * 2 // 3)
        self._planes = [
            nv12[:full].reshape(full, width, 1).contiguous(),
            nv12[full:].reshape(full // 2, width // 2, 2).contiguous(),
        ]

    def cuda(self):
        return list(self._planes)


def _rgb_float_to_nv12(rgb):
    """BT.709 limited-range RGB float CHW [0,1] -> NV12 uint8 (H*3/2, W) on GPU."""
    import torch

    _, height, width = rgb.shape
    red = rgb[0].clamp(0.0, 1.0)
    green = rgb[1].clamp(0.0, 1.0)
    blue = rgb[2].clamp(0.0, 1.0)
    y_plane = 16.0 + 219.0 * (0.2126 * red + 0.7152 * green + 0.0722 * blue)
    u_plane = 128.0 + 224.0 * (-0.1146 * red - 0.3854 * green + 0.5000 * blue)
    v_plane = 128.0 + 224.0 * (0.5000 * red - 0.4542 * green - 0.0458 * blue)
    y_u8 = y_plane.round().clamp(0, 255).to(torch.uint8)
    u_ds = u_plane.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    v_ds = v_plane.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    u_u8 = u_ds.round().clamp(0, 255).to(torch.uint8)
    v_u8 = v_ds.round().clamp(0, 255).to(torch.uint8)
    uv = torch.stack((u_u8, v_u8), dim=2).reshape(height // 2, width)
    return torch.cat((y_u8, uv), dim=0).contiguous()


def _as_bytes(blob) -> bytes:
    if blob is None:
        return b""
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)
    if hasattr(blob, "tobytes"):
        data = blob.tobytes()
        return data if data else b""
    try:
        if len(blob) == 0:
            return b""
    except TypeError:
        return b""
    return bytes(bytearray(blob))


def _meta_get(meta, *names, default=None):
    if meta is None:
        return default
    if isinstance(meta, dict):
        for name in names:
            if name in meta and meta[name] not in (None, ""):
                return meta[name]
        return default
    for name in names:
        if hasattr(meta, name):
            value = getattr(meta, name)
            if value not in (None, ""):
                return value
    return default


def _decoder_meta(decoder) -> tuple[int, int, float, int]:
    meta = None
    if hasattr(decoder, "get_stream_metadata"):
        try:
            meta = decoder.get_stream_metadata()
        except Exception:
            meta = None
    width = int(_meta_get(meta, "width", "Width") or 0)
    height = int(_meta_get(meta, "height", "Height") or 0)
    # PyNvVideoCodec's StreamMetadata names the frame rate "average_fps"
    # (not avg_frame_rate/fps) - miss it and every encode silently drops to
    # the 24 fps default below, stretching a 60 fps DLSS-FG master 2.5x.
    fps_raw = _meta_get(
        meta, "average_fps", "avg_frame_rate", "frame_rate", "fps", "FrameRate"
    )
    fps = 0.0
    if isinstance(fps_raw, (tuple, list)) and len(fps_raw) == 2:
        num, den = float(fps_raw[0]), float(fps_raw[1])
        if den:
            fps = num / den
    elif fps_raw not in (None, "", 0, "0/0"):
        text = str(fps_raw)
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            den = float(den_s)
            if den:
                fps = float(num_s) / den
        else:
            try:
                fps = float(text)
            except ValueError:
                fps = 0.0
    frames = int(_meta_get(meta, "num_frames", "n_frames") or 0)
    if not frames:
        try:
            frames = int(len(decoder))
        except Exception:
            frames = 0
    return width, height, fps, frames


def _iter_decoded_frames(decoder):
    if hasattr(decoder, "get_batch_frames"):
        while True:
            batch = decoder.get_batch_frames(1)
            if not batch:
                break
            yield from batch
        return
    total = len(decoder)
    for index in range(total):
        yield decoder[index]


def _rgb_tensor_from_decoded(frame, torch_mod):
    tensor = torch_mod.from_dlpack(frame)
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        planar = tensor
    elif tensor.ndim == 3 and tensor.shape[-1] == 3:
        planar = tensor.permute(2, 0, 1).contiguous()
    else:
        raise RuntimeError(f"unexpected decoded frame shape {tuple(tensor.shape)}")
    return planar.to(dtype=torch_mod.float32).div_(255.0).contiguous()


def _encode_surface(encoder, nv12):
    return encoder.Encode(_GpuSurface(nv12))


def _packet_bytes(packets) -> bytes:
    """Flatten Encode()/EndEncode() output (list of packet dicts) to bytes."""
    if packets is None:
        return b""
    if isinstance(packets, dict):
        packets = [packets]
    if isinstance(packets, (bytes, bytearray, memoryview)):
        return bytes(packets)
    out = bytearray()
    for packet in packets:
        data = packet.get("data") if isinstance(packet, dict) else packet
        if data:
            out += _as_bytes(data)
    return bytes(out)


def _mux_elementary(elem: Path, dest: Path, fps: float) -> None:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-fflags",
        "+genpts",
        "-r",
        str(fps if fps else 24),
        "-i",
        str(elem),
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0 or not dest.exists():
        detail = (result.stderr or result.stdout or "ffmpeg mux failed")[-800:]
        raise RuntimeError(detail)


def _create_nvenc(nvc, width: int, height: int, gpu: int, codec: str, preset: str, fps: float, bitrate: int):
    fps_int = max(1, int(round(fps if fps else 24)))
    params = {
        "gpu_id": gpu,
        "codec": codec,
        "preset": preset,
        "tuning_info": "high_quality",
        "rc": "constqp",
        "constqp": 16,
        "fps": fps_int,
        "bitrate": int(bitrate),
        "colorspace": "bt709",
    }
    try:
        return nvc.CreateEncoder(width, height, "NV12", False, **params)
    except Exception:
        params.pop("constqp", None)
        params["rc"] = "vbr"
        return nvc.CreateEncoder(width, height, "NV12", False, **params)


def _run_video_gpu(args: argparse.Namespace) -> int:
    import torch
    import nvvfx
    import PyNvVideoCodec as nvc

    gpu = int(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in worker Python")

    torch.cuda.set_device(gpu)
    device_name = torch.cuda.get_device_name(gpu)
    quality = getattr(nvvfx.effects.QualityLevel, args.quality)
    out_w = max(8, round(int(args.width) / 8) * 8)
    out_h = max(8, round(int(args.height) / 8) * 8)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    codec = _normalize_codec(args.codec)
    preset = _normalize_preset(args.preset)

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    if not input_path.exists():
        _log_event({"ok": False, "error": f"input not found: {input_path}"})
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _log_event(
        {
            "log": "device",
            "mode": "video",
            "path": "gpu",
            "cuda_index": gpu,
            "device": device_name,
            "quality": args.quality,
            "output": f"{out_w}x{out_h}",
        }
    )

    color = getattr(nvc, "OutputColorType")
    decoder_kwargs = {
        "gpu_id": gpu,
        "use_device_memory": True,
        "output_color_type": color.RGBP,
    }
    try:
        decoder = nvc.SimpleDecoder(str(input_path), **decoder_kwargs)
    except TypeError:
        decoder = nvc.SimpleDecoder(str(input_path), gpu_id=gpu)

    input_width, input_height, fps, total_frames = _decoder_meta(decoder)
    if fps <= 0:
        fps = probe_fps(input_path)
    if fps <= 0:
        fps = 24.0

    encoder = _create_nvenc(
        nvc, out_w, out_h, gpu, codec, preset, fps, int(args.bitrate)
    )
    _log_event({"log": "encoder", "codec": codec, "path": "gpu", "preset": preset})

    elem_ext = {"h264": ".h264", "hevc": ".hevc", "av1": ".ivf"}.get(codec, ".h264")
    elem_path = output_path.with_name(f"{output_path.stem}_nvenc{elem_ext}")
    started = time.time()
    processed = 0
    try:
        with nvvfx.VideoSuperRes(quality=quality, device=gpu) as sr:
            if input_width > 0:
                sr.input_width = input_width
            if input_height > 0:
                sr.input_height = input_height
            sr.output_width = out_w
            sr.output_height = out_h
            sr.load()
            _log_event({"log": "model_loaded", "loaded": bool(sr.is_loaded)})

            with elem_path.open("wb") as handle:
                for frame in _iter_decoded_frames(decoder):
                    rgb_input = _rgb_tensor_from_decoded(frame, torch)
                    if input_width <= 0:
                        input_width = int(rgb_input.shape[2])
                        input_height = int(rgb_input.shape[1])
                    output = sr.run(rgb_input, stream_ptr=stream_ptr)
                    rgb_output = torch.from_dlpack(output.image).clone()
                    nv12 = _rgb_float_to_nv12(rgb_output)
                    # NVENC Encode() reads the NV12 surface without waiting for
                    # the async torch kernels that built it; without a stream
                    # sync every ~15s a frame encodes as garbage green blocks.
                    torch.cuda.current_stream().synchronize()
                    handle.write(_packet_bytes(_encode_surface(encoder, nv12)))
                    processed += 1
                torch.cuda.current_stream().synchronize()
                handle.write(_packet_bytes(encoder.EndEncode()))

        _mux_elementary(elem_path, output_path, fps)
    finally:
        elem_path.unlink(missing_ok=True)

    elapsed = time.time() - started
    payload = {
        "ok": True,
        "mode": "video",
        "path": "gpu",
        "device": device_name,
        "cuda_index": gpu,
        "codec": codec,
        "preset": preset,
        "frames": processed,
        "expected_frames": total_frames,
        "input": f"{input_width}x{input_height}",
        "output": f"{out_w}x{out_h}",
        "seconds": round(elapsed, 2),
        "fps": round(processed / elapsed, 2) if elapsed else 0.0,
    }
    _log_event(payload)
    return 0


def _run_video_host(args: argparse.Namespace) -> int:
    import av
    import torch
    import nvvfx

    gpu = int(args.device)
    if not torch.cuda.is_available():
        _log_event({"ok": False, "error": "CUDA unavailable in worker Python"})
        return 2

    torch.cuda.set_device(gpu)
    device_name = torch.cuda.get_device_name(gpu)
    quality = getattr(nvvfx.effects.QualityLevel, args.quality)
    out_w = max(8, round(int(args.width) / 8) * 8)
    out_h = max(8, round(int(args.height) / 8) * 8)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    codec = _normalize_codec(args.codec)

    _log_event(
        {
            "log": "device",
            "mode": "video",
            "path": "host",
            "cuda_index": gpu,
            "device": device_name,
            "quality": args.quality,
            "output": f"{out_w}x{out_h}",
        }
    )

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    if not input_path.exists():
        _log_event({"ok": False, "error": f"input not found: {input_path}"})
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_container = av.open(str(input_path))
    input_stream = input_container.streams.video[0]
    input_stream.thread_type = "AUTO"

    input_width = int(input_stream.codec_context.width or 0)
    input_height = int(input_stream.codec_context.height or 0)
    fps = float(input_stream.average_rate) if input_stream.average_rate else 24.0
    total_frames = int(input_stream.frames or 0)

    output_container = av.open(str(output_path), mode="w")
    video_stream, codec_name = _pick_video_encoder(
        output_container, out_w, out_h, fps, int(args.bitrate), codec
    )
    _log_event({"log": "encoder", "codec": codec_name, "path": "host"})

    started = time.time()
    processed = 0
    with nvvfx.VideoSuperRes(quality=quality, device=gpu) as sr:
        sr.input_width = input_width
        sr.input_height = input_height
        sr.output_width = out_w
        sr.output_height = out_h
        sr.load()
        _log_event({"log": "model_loaded", "loaded": bool(sr.is_loaded)})

        for frame in input_container.decode(input_stream):
            rgb_input = _avframe_to_rgb_float(frame, gpu)
            output = sr.run(rgb_input, stream_ptr=stream_ptr)
            rgb_output = torch.from_dlpack(output.image).clone()
            frame_np = (
                rgb_output.clamp(0.0, 1.0)
                .mul_(255.0)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            out_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in video_stream.encode(out_frame):
                output_container.mux(packet)
            processed += 1

    for packet in video_stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    input_container.close()

    elapsed = time.time() - started
    payload = {
        "ok": True,
        "mode": "video",
        "path": "host",
        "device": device_name,
        "cuda_index": gpu,
        "codec": codec_name,
        "frames": processed,
        "expected_frames": total_frames,
        "input": f"{input_width}x{input_height}",
        "output": f"{out_w}x{out_h}",
        "seconds": round(elapsed, 2),
        "fps": round(processed / elapsed, 2) if elapsed else 0.0,
    }
    _log_event(payload)
    return 0


def run_video(args: argparse.Namespace) -> int:
    if _pynv_available():
        try:
            return _run_video_gpu(args)
        except Exception as exc:  # noqa: BLE001
            _log_event({"log": "encoder", "path": "gpu_fail", "error": str(exc)[:800]})
    return _run_video_host(args)


def run_frames(args: argparse.Namespace) -> int:
    import torch
    import nvvfx
    from PIL import Image
    import numpy as np

    gpu = int(args.device)
    if not torch.cuda.is_available():
        _log_event({"ok": False, "error": "CUDA unavailable in worker Python"})
        return 2

    torch.cuda.set_device(gpu)
    device_name = torch.cuda.get_device_name(gpu)
    paths = sorted(args.frames_in.glob(f"*.{args.ext}"))
    if not paths:
        _log_event({"ok": False, "error": f"no *.{args.ext} in {args.frames_in}"})
        return 2

    args.frames_out.mkdir(parents=True, exist_ok=True)
    quality = getattr(nvvfx.effects.QualityLevel, args.quality)
    out_w = max(8, round(int(args.width) / 8) * 8)
    out_h = max(8, round(int(args.height) / 8) * 8)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    _log_event(
        {
            "log": "device",
            "mode": "frames",
            "cuda_index": gpu,
            "device": device_name,
            "quality": args.quality,
            "output": f"{out_w}x{out_h}",
            "frames": len(paths),
        }
    )

    started = time.time()
    with nvvfx.VideoSuperRes(quality=quality, device=gpu) as sr:
        sr.output_width = out_w
        sr.output_height = out_h
        sr.load()
        for index, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().cuda()
            dlpack_out = sr.run(tensor, stream_ptr=stream_ptr).image
            out = torch.from_dlpack(dlpack_out).clone()
            out_img = (
                out.clamp(0.0, 1.0).detach().float().cpu().numpy() * 255.0
            ).round().astype("uint8")
            dest = args.frames_out / f"{index + 1:06d}.{args.ext}"
            if args.ext.lower() in {"jpg", "jpeg"}:
                Image.fromarray(out_img).save(dest, quality=95)
            else:
                Image.fromarray(out_img).save(dest)

    elapsed = time.time() - started
    payload = {
        "ok": True,
        "mode": "frames",
        "device": device_name,
        "cuda_index": gpu,
        "frames": len(paths),
        "width": out_w,
        "height": out_h,
        "seconds": round(elapsed, 2),
        "fps": round(len(paths) / elapsed, 2) if elapsed else 0.0,
    }
    _log_event(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Worker entry point, run under the stack interpreter.

    ``python -m fast_rtxvsr.vsr ...`` keeps the exact CLI + JSON event
    contract AutoTube's production wrapper expects, so the standalone worker
    is a drop-in for that subprocess.
    """
    parser = argparse.ArgumentParser(prog="fast-rtxvsr-worker")
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--frames-in", type=Path, default=None)
    parser.add_argument("--frames-out", type=Path, default=None)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument(
        "--quality", default="ULTRA", choices=["LOW", "MEDIUM", "HIGH", "ULTRA"]
    )
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--bitrate", type=int, default=16_000_000)
    parser.add_argument(
        "--codec",
        default="h264",
        help="NVENC codec: h264 (default), hevc, av1. nvenc aliases h264.",
    )
    parser.add_argument(
        "--preset",
        default="P7",
        help="NVENC preset P1..P7 (default P7, highest quality).",
    )
    args = parser.parse_args(argv)

    if args.input_video and args.output_video:
        return run_video(args)
    if args.frames_in and args.frames_out:
        return run_frames(args)
    parser.error("pass --input-video + --output-video, or --frames-in + --frames-out")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        _log_event({"ok": False, "error": str(exc)})
        raise SystemExit(1)
