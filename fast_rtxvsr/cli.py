"""fast-rtxvsr command line: setup, probe, and run.

    fast-rtxvsr setup [--python PATH]         provision (or point at) a worker env
    fast-rtxvsr probe [--python PATH]         show what the worker env can see
    fast-rtxvsr run INPUT [options]           upscale a video to delivery size

``fast-rtxvsr run`` stdout is machine-readable: line-delimited JSON events
(device / encoder / model_loaded / ok / done), one object per line. Human
progress goes to stderr, so ``2>nul`` yields a clean event stream.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .env import (
    ensure_worker_python,
    probe_python,
    repo_root,
    resolve_python,
)
from .media import attach_source_audio, probe_video_summary
from .vsr import _normalize_codec

QUALITY_CHOICES = ("LOW", "MEDIUM", "HIGH", "ULTRA")


def _resolve_video_dest(src: Path, out_dir: Path | None) -> Path:
    """Output file for one clip.

    With --out-dir the clip writes directly there; by default outputs land
    under <repo>/out/<parent-project>/<clip-stem>/<clip-stem>_vsr.mp4.
    """
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve() / f"{src.stem}_vsr.mp4"
    return repo_root() / "out" / src.parent.name / src.stem / f"{src.stem}_vsr.mp4"


def _resolve_frames_dest(frames_in: Path, frames_out: Path | None) -> Path:
    if frames_out is not None:
        return Path(frames_out).expanduser().resolve()
    return repo_root() / "out" / frames_in.parent.name / f"{frames_in.name}_up"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-rtxvsr",
        description=(
            "Standalone NVIDIA RTX Video Super Resolution. Drives the "
            "nvidia-vfx VideoSuperRes model directly: GPU decode (NVDEC) -> "
            "VSR -> GPU encode (NVENC) in one pass, no ComfyUI server. Run "
            "'fast-rtxvsr setup' once to provision the worker environment."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Provision the VSR worker environment.")
    setup.add_argument(
        "--python",
        default=None,
        help="Reuse an interpreter that already has the stack instead of "
        "creating the repo venv (e.g. a ComfyUI venv).",
    )

    probe = sub.add_parser(
        "probe", help="Check CUDA + nvidia-vfx + PyNvVideoCodec/PyAV."
    )
    probe.add_argument("--python", default=None, help="Interpreter to probe.")

    run = sub.add_parser(
        "run",
        help="Upscale one video to the target size (or upscale a PNG/JPG folder).",
    )
    run.add_argument(
        "input",
        nargs="*",
        default=None,
        help="Input video file(s). Omit when using --frames-in.",
    )
    run.add_argument(
        "--width", type=int, default=1920, help="Output width (default 1920)."
    )
    run.add_argument(
        "--height", type=int, default=1080, help="Output height (default 1080)."
    )
    run.add_argument(
        "--quality",
        default="ULTRA",
        choices=QUALITY_CHOICES,
        help="VSR quality level (default ULTRA).",
    )
    run.add_argument(
        "--codec",
        default="h264",
        help="NVENC codec: h264 (default), hevc, av1. nvenc aliases h264.",
    )
    run.add_argument(
        "--preset",
        default="P7",
        help="NVENC preset P1..P7 (default P7, highest quality).",
    )
    run.add_argument(
        "--bitrate", type=int, default=16_000_000, help="Encoder bitrate (default 16 Mbps)."
    )
    run.add_argument("--device", type=int, default=0, help="CUDA device index.")
    run.add_argument(
        "--audio-bitrate",
        default="192k",
        help="AAC bitrate when re-muxing source audio (default 192k).",
    )
    run.add_argument("--out-dir", default=None, help="Override output directory.")
    run.add_argument("--python", default=None, help="Worker interpreter to use.")
    run.add_argument(
        "--frames-in",
        default=None,
        type=Path,
        help="Legacy debug mode: folder of PNG/JPG frames to upscale.",
    )
    run.add_argument(
        "--frames-out",
        default=None,
        type=Path,
        help="Legacy debug mode: output folder for upscaled frames.",
    )
    run.add_argument("--ext", default="jpg", help="Frame extension for --frames-in.")
    return parser


def _print_probe(python: Path, info: dict) -> None:
    ok = info.get("healthy")
    state = "ready" if ok else "NOT ready"
    print(f"[fast-rtxvsr] {python} : {state}")
    print(f"  CUDA: {info.get('cuda_available')}  device: {info.get('device')}")
    print(f"  nvidia-vfx (nvvfx): {'yes' if info.get('nvvfx_ok') else 'no'}")
    print(f"  PyNvVideoCodec (GPU I/O): {'yes' if info.get('pynvvideocodec_ok') else 'no'}")
    print(f"  PyAV (host fallback): {'yes' if info.get('pyav_ok') else 'no'}")
    if info.get("error"):
        print(f"  error: {info['error']}")


def cmd_setup(args: argparse.Namespace) -> int:
    if args.python:
        python = resolve_python(args.python)
        info = probe_python(python)
        _print_probe(python, info)
        if not info["healthy"]:
            print(
                f"[fast-rtxvsr] {python} is not ready; run `fast-rtxvsr setup` "
                "without --python to provision the repo venv.",
                file=sys.stderr,
            )
            return 1
        return 0
    python = ensure_worker_python(None)
    info = probe_python(python)
    _print_probe(python, info)
    return 0 if info["healthy"] else 1


def cmd_probe(args: argparse.Namespace) -> int:
    python = resolve_python(args.python)
    info = probe_python(python)
    print(json.dumps(info, indent=2))
    return 0 if info["healthy"] else 1


def _run_worker(python: Path, argv: list[str]) -> list[dict]:
    """Spawn the VSR worker, return its JSON events in order.

    stdout of this process stays pure JSON: each worker event is replayed
    verbatim. Anything non-JSON is dropped from stdout and only surfaces as
    error detail on stderr.
    """
    env = os.environ.copy()
    root = str(repo_root())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    result = subprocess.run(
        [str(python), "-m", "fast_rtxvsr.vsr", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    events: list[dict] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for event in events:
        print(json.dumps(event), flush=True)
    if result.returncode != 0 or not events:
        tail = (result.stderr or result.stdout or "worker failed")[-1500:]
        raise RuntimeError(tail)
    return events


def _final_payload(events: list[dict]) -> dict:
    for event in reversed(events):
        if event.get("ok") is not None or event.get("error"):
            return event
    return {}


def _run_one_video(
    src: Path,
    dest: Path,
    python: Path,
    codec: str,
    args: argparse.Namespace,
) -> int:
    raw = dest.with_name(f"{dest.stem}_vsr_raw.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[fast-rtxvsr] RTX VSR quality={args.quality} codec={codec} "
        f"preset={args.preset} {src.name} -> {args.width}x{args.height}",
        file=sys.stderr,
    )
    started = time.time()
    try:
        events = _run_worker(
            python,
            [
                "--input-video",
                str(src),
                "--output-video",
                str(raw),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--quality",
                args.quality,
                "--device",
                str(args.device),
                "--codec",
                codec,
                "--preset",
                args.preset,
                "--bitrate",
                str(args.bitrate),
            ],
        )
        payload = _final_payload(events)
        if not payload.get("ok"):
            print(f"[fast-rtxvsr] VSR failed: {payload.get('error')}", file=sys.stderr)
            return 1
        t_mux = time.time()
        attach_source_audio(
            raw,
            src,
            dest,
            audio_bitrate=args.audio_bitrate,
            encode=codec,
            preset=args.preset,
        )
        print(
            f"[fast-rtxvsr] audio mux {time.time() - t_mux:.1f}s",
            file=sys.stderr,
        )
    finally:
        raw.unlink(missing_ok=True)

    elapsed = time.time() - started
    print(
        f"[fast-rtxvsr] {probe_video_summary(dest)}  "
        f"{elapsed:.1f}s wall  {payload.get('fps')} frames/s end-to-end",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "event": "done",
                "mode": "video",
                "input": str(src),
                "output": str(dest),
                "wall_clock_sec": round(elapsed, 2),
                "frames": payload.get("frames"),
                "expected_frames": payload.get("expected_frames"),
                "fps": payload.get("fps"),
                "codec": payload.get("codec"),
                "preset": payload.get("preset"),
                "path": payload.get("path"),
                "device": payload.get("device"),
                "out_probe": probe_video_summary(dest),
            }
        ),
        flush=True,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # ensure_worker_python verifies the interpreter is healthy (or provisions
    # the venv); a broken one raises before any work starts.
    python = ensure_worker_python(args.python)
    codec = _normalize_codec(args.codec)

    if args.frames_in:
        frames_in = Path(args.frames_in).expanduser().resolve()
        frames_out = _resolve_frames_dest(frames_in, args.frames_out)
        started = time.time()
        events = _run_worker(
            python,
            [
                "--frames-in",
                str(frames_in),
                "--frames-out",
                str(frames_out),
                "--ext",
                args.ext,
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--quality",
                args.quality,
                "--device",
                str(args.device),
            ],
        )
        payload = _final_payload(events)
        if not payload.get("ok"):
            print(f"[fast-rtxvsr] VSR failed: {payload.get('error')}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "event": "done",
                    "mode": "frames",
                    "wall_clock_sec": round(time.time() - started, 2),
                    "frames": payload.get("frames"),
                    "frames_out": str(frames_out),
                }
            ),
            flush=True,
        )
        return 0

    if not args.input:
        print("[fast-rtxvsr] run needs an INPUT video (or --frames-in)", file=sys.stderr)
        return 2
    for item in args.input:
        src = Path(item).expanduser().resolve()
        if not src.is_file():
            print(f"[fast-rtxvsr] input not found: {src}", file=sys.stderr)
            return 2
        dest = _resolve_video_dest(src, args.out_dir)
        code = _run_one_video(src, dest, python, codec, args)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "setup":
            return cmd_setup(args)
        if args.command == "probe":
            return cmd_probe(args)
        if args.command == "run":
            return cmd_run(args)
    except Exception as exc:  # noqa: BLE001 - clean failure, no traceback
        print(f"[fast-rtxvsr] error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
