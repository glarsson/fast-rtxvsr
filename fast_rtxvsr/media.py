"""FFmpeg / ffprobe helpers for the standalone VSR CLI.

The VSR worker always writes a video-only elementary mux first (decode ->
VSR -> encode on the GPU); this module re-muxes the source audio back on with
a silence pad so a finished file is never shorter or silent, exactly like the
production AutoTube wrapper does.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    """ffmpeg binary: FAST_RTXVSR_FFMPEG override, else PATH."""
    return os.environ.get("FAST_RTXVSR_FFMPEG") or "ffmpeg"


def ffprobe_exe() -> str:
    """ffprobe binary: FAST_RTXVSR_FFPROBE override, else PATH."""
    return os.environ.get("FAST_RTXVSR_FFPROBE") or "ffprobe"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def probe_fps(path: Path) -> float:
    """Best-effort ffprobe average frame rate; 24.0 when unknown."""
    result = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        return 24.0
    try:
        streams = (json.loads(result.stdout or "{}").get("streams") or [{}])[0]
    except json.JSONDecodeError:
        return 24.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(streams.get(key) or "").strip()
        if not raw or raw in {"0/0", "N/A"}:
            continue
        if "/" in raw:
            num, den = raw.split("/", 1)
            try:
                n, d = float(num), float(den)
            except ValueError:
                continue
            if d > 0:
                return n / d
        try:
            return float(raw)
        except ValueError:
            continue
    return 24.0


def has_audio(path: Path) -> bool:
    result = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return "audio" in (result.stdout or "").lower()


def probe_video_summary(path: Path) -> str:
    result = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,bit_rate,codec_name:format=bit_rate,size",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        return f"{path.name} (unreadable)"
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    width = stream.get("width")
    height = stream.get("height")
    bitrate = int(stream.get("bit_rate") or fmt.get("bit_rate") or 0)
    size = int(fmt.get("size") or path.stat().st_size)
    mbps = bitrate / 1_000_000 if bitrate else 0.0
    return f"{path.name} {width}x{height} {mbps:.2f} Mbps {size // 1024} KB"


def _mux_copy_only(video: Path, dest: Path) -> None:
    """No source audio: copy the VSR video and add faststart."""
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(video),
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not dest.exists():
        detail = (result.stderr or result.stdout or "ffmpeg copy mux failed")[-800:]
        raise RuntimeError(detail)


def _mux_padded_audio(
    video: Path,
    src: Path,
    dest: Path,
    audio_bitrate: str,
    encode: str,
    preset: str,
) -> None:
    """Mux the VSR picture with padded source audio (copy video codec).

    A bare ``-shortest`` would trim a fade-to-black tail down to the audio
    length, so the audio is padded with silence first and ``-shortest`` only
    clips the padding.
    """
    pad_cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(video),
        "-i",
        str(src),
        "-filter_complex",
        "[1:a]apad[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = _run(pad_cmd)
    if result.returncode == 0 and dest.exists():
        return
    print(
        f"[fast-rtxvsr] audio pad mux failed ({(result.stderr or result.stdout)[-400:]}); "
        "re-encoding video",
        flush=True,
    )
    ffmpeg_codec = {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "av1": "av1_nvenc",
    }.get(encode, "h264_nvenc")
    reencode = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(video),
        "-i",
        str(src),
        "-filter_complex",
        "[1:a]apad[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        ffmpeg_codec,
        "-preset",
        preset.lower(),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = _run(reencode)
    if result.returncode != 0 or not dest.exists():
        detail = (result.stderr or result.stdout or "audio re-mux failed")[-1500:]
        raise RuntimeError(detail)


def attach_source_audio(
    video: Path,
    src: Path,
    dest: Path,
    audio_bitrate: str = "192k",
    encode: str = "h264",
    preset: str = "P7",
) -> None:
    """Re-mux source audio onto a finished VSR file at ``dest``.

    Pads short source audio with silence so the VSR picture (and any tail
    fade) is never trimmed. Copies the video stream when the container allows,
    and re-encodes through NVENC only when a copy mux fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not has_audio(src):
        _mux_copy_only(video, dest)
        return
    _mux_padded_audio(video, src, dest, audio_bitrate, encode, preset)
