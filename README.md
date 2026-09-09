# fast-rtxvsr

Standalone NVIDIA RTX Video Super Resolution for video files: upscale a clip
to delivery size with the real VSR model, straight from the command line.
No ComfyUI server, no PNG roundtrips, no separate denoise pass. Decode,
super-resolve, and encode all happen on the GPU in one pass.

```text
fast-rtxvsr run input.mp4 --width 1920 --height 1080
-> out/<project>/input/input_vsr.mp4   (1920x1080, VSR ULTRA)
```

## Why this exists

NVIDIA ships RTX Video Super Resolution as a closed, driver-integrated
feature and as a ComfyUI-style SDK wheel (`nvidia-vfx`). Public wrappers for
the SDK are few, graph-bound, or roundtrip every frame through PNG files on
the CPU. This repo drives the `VideoSuperRes` model directly and keeps pixels
on the GPU for the whole pipeline:

1. **One GPU pass, zero frame I/O** - NVDEC decodes straight into GPU memory,
   VSR ULTRA runs the model, and NVENC encodes the NV12 result through
   PyNvVideoCodec's GPU-buffer interface. There is no `ffmpeg -i` +
   `-vf scale` + PNG dump between stages, which is where other pipelines spend
   most of their time and bandwidth.
2. **A PyAV host fallback** - if the PyNvVideoCodec wheel is missing or its
   NVDEC/NVENC init fails, the same model runs on decoded host frames. A
   broken I/O wheel never takes the model down with it.
3. **Fixes that make the GPU path actually work.** This port carries the
   accumulated fixes that separate a working NVDEC -> VSR -> NVENC loop from a
   garbled one:
   - A stream sync before every `Encode()`. NVENC reads the NV12 surface
     without waiting for the async torch kernels that just built it; without a
     sync every ~15 seconds a frame encodes as garbage green blocks.
   - A CUDA-array-interface surface for the encoder. PyNvVideoCodec 2.x
     dispatches `Encode()` on `__dlpack__` first, which trips torch 2.13's
     keyword-only stream argument; GPU-buffer mode wants one CUDA array per
     NV12 plane instead (`_GpuSurface`).
   - Correct frame rate from the decoder. PyNvVideoCodec's stream metadata
     names the rate `average_fps`, not `avg_frame_rate` - miss it and every
     encode silently drops to the 24 fps default, stretching a 60 fps master
     to 2.5x duration. An ffprobe fallback covers containers that omit the
     field entirely.
   - BT.709 limited-range color math done on the GPU (YUV plane conversion +
     4:2:0 chroma subsample as tensor ops), and `bt709` colorspace flagged on
     the NVENC stream.
4. **No silent video.** Source audio is re-muxed onto the VSR result with a
   silence pad, so the picture (and any tail fade) is never trimmed and the
   file is never muted.

### Measured on this repo (Sep 2026, RTX 5060 Ti 16 GB, driver 616.56)

VSR ULTRA versus NVIDIA's other AI upscaler (DLSS Video Upscale, feature 18)
on identical 12-second gen-size excerpts upscaled to delivery size. Both legs
ran inside the same benchmark harness; raw reports are in `bench/`.

| Excerpt (gen size -> delivery) | RTX VSR ULTRA | DLSS Video Upscale | VSR faster |
| --- | ---: | ---: | ---: |
| boucle-furniture, 768x1344 -> 1080x1920 | 10.7 s (26.7 fps e2e) | 22.6 s (12.7 fps e2e) | **2.1x** |
| winged-couriers, 1344x768 -> 1920x1080 | 10.8 s (26.5 fps e2e) | 16.5 s (17.5 fps e2e) | **1.5x** |

A Laplacian-variance sharpness heuristic on stills from each excerpt favored
VSR at five of the six sampled frames (the DLSS report also shows its noise
reduction falling back to native on this content). These wall-clock numbers
include each leg's own harness overhead; the pure VSR worker core on the same
winged-couriers excerpt runs the 288-frame pass in ~3 s at ~80-95 frames/s
with the standalone CLI in this repo (two runs, same GPU).

## Requirements

- Windows
- An NVIDIA RTX GPU
- A current NVIDIA display driver (the SDK targets the modern driver line)
- FFmpeg and FFprobe on `PATH` (or set `FAST_RTXVSR_FFMPEG` /
  `FAST_RTXVSR_FFPROBE` to the executables)
- Python >= 3.10, **3.12 recommended** (the PyNvVideoCodec CUDA wheels are
  published for cp312)

The worker environment needs CUDA torch, NVIDIA's `nvidia-vfx` SDK wheel
(~490 MB, from NVIDIA's own index), `PyNvVideoCodec`, and the CUDA 12 runtime
DLL those wheels link. `fast-rtxvsr setup` installs all of it into a venv this
repo owns - nothing is installed into a shared Python, and nothing downloads
at render time (the VSR model runtime ships inside the `nvidia-vfx` wheel).

## Install

```bash
pip install -e .
```

## Provision the worker environment (once)

```bash
fast-rtxvsr setup
```

This creates `.venv/` under the repo and installs torch (cu130) plus the
NVIDIA VFX stack into it (several GB, one-time). If you already have an
interpreter with the stack - for example a ComfyUI venv that stages the same
wheels - skip the download entirely and point at it:

```bash
fast-rtxvsr setup --python D:/path/to/ComfyUI/.venv/Scripts/python.exe
# or for every future run:
set FAST_RTXVSR_PYTHON=D:/path/to/ComfyUI/.venv/Scripts/python.exe
```

`fast-rtxvsr probe` prints what a given interpreter can see
(CUDA / nvidia-vfx / PyNvVideoCodec / PyAV).

## Usage

```bash
# Upscale a 1344x768 clip to 1080p delivery
fast-rtxvsr run clip.mp4 --width 1920 --height 1080

# Several clips, explicit output dir, HEVC master
fast-rtxvsr run a.mp4 b.mp4 c.mp4 --out-dir ./delivery \
    --width 3840 --height 2160 --codec hevc --preset P7

# Full option set
fast-rtxvsr run input.mp4 --width 1920 --height 1080 --quality ULTRA \
    --codec h264 --preset P7 --bitrate 16000000 --device 0
```

By default each output lands under `<repo>/out/<source-project>/<clip-stem>/`
as `<clip-stem>_vsr.mp4`; `--out-dir` writes clips directly there. Output
dimensions are rounded up to an 8px multiple (the VSR model's alignment).

### CLI reference

```
fast-rtxvsr setup [--python PATH]
fast-rtxvsr probe [--python PATH]
fast-rtxvsr run INPUT [options]

  --width WIDTH        Output width (default 1920)
  --height HEIGHT      Output height (default 1080)
  --quality QUALITY    LOW | MEDIUM | HIGH | ULTRA  (default ULTRA)
  --codec CODEC        h264 (default) | hevc | av1  (nvenc aliases h264)
  --preset PRESET      NVENC P1..P7 (default P7, highest quality)
  --bitrate BITRATE    Encoder bitrate (default 16000000 / 16 Mbps)
  --device DEVICE      CUDA device index (default 0)
  --audio-bitrate R    AAC rate for the audio re-mux (default 192k)
  --out-dir DIR        Override output directory
  --python PATH        Worker interpreter (default: repo .venv, then
                       FAST_RTXVSR_PYTHON)
  --frames-in DIR      Legacy debug mode: upscale a PNG/JPG folder
  --frames-out DIR     Output folder for --frames-in
  --ext EXT            Frame extension for --frames-in (default jpg)
```

The NVDEC/NVENC GPU path runs by default when PyNvVideoCodec imports; a
failure mid-init logs a `gpu_fail` event and falls back to the PyAV host
path automatically. `--frames-in` mode is the PNG/JPG roundtrip path kept for
debugging only.

### Machine-readable progress

`fast-rtxvsr run` stdout is pure JSON, one object per line - easy to consume
from a script:

```json
{"log": "device", "mode": "video", "path": "gpu", "cuda_index": 0, "device": "NVIDIA GeForce RTX 5060 Ti", "quality": "ULTRA", "output": "1920x1080"}
{"log": "encoder", "codec": "h264", "path": "gpu", "preset": "P7"}
{"log": "model_loaded", "loaded": true}
{"ok": true, "mode": "video", "path": "gpu", "device": "NVIDIA GeForce RTX 5060 Ti", "frames": 288, "expected_frames": 288, "input": "1344x768", "output": "1920x1080", "seconds": 3.07, "fps": 93.87}
{"event": "done", "mode": "video", "input": "D:/in/clip.mp4", "output": "D:/out/clip_vsr.mp4", "wall_clock_sec": 8.4, "frames": 288, "fps": 93.87, "codec": "h264", "preset": "P7", "out_probe": "clip_vsr.mp4 1920x1080 14.20 Mbps 21061 KB"}
```

Human progress and the end-of-run summary go to stderr, so redirecting
`2>nul` yields a clean event stream.

## How it works

The package drives NVIDIA's `nvvfx.VideoSuperRes` directly. On the GPU path,
`PyNvVideoCodec.SimpleDecoder` opens the file with `use_device_memory=True`
and NVDEC decodes frames into device memory; each decoded RGBP frame becomes
a CUDA float tensor (`from_dlpack`, zero copy), the model runs at the chosen
quality, and the output tensor is converted to NV12 in the same kernel launch
stream. NVENC then encodes the NV12 planes through a per-plane CUDA Array
Interface object, with a stream sync before every encode so the encoder never
reads a half-written surface. The elementary stream is muxed with `genpts` and
the source audio re-muxed on top (AAC 192k, silence-padded). A decoded
60 fps source keeps its 60 fps timestamp - the pipeline never assumes 24 fps.

The worker also ships a `python -m fast_rtxvsr.vsr` entry point whose CLI and
JSON event contract match the worker AutoTube's pipeline spawns, so it is a
drop-in replacement for that subprocess.

## Troubleshooting

**"missing python" / setup created nothing** - the repo `.venv` does not
exist yet. Run `fast-rtxvsr setup`, or point `FAST_RTXVSR_PYTHON` /
`--python` at an interpreter that already has the stack.

**"nvidia-vfx not importable"** - the SDK wheel is installed from NVIDIA's
index (`https://pypi.nvidia.com`), not PyPI (PyPI carries only an sdist that
cannot build on Windows). Re-run `fast-rtxvsr setup`; it installs
`nvidia-vfx` in its own pip call so NVIDIA's index cannot shadow other
packages.

**"CUDA unavailable"** - no usable CUDA torch in the worker interpreter. The
setup venv pins torch cu130; if you pointed `--python` at another venv, it
needs a CUDA build of torch.

**"DLL load failed while importing _PyNvVideoCodec"** - the CUDA 12 runtime
DLL is missing. `fast-rtxvsr setup` installs `nvidia-cuda-runtime-cu12`;
pointing at a ComfyUI venv that lacks it triggers the same error (the PyAV
host fallback still runs if `av` is present).

**ffmpeg / ffprobe not found** - install FFmpeg and add it to `PATH`, or set
`FAST_RTXVSR_FFMPEG` / `FAST_RTXVSR_FFPROBE`.

**Output is stretched or the duration doubled** - the source frame rate was
missed (see the `average_fps` fix above). Upgrade to a recent worker; the
ffprobe fallback covers containers whose metadata omits the rate.

## Licensing

- This repo (the CLI wrapper, media helpers, docs): **MIT**, see `LICENSE`.
- The VSR model and NVIDIA Video Effects runtime come from the
  `nvidia-vfx` wheel published by NVIDIA on `https://pypi.nvidia.com`. Its
  license agreements ship inside the wheel under
  `nvidia_vfx-*/dist-info/licenses/` (NVIDIA Software License Agreement +
  NVIDIA Open Model License) and include redistribution terms - review them
  before distributing or shipping an application that embeds the runtime.
- `PyNvVideoCodec` is NVIDIA's CUDA video codec binding, distributed under
  its own terms in the wheel metadata.

fast-rtxvsr does not vendor any NVIDIA binaries in its own repository;
`fast-rtxvsr setup` installs them into a gitignored venv.

## Out of scope

This is the fast RTX Video Super Resolution CLI only. DLSS video upscale,
DLSS frame interpolation, and any ComfyUI-graph integration are not covered
here. For DLSS Frame Generation see the companion
[fast-dlssfg](https://github.com/glarsson/fast-dlssfg) project.
