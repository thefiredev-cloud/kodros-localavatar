# Build guide: kodros-localavatar

## Requirements

- Linux, `ffmpeg` and `ffprobe` on `PATH`, `git`, `uv` (used by `scripts/install.sh`).
- Python 3.12 for the CLI and Python 3.10 for the SadTalker engine. SadTalker needs numpy 1.23 and torch 2.1 (cu121). Kokoro ONNX needs numpy 2. That is why there are two virtual environments.
- An NVIDIA GPU for the face animation. A CPU run works but takes close to an hour for a few seconds of video.
- About 2.5 GB of disk for the SadTalker weights.

## 1. Install

```bash
scripts/install.sh
```

This creates `~/.venvs/localavatar` (CLI, Kokoro, tests) and `~/.venvs/localavatar-engine` (SadTalker), installs `engine-requirements.txt`, runs `localavatar setup` and links `~/.local/bin/localavatar`. Override the engine interpreter with `LOCALAVATAR_ENGINE_PYTHON`. Data lives in `~/.local/share/localavatar` (mode 0700, override with `LOCALAVATAR_HOME`).

Optional GPU-busy gate: set `LOCALAVATAR_GPU_BLOCK` to a comma-separated list of process-name fragments (for example `blender,comfyui`). While one of them runs, `make` refuses GPU work with "a GPU-reserved process is running" and tells you to stop it or pass `--cpu`. It is empty by default, so nothing is blocked and `/proc` is not scanned.

`localavatar setup` clones SadTalker at the pinned commit `cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8` (2023-10-11) and downloads the SadTalker v0.0.2-rc weights, GFPGAN v1.4 and the facexlib detectors from GitHub. It is the only step that uses the network.

Put the Kokoro model files `kokoro-v1.0.onnx` and `voices-v1.0.bin` in `~/.local/share/kokoro`, or point `KOKORO_DIR` at the folder that holds them.

## 2. Test

```bash
~/.venvs/localavatar/bin/python -m pytest -q
```

87 tests. They cover script splitting (word preservation, limits, merges, paragraphs), duration math, real ffmpeg renders whose output must end with the audio (pad, trim and equal cases, three aspects, captions burned), CLI parsing and flag plumbing into the SadTalker command. Special-path renders use generated black video and a sine tone, then decode the output and require visible caption pixels and non-silent audio. The web UI tests start the real HTTP server on a random loopback port with a fake render runner: multipart upload, validation, queue order and cancel, restart recovery, range downloads, Host and Origin refusal, voice samples, GPU health. The CUDA preflight tests fake the probe, and the GPU-busy gate tests start a throwaway subprocess. The tests do not run SadTalker, Kokoro, network models or services.

## 3. Run

```bash
localavatar make --photo FACE.jpg --script script.txt --voice af_heart --out out.mp4 \
                 [--aspect source|9:16|1:1|16:9] [--captions] [--size 256|512] [--motion]
localavatar make --photo FACE.jpg --audio my-recording.m4a --out out.mp4   # your own voice, no captions
localavatar make --photo FACE.jpg --text "Short inline script" --out out.mp4   # script given on the command line
localavatar doctor    # weights, Kokoro files, free VRAM, GPU-busy gate
localavatar serve [--port 8870]
```

The web UI takes a photo, a script, a voice (English Kokoro voices with a sample button and speed slider), a framing, captions and Fast or Best quality. Jobs queue and render one at a time. Uploads are capped at 25 MB (photo 20 MB, at least 128 px) and scripts at 6,000 characters. Job data is under `~/.local/share/localavatar/web/jobs/<id>/` (mode 0700). Add `--no-enhance` to skip face restore and `--cpu` to avoid the GPU.

## Pipeline

1. Split the script (`script.py`): sentences, long ones cut at clauses then words (220 character cap for Kokoro), tiny ones merged. Blank lines are paragraph pauses.
2. Voice (`tts.py`): Kokoro 82M ONNX on CPU. Chunks are trimmed and joined with 0.28 s and 0.6 s pauses. Narration is cached by text, voice and speed. `--audio FILE` skips this step.
3. Animate (`animate.py`, `engine_run.py`): SadTalker in its own venv, one model stage at a time, batch 1. Audio drives expression and pose. `--preprocess full --still` animates the face crop and pastes it back. GFPGAN v1.4 restores face detail per frame at 2x. 25 fps.
4. Fit and render (`render.py`, `timing.py`): pad or trim so the video ends with the audio, optional 9:16, 1:1 or 16:9 layout (sharp portrait over a blurred copy), optional burned captions, loudnorm to -16 LUFS, NVENC with a libx264 fallback.
5. Captions (`captions.py`): up to three words on screen. Chunk times are exact from the voice step. Word times inside a chunk are spread by length.

Output is `out.mp4` plus `out.json` (chunks, timings, durations, engine, license, runtime, GPU peak). If a process listed in `LOCALAVATAR_GPU_BLOCK` is running, `make` refuses GPU work.

## Model choice

| Candidate | License (checked 2026-10-21) | Verdict |
|---|---|---|
| SadTalker (OpenTalker, CVPR 2023) | Apache-2.0 code, README says the non-commercial restriction was removed, weights on GitHub releases | chosen |
| MuseTalk 1.5 | MIT code, weights "any purpose, even commercially" | needs a source video, re-renders the mouth only |
| LivePortrait | MIT code, bundled InsightFace detector weights marked non-commercial, no upstream audio driver | needs a separate audio-to-motion model |
| Hallo / Hallo2 | MIT code, diffusion based, upstream reports long render times and high VRAM | too heavy for a shared 16 GB card (not benchmarked here) |
| Wav2Lip | repo restricts use to personal, research and non-commercial | license, and mouth-only |

SadTalker is the only one here that turns one still photo and audio into head motion, blinks and lip sync with a permissive code license in a few GB of VRAM. Caveat: SadTalker's face-reconstruction net (Deep3DFaceRecon) was trained with the Basel Face Model, whose license is research-oriented, and GFPGAN was trained on FFHQ (CC BY-NC-SA). Get a license review before selling output or shipping the weights in a product. SadTalker and GFPGAN come from research groups outside the US.

## Results

Proof input: `fixtures/astronaut-public-domain.png`, a NASA portrait of astronaut Eileen Collins from scikit-image's sample data (public domain, US federal work), and `fixtures/proof-script.txt`.

GPU run, 2026-10-21, on a shared RTX 4080 16 GB (load average 25 to 70 on 3 CPUs, 12.5 GB of the GPU already in use by other services). Command: `localavatar make --photo fixtures/astronaut-public-domain.png --script fixtures/proof-script.txt --voice af_heart --aspect 9:16 --captions --out collins-9x16.mp4`.

| Stage | Time | Peak VRAM |
|---|---|---|
| Kokoro TTS, CPU (67 words, 23.15 s audio) | 83 s | 0 |
| SadTalker 3DMM fit / audio-to-motion | 21 s / 7 s | 472 / 194 MiB (torch) |
| Face render 578 frames / paste | 2 min 15 s / 1 min 14 s | 2,323 MiB (torch) |
| GFPGAN face restore, 2x (2.23 s per frame, likely CPU-bound under load, not profiled) | 17 min | same stage |
| ffmpeg 9:16 + captions + loudnorm, NVENC | 84 s | |
| Total | 27 min 13 s | 3,118 MiB (process tree, nvidia-smi) |

Output 1080x1920, 25 fps, video 23.16 s against audio 23.147 s. `scripts/mouth_motion.py` measured mouth-region change 3.84 during speech against 2.03 in pauses, and 0.01 in a background control box. The lip shapes lean toward smile and teeth, and wide vowels are weak. With `--no-enhance` the same script at 512x512 took 7 min 6 s (2,218 MiB peak).

CPU run, 2026-10-29: the GPU refused new CUDA contexts, so a short test ran on CPU at load average 46 to 95. Kokoro took 24.9 s for 6.36 s of audio. SadTalker on CPU took 58 min 10 s for 158 frames at 256 px with no face restore. The ffmpeg stage took a few seconds.

Two early runs hit CUDA out-of-memory with SadTalker's stock `inference.py` (all models resident, batch 2). `engine_run.py` loads stages one at a time with batch 1.

## Layout

```text
localavatar/   script, tts, animate, engine_run, render, timing, captions, gpu, models, webui
tests/         87 tests
scripts/       install.sh, setup_models.py, mouth_motion.py
fixtures/      public-domain portrait and proof script
docs/architecture.svg
```
