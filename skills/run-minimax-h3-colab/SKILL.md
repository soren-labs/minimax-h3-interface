---
name: run-minimax-h3-colab
description: "Fully automate MiniMax H3 Turbo generation or H3 Base video repaint batches on one Google Colab G4: decide shutdown before launch, upload the complete batch once, install and start ComfyUI once, process all items sequentially in one remote job, verify and download every artifact, and stop only after the complete batch when requested."
---

# Run MiniMax H3 on Colab

Use the bundled orchestrator as the sole normal-path entry point. Do not reproduce its lifecycle with individual `colab` commands.

## Decide the batch lifecycle before launch

Before the first non-dry-run invocation:

1. Collect the complete known job queue, including every prompt and reference image.
2. Ask the user one blocking lifecycle question: after all currently queued jobs are downloaded and verified, should the Colab assignment be stopped or kept running for likely follow-up jobs?
3. Do not create or mutate a Colab assignment until the user answers `stop after batch` or `keep running`.

Upload the complete known queue and execute it as one remote background job. Environment setup, model loading, and ComfyUI startup happen once; all batch items run sequentially inside that job. Never open a fresh assignment between prompts. If the user adds work after a retained batch, reuse the same assignment.

When the user chooses `keep running`, report the retained assignment name and that it may continue consuming quota or incurring cost. Stop it only on an explicit later instruction. When the user chooses `stop after batch`, stop only after the final queued artifact passes verification.

## Run a production job

Collect these parameters: steps, width, height, duration or exact frames, prompt, reference image, and optional seed. Then execute:

```bash
python3 /mnt/c/Users/zheng/.codex/skills/run-minimax-h3-colab/scripts/h3_colab.py \
  --steps 8 --width 1920 --height 1080 --duration 8 \
  --prompt-file /absolute/path/prompt.txt \
  --reference-image /absolute/path/reference.png \
  --output-dir /absolute/path/results
```

When the user provides only a scene brief instead of a finished H3 prompt, read [references/prompt-format.md](references/prompt-format.md) before creating the prompt file.

Keep this one command attached and relay its live output. It automatically:

1. Reuses the named assignment or repairs a single `[?]` server assignment.
2. Creates one G4 only if the server has no assignment.
3. Uploads inputs and the bundled cu130/sm120 SageAttention wheel.
4. Runs CUDA 13, cu130 Torch, ComfyUI, pinned MiniMax H3/Turbo/KJ components, I2V, and native H3 audio as a remote background job.
5. Polls incremental production logs without holding a long foreground Colab execution.
6. Downloads and verifies the archive and detailed logs.
7. Keeps or stops the assignment after verified success according to the preflight batch policy.

The script normalizes dimensions to multiples of 32 and duration to H3's `5 + 17*n` frame grid, and prints the effective values before starting. Use `--frames` when exact grid control is needed. Use `--dry-run` to validate parameters without touching Colab.

## Run a Base repaint batch

Use a JSON manifest with `jobs`, where each job contains `id`, `input_video`, either `prompt` or `prompt_file`, and optional `seed`. Every source must have video plus audio, run at the requested FPS, and use the H3 `5 + 17*n` frame grid.

```bash
python3 /mnt/c/Users/zheng/.codex/skills/run-minimax-h3-colab/scripts/h3_colab.py \
  --mode repaint \
  --batch-manifest /absolute/path/repaint-batch.json \
  --ngc-env /absolute/path/.ngc.env \
  --resolution 1920x1088 \
  --steps 4 \
  --denoise 0.20 \
  --output-dir /absolute/path/results
```

Repaint mode is a single remote batch. It uses the pinned H3 Ref2VA Base INT8 model, NVIDIA VFX Upscale 2x followed by a Lanczos center-crop resize, video/audio VAE encoding, `PT_H3ConcatAVLatent`, KJ SageAttention, SolAttn, `res_multistep`, and the `beta` scheduler. This explicit Upscale substitution is used on the Colab G4 RTX PRO 6000 Server SKU because RTX VSR cannot initialize there. It never loads the Turbo LoRA or Turbo sampler. After download, finalization stream-copies each source AAC track over the repainted H.264 clip so the delivered video retains the original compressed audio without a second AAC encode. Before stopping an assignment, the orchestrator downloads the batch archive, checks required benchmark and log artifacts, verifies every clip by full decode, concatenates the clips locally, verifies the final video, and records SHA256 values.

## Failure rule

An error, timeout, connection loss, or interruption exits nonzero with `FAILED_INSTANCE_PRESERVED` and an `operator-state.json`. Never automatically stop, recreate, or restart that instance. Read [references/failure-recovery.md](references/failure-recovery.md) and inspect the preserved state before choosing an intervention.

For either mode, the failure rule applies to the entire single-instance batch. Do not resubmit automatically after a failure.
