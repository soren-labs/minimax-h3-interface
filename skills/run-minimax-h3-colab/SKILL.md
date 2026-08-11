---
name: run-minimax-h3-colab
description: "Fully automate one or a batch of parameterized MiniMax H3 Turbo image-to-video jobs on one Google Colab G4: ask for the post-batch shutdown policy before launch, reuse one assignment across every queued prompt, upload the bundled CUDA 13/sm120 SageAttention wheel and inputs, install cu130 once, run H3 with native audio, verify every artifact, and stop only after the batch when requested. Use when asked to run H3 video from reference images with step count, resolution, duration, prompts, seeds, or native H3 audio while minimizing repeated setup and manual intervention."
---

# Run MiniMax H3 on Colab

Use the bundled orchestrator as the sole normal-path entry point. Do not reproduce its lifecycle with individual `colab` commands.

## Decide the batch lifecycle before launch

Before the first non-dry-run invocation:

1. Collect the complete known job queue, including every prompt and reference image.
2. Ask the user one blocking lifecycle question: after all currently queued jobs are downloaded and verified, should the Colab assignment be stopped or kept running for likely follow-up jobs?
3. Do not create or mutate a Colab assignment until the user answers `stop after batch` or `keep running`.

Run every queued job sequentially on one assignment. Never create a new assignment between prompts. For a batch of `N` jobs:

- Invoke jobs `1..N-1` with `--keep-on-success`.
- For job `N`, omit `--keep-on-success` only when the user chose `stop after batch`; otherwise include it.
- Download and verify each job before submitting the next one.
- If the user adds jobs while the assignment is retained, append them to the same assignment instead of starting another one.

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

## Failure rule

An error, timeout, connection loss, or interruption exits nonzero with `FAILED_INSTANCE_PRESERVED` and an `operator-state.json`. Never automatically stop, recreate, or restart that instance. Read [references/failure-recovery.md](references/failure-recovery.md) and inspect the preserved state before choosing an intervention.

For multiple queued jobs, `--keep-on-success` is mandatory on every non-final job. On the final job, follow the lifecycle choice collected before launch.
