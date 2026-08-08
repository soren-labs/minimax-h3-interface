---
name: run-minimax-h3-colab
description: "Fully automate a parameterized MiniMax H3 Turbo image-to-video production job on a Google Colab G4: safely reuse or rebind [?] assignments, create a G4 only when none exists, upload the bundled CUDA 13/sm120 SageAttention wheel and inputs, install cu130, launch ComfyUI and the H3 workflow in the background, stream logs, download and verify all artifacts, and stop the instance only after success. Use when asked to run H3 video from a reference image with step count, resolution, duration, prompt, seed, or native H3 audio while minimizing manual agent intervention."
---

# Run MiniMax H3 on Colab

Use the bundled orchestrator as the sole normal-path entry point. Do not reproduce its lifecycle with individual `colab` commands.

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
7. Stops the Colab assignment only after verified success.

The script normalizes dimensions to multiples of 32 and duration to H3's `5 + 17*n` frame grid, and prints the effective values before starting. Use `--frames` when exact grid control is needed. Use `--dry-run` to validate parameters without touching Colab.

## Failure rule

An error, timeout, connection loss, or interruption exits nonzero with `FAILED_INSTANCE_PRESERVED` and an `operator-state.json`. Never automatically stop, recreate, or restart that instance. Read [references/failure-recovery.md](references/failure-recovery.md) and inspect the preserved state before choosing an intervention.

Do not use `--keep-on-success` unless the user explicitly asks to retain a successfully completed instance.
