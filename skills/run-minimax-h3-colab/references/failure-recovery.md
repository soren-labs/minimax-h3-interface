# Failure recovery

The orchestrator deliberately preserves the active Colab assignment on every non-success path. `operator-state.json` records the session, endpoint when known, normalized job config, output directory, and last error.

## `[?]` or missing local binding

`[?]` means the server assignment outlived local session state. It is not proof that the VM was reclaimed. The next normal invocation repairs a single assignment using its fresh runtime proxy token and starts a keep-alive process. It refuses to guess when multiple unbound assignments exist.

Never run `colab new` while any server assignment exists. Never delete an assignment merely because a local `exec`, upload, download, 401, or 404 failed.

## Connection or polling failure

1. Open the recorded `operator-state.json`.
2. Run `colab sessions` and identify the recorded endpoint.
3. Re-establish the local binding before doing anything to the VM.
4. Inspect `/content/h3-job-<job-id>.log`, the PID, GPU activity, and the expected archive.
5. If the remote PID is healthy, resume observation; do not launch a second job.
6. If the job completed, download `/content/h3-results-<job-id>.tar.gz` and the log and validate them. Preserve the assignment when more queued jobs remain or the user selected `keep running`; stop it only after the final verified job when the user selected `stop after batch`.

## Production failure

Inspect the preserved remote log and ComfyUI logs in `/content/h3-results-<job-id>/logs`. Decide explicitly whether the existing environment is repairable. A restart, stop, or new instance is an operator decision; the normal automation contains no mid-run shutdown or replacement branch.

## Success criteria

Treat a run as successful only when the remote success marker exists, the archive downloads, the tar can be opened, and `runs.json` is present. The normal script then marks the local state `complete`. It preserves the assignment with `--keep-on-success`; without that flag it stops the assignment after verification.

For a batch failure, do not submit the next queued job until the preserved instance and failed job have been inspected. Never work around a failed batch item by creating a second assignment.
