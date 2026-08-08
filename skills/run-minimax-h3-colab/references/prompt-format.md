# H3 I2VA prompt format

Follow MiniMax's official H3 prompt-writing structure for a first-frame image-to-video job. Write the rewrite sections in English while preserving spoken dialogue, lyrics, and visible text in their original language.

Use this field order:

1. State exactly when `<Picture 1>` is referenced, normally at `0.00 seconds`, and associate it with the shot.
2. `integrated_multimodal_description`: describe the continuous audiovisual timeline, including composition, subjects, environment, actions, camera movement, physical continuity, and the exact timing of major changes. Preserve the supplied first frame and avoid unresolved reference labels.
3. `overall_soundscape`: describe diegetic ambience, effects, voices, spatial placement, and timing. This drives H3's native audio path.
4. `non_diegetic_music`: describe score, instruments, tempo, emotional progression, and ending behavior; explicitly say none when no score is wanted.

Match all timing to the effective duration printed by the orchestrator. For architectural or landscape keyframes, specify controlled camera amplitude, parallax layers, structural stability, environmental motion, and prohibited cuts/morphing/text where relevant.

Official source: https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing
