# Skin Lesion R&D: working notes for Claude

Goal: one Q1 journal paper (not a thesis). The workflow is code-first; benchmark results feed back into the proposal.

At the start of each session, read [SUMMARY.md](SUMMARY.md), [TASKS.md](TASKS.md) and [KNOWLEDGE.md](KNOWLEDGE.md).

Keep them current:
- New information from the user, or a verified fact → KNOWLEDGE.md (sections 1–3).
- Task done or added → TASKS.md.
- Milestone reached → one row in SUMMARY.md's progress log, and update its status line.

Rules:
- Split HAM10000 by `lesion_id`, never by image.
- Raw data in `data/raw/` is read-only.
- Run Python that reads downloaded data with `python3 -I`.
