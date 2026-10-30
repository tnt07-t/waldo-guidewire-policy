# Documentation guide

## For people

Read only what you need, in this order:

1. [APPROACHES.md](APPROACHES.md) — how the pipeline fits together, what each group's work adds, and the decisions still open.
2. [DATASET.md](DATASET.md) — reference for the supplied data, restrictions, and completed validation. No need to reread every check to use the project.
3. [LOADER.md](LOADER.md) — how to read a case and what comes back; use this when working with the loader.
4. Your group's progress: [Group 1](grp1/PROGRESS.md) or [Group 2](grp2/PROGRESS.md) — assignments, meaningful outcomes, and next steps.

## Working together

- Review constraints and tradeoffs before planning; agree on the plan before implementation. Ask when unsure, and keep proposals distinct from decisions.
- Keep code simple and comment non-obvious logic concisely. Do not change shared inputs or preprocessing without agreement.
- Update notes for meaningful outcomes, not individual commands. Keep the main explanations short and record results once.
- Keep real images, records, and sensitive outputs on Perlmutter, including notebook outputs.
- Fetch and inspect upstream changes, preserve local work, and reconcile before sharing changes. Keep unrelated work in separate commits and describe meaningful changes and verification.

## For AI agents

Read the project README, `APPROACHES.md`, `DATASET.md`, and the relevant group log before substantive work. Use `LOADER.md` for loader tasks. Distinguish confirmed facts from proposals, preserve existing work, and keep sensitive data on Perlmutter. Personal agent preferences are not part of these shared docs.
