# Phase 7 Fixed Regression Set Notes

This file freezes the first ten-paper regression set for the Phase 7 corpus-sampling baseline.

## Why This Set Stays Stable

- The set is intentionally fixed so Phase 8 and later iterations can compare compiler behavior against the same paper ids instead of re-randomizing the baseline every cycle.
- The committed manifest stores only `corpus_paper_id`, `display_title`, and `corpus_relative_ref` so the repo stays portable and does not capture the machine-local UNC share path.
- Updates to this set should be deliberate review events, not a side effect of rerunning the CLI with a new seed.

## Path And Format Coverage

The first fixed set is deliberately mixed rather than taking the first ten sorted rows blindly.

- `1000`, `1001`, and `1005` are healthy markdown-plus-text pairs where markdown remains the preferred source.
- `1017` and `1023` have both variants but fall back to `txt` because the markdown twin is unreadable; these entries prove fallback behavior is part of the stable baseline.
- `1002`, `1004`, `1007`, `1010`, and `1012` are `txt`-only eligible papers whose nested markdown directories are missing, which keeps corpus-health separation visible in the baseline report.

## Selection Heuristic

The current ten papers were chosen to keep three things true at once:

1. The set stays inside one broadly related data-driven / computational-mechanics slice so later quality comparisons are interpretable.
2. The set spans multiple path shapes: healthy markdown-first pairs, markdown-broken txt fallbacks, and txt-only survivors.
3. The set includes several different modeling phrasings such as constitutive response, multiscale mechanics, deep material networks, reinforcement learning, and geometric deep learning.

## How To Update Later

- Review the current fixed-set report and identify which path shapes or topic patterns are underrepresented.
- Replace entries intentionally, keeping the file at exactly ten rows unless the milestone requirements change.
- Preserve the same field schema and continue using corpus-relative refs only.
- Record the rationale for each new row so reviewers can tell the difference between a curated refresh and an accidental resort.
