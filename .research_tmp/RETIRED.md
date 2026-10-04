# Retired to cold storage (2026-10-04)

These directories held retired experiments (PaperScope / AirQA archive, baseline model weights, the E2 need-gap
study, Stage B pilots, one-off scratch, the 09-26 benchmark audit). Each was copied to the NAS, every file was
verified by size and sha256 against the original (per-directory reports in `_logs/verify_*.json` there), and only
then deleted here. Two tracked scripts under stageB were among them; they remain in git history.

| Was | Now | Files | Size |
|---|---|---|---|
| `.research_tmp/experiments/archive/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\experiments\archive` | 52,610 | 20.0 GB |
| `.research_tmp/experiments/baselines/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\experiments\baselines` | 23,842 | 4.2 GB |
| `.research_tmp/experiments/e2_need_gap/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\experiments\e2_need_gap` | 7,164 | 2.4 GB |
| `.research_tmp/experiments/stageB/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\experiments\stageB` | 1,191 | 1.0 GB |
| `.research_tmp/archive_oneoff/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\archive_oneoff` | 3,313 | 0.1 GB |
| `.research_tmp/benchmark-audit-0926/` | `\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\benchmark-audit-0926` | 29 | 0.1 GB |

Restore one with `robocopy <Now> <Was> /E`. No reported number depends on them (see `legacy/INDEX.md`).
