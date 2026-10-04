# artifacts/

Large data the code depends on but git does not carry (single files > 5 MB, vectors, caches). Each entry in
`MANIFEST.tsv` pins one file by sha256 and size; the files themselves stay at the listed paths on the working machine
and in the NAS backup (`\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-backup-20261004`).

- The answer KB used for every reported CS2/DSB number is `cs2/base_kb_v2/{papers,records,state_merged}.json` +
  `record_vecs.f32` (+ `.meta.json`), marked "answer KB (frozen v9b input)". `compilescholar verify` recomputes these
  hashes for a run.
- Files marked "intermediate / run output" were tracked before 2026-10-04 and were untracked then; their contents are
  still in the pushed git history at those paths.
- Publication: the KB snapshot will be released as a data artifact with this manifest (location and license pending,
  UPGRADE-PLAN decision 12).
