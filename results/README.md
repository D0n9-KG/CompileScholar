# results/

Frozen runs whose numbers appear in the paper. This is the only place run outputs are tracked in git; working runs go
to `runs/` (ignored). Files over 1 MB are gzip-compressed (`mtime=0`, deterministic); each run's `MANIFEST.tsv` gives
the sha256 and size of every **original** (uncompressed) file and where it came from. Size rule for this folder: a
single file may exceed the repository-wide 5 MB limit only when it is the compressed output of a frozen run.

| Run | What | Headline |
|---|---|---|
| `cs2-test100-v9b-20261004/` | CS2 test, 100 questions, frozen v9b (pre-upgrade record): ours ×2 runs, Claude Code harness, GPT-Researcher (same 27B), archived Elicit / SciSpace / OpenAI DR re-judged; judge DeepSeek-V4.1-Flash (legacy adapter) | ours 0.829, SciSpace 0.837 (−0.008 n.s.), Elicit 0.798, harness 0.744, GPTR 0.736, OpenAI DR 0.684 (`paired_stats.txt`) |

Recompute the table from this folder:

```bash
python - <<'EOF'
import gzip, json, shutil, pathlib, tempfile
run = pathlib.Path("results/cs2-test100-v9b-20261004"); tmp = pathlib.Path(tempfile.mkdtemp())
for gz in run.rglob("scores_deepseek.json.gz"):
    with gzip.open(gz) as fi, open(tmp / f"{gz.parent.name}.json", "wb") as fo: shutil.copyfileobj(fi, fo)
print(tmp)
EOF
compilescholar score --split test --ref <tmp>/ours_r1.json,<tmp>/ours_r2.json --others harness=<tmp>/harness.json \
  gptr=<tmp>/gptr.json elicit=<tmp>/elicit.json scispace=<tmp>/scispace.json openai_dr=<tmp>/openai_dr.json
```
