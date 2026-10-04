# -*- coding: utf-8 -*-
"""W6 S8: promote the frozen CS2 test run (v9b) into results/ — the only place run outputs are tracked.

Each file is gzip-compressed (deterministic: mtime 0) when larger than 1 MB; results/<run>/MANIFEST.tsv lists
path, sha256 of the ORIGINAL uncompressed file, bytes, and the original location. Originals stay where they are.
"""
import gzip
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
C = REPO / ".research_tmp" / "experiments" / "benchmarks" / "cs2"
RUN = REPO / "results" / "cs2-test100-v9b-20261004"
FILES = {
    "ours_r1/answers.json": C / "arm_vnext" / "answers_test100_r1.json",
    "ours_r1/judge_input.json": C / "judge_input_vnext_test100_r1.json",
    "ours_r1/scores_deepseek.json": C / "direct_scores_vnext_test100_r1_ds.json",
    "ours_r1/config.json": C / "arm_vnext" / "config_test100_r1.json",
    "ours_r2/answers.json": C / "arm_vnext" / "answers_test100_r2.json",
    "ours_r2/judge_input.json": C / "judge_input_vnext_test100_r2.json",
    "ours_r2/scores_deepseek.json": C / "direct_scores_vnext_test100_r2_ds.json",
    "ours_r2/config.json": C / "arm_vnext" / "config_test100_r2.json",
    "harness/answers.json": C / "arm_harness" / "answers_harness_test100.json",
    "harness/judge_input.json": C / "judge_input_harness_test100.json",
    "harness/scores_deepseek.json": C / "direct_scores_harness_test100_ds.json",
    "gptr/answers.json": C / "arm_gptr" / "answers_gptr_test100.json",
    "gptr/judge_input.json": C / "arm_gptr" / "judge_input_gptr_test100.json",
    "gptr/scores_deepseek.json": C / "direct_scores_gptr_test100_ds.json",
    "elicit/judge_input.json": C / "judge_input_elicit_test.json",
    "elicit/scores_deepseek.json": C / "direct_scores_elicit_test_ds.json",
    "scispace/judge_input.json": C / "judge_input_scispace_test.json",
    "scispace/scores_deepseek.json": C / "direct_scores_scispace_test_ds.json",
    "openai_dr/judge_input.json": C / "judge_input_openai_dr_test.json",
    "openai_dr/scores_deepseek.json": C / "direct_scores_openai_dr_test_ds.json",
    "paired_stats.txt": C / "paired_stats_test100_v9b.txt",
    "FREEZE.json": C / "FREEZE_CS2_TEST_1003_v9b.json",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows = ["path\tsha256_original\tbytes_original\tsource"]
for rel, src in FILES.items():
    if not src.exists():
        print("MISSING", src)
        continue
    dst = RUN / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size > 1 << 20:
        dst = dst.with_name(dst.name + ".gz")
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            with gzip.GzipFile(fileobj=fo, mode="wb", mtime=0, compresslevel=9) as gz:
                shutil.copyfileobj(fi, gz)
    else:
        shutil.copyfile(src, dst)
    rows.append(f"{dst.relative_to(RUN).as_posix()}\t{sha(src)}\t{src.stat().st_size}\t{src.relative_to(REPO).as_posix()}")
(RUN / "MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
big = [p for p in RUN.rglob("*") if p.is_file() and p.stat().st_size > 5 << 20]
print(f"{len(rows) - 1} files -> {RUN}; total {sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) / 1e6:.1f} MB; "
      f">5MB after compression: {[p.name for p in big]}")
