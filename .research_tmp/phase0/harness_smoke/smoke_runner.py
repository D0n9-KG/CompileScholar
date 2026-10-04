# -*- coding: utf-8 -*-
"""Phase 0.1 harness smoke runner.

Runs a fixed multi-step tool-loop task suite against a harness CLI
(claude | codex) backed by the local GPUStack Qwen3.8-27B, then
validates artifacts and writes a results table.

Usage (from .research_tmp/phase0/harness_smoke):
  python smoke_runner.py claude
  python smoke_runner.py codex --only T1,T3
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "workspace"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TASK_TIMEOUT_S = 900
MAX_TURNS = 20


def load_env_file(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip()
    return env


DOTENV = load_env_file(HERE.parents[2] / ".env")
LOCAL_KEY = DOTENV["LOCAL_API_KEY"]

COMMON_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "GPUSTACK_API_KEY": LOCAL_KEY,
}
CLAUDE_ENV = {
    "ANTHROPIC_BASE_URL": "http://192.168.199.73",
    "ANTHROPIC_MODEL": "Qwen3.8-27B",
    "ANTHROPIC_SMALL_FAST_MODEL": "Qwen3.8-27B",
    "ANTHROPIC_AUTH_TOKEN": LOCAL_KEY,
}


# ------------------------------------------------------------- validation

def validate_json_file(name, check):
    def v(result_text):
        p = WORKSPACE / name
        if not p.exists():
            return False, "artifact missing"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"invalid JSON: {e}"
        try:
            ok = check(data)
        except Exception as e:
            return False, f"schema check error: {e}"
        return (True, "ok") if ok else (False, "schema check failed")

    return v


def extract_inline_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def inline_ok(check):
    def v(result_text):
        data = extract_inline_json(result_text)
        if data is None:
            return False, "no parseable JSON in result"
        return check(data)

    return v


def expected_total_lines():
    return sum(
        len(p.read_text(encoding="utf-8").splitlines())
        for p in (WORKSPACE / "papers").glob("*.txt")
    )


# ---------------------------------------------------------------- tasks
# each task: id, prompt, allowed (claude tool list), validate(result_text)

TASKS = [
    dict(
        id="T1",
        prompt=(
            "Read the file papers/paperA.txt, then reply with ONLY this JSON "
            '(no markdown fences): {"paper": "A", "published": "<date from '
            'file>", "key_result": "<one sentence>"}'
        ),
        allowed="Read",
        validate=inline_ok(
            lambda d: (
                ("2020-01-23" in json.dumps(d)),
                "ok" if "2020-01-23" in json.dumps(d) else "date wrong",
            )
        ),
    ),
    dict(
        id="T2",
        prompt=(
            "Read all three files in the papers/ directory. Write a file "
            "out_T2.json containing a JSON array with one object per paper: "
            '{"file": "<filename>", "published": "<date>", "key_result": '
            '"<one sentence>", "conditions": "<one sentence>"}. Reply DONE '
            "when written. Do not include markdown fences in the file."
        ),
        allowed="Read Glob Write",
        validate=validate_json_file(
            "out_T2.json",
            lambda d: isinstance(d, list)
            and len(d) == 3
            and all(
                set(o) >= {"file", "published", "key_result", "conditions"}
                for o in d
            ),
        ),
    ),
    dict(
        id="T3",
        prompt=(
            "Search the papers/ directory for the word 'Limitation' (case "
            "insensitive). For every paper that mentions it, reply with ONLY "
            "a JSON array of objects "
            '[{"file": "...", "limitation": "<full limitation sentence>"}] '
            "(no markdown fences)."
        ),
        allowed="Read Grep Glob",
        validate=inline_ok(
            lambda d: (
                isinstance(d, list) and len(d) == 3,
                "ok"
                if isinstance(d, list) and len(d) == 3
                else f"expected 3 limitations, got {len(d) if isinstance(d, list) else type(d)}",
            )
        ),
    ),
    dict(
        id="T4",
        prompt=(
            "Using shell commands only (not by reading the files with the "
            "Read tool), count the total number of lines across all files in "
            "papers/. Then reply with ONLY "
            '{"total_lines": <number>, "files_counted": <number>} (no '
            "markdown fences)."
        ),
        allowed="Bash",
        validate=inline_ok(
            lambda d: (
                d.get("total_lines") == expected_total_lines()
                and d.get("files_counted") == 3,
                "ok"
                if d.get("total_lines") == expected_total_lines()
                else f"total_lines={d.get('total_lines')} expected {expected_total_lines()}",
            )
        ),
    ),
    dict(
        id="T5",
        prompt=(
            "Compile a mini research report answering: 'What is known about "
            "compute efficiency in transformer training, according to the "
            "papers in papers/ ?' Write it to out_T5.json with exactly this "
            'shape: {"sections": [{"heading": str, "claims": [{"text": str, '
            '"source_file": str, "source_line": int}]}], "open_questions": '
            "[str]}. Every claim must cite the file and line number you "
            "actually read. Use at least 2 sections and at least 4 claims "
            "total. Reply DONE when written."
        ),
        allowed="Read Glob Grep Write",
        validate=validate_json_file(
            "out_T5.json",
            lambda d: isinstance(d, dict)
            and isinstance(d.get("sections"), list)
            and len(d["sections"]) >= 2
            and sum(len(s.get("claims", [])) for s in d["sections"]) >= 4
            and all(
                (WORKSPACE / c["source_file"]).exists()
                and isinstance(c.get("source_line"), int)
                for s in d["sections"]
                for c in s.get("claims", [])
            ),
        ),
    ),
    dict(
        id="T6",
        prompt=(
            "The file broken_T6.json is invalid JSON. Fix it so that "
            "`python -m json.tool broken_T6.json` succeeds — verify by "
            "actually running that command. Then add key \"fixed\": true. "
            "Reply DONE when verified."
        ),
        allowed="Read Write Bash",
        validate=validate_json_file(
            "broken_T6.json",
            lambda d: d.get("fixed") is True and d.get("papers") == 3,
        ),
    ),
]


# ---------------------------------------------------------------- runners

# npm 全局安装的 CLI 在 Windows 上是 .cmd 壳，%* 会把 prompt 里的 <>&|
# 当重定向符二次解析（实测 "<date from file>" 触发 file-not-found），
# 因此一律直调包内原生 exe，绕过 cmd.exe。
NPM_ROOT = Path(os.environ.get("APPDATA", "")) / "npm"
CLAUDE_EXE = str(
    NPM_ROOT / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
)
CODEX_EXE = str(
    NPM_ROOT / "node_modules" / "@openai" / "codex" / "node_modules"
    / "@openai" / "codex-win32-x64" / "vendor" / "x86_64-pc-windows-msvc"
    / "bin" / "codex.exe"
)


def run_claude(task, env):
    cmd = [
        CLAUDE_EXE, "-p", task["prompt"],
        "--model", "Qwen3.8-27B",
        "--output-format", "json",
        "--max-turns", str(MAX_TURNS),
        "--allowedTools", task["allowed"],
    ]
    return subprocess.run(
        cmd, cwd=WORKSPACE, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=TASK_TIMEOUT_S,
    )


def run_codex(task, env):
    cmd = [
        CODEX_EXE, "exec",
        "--skip-git-repo-check",
        "-c", "model_provider=gpustack",
        "-c", "model=Qwen3.8-27B",
        "-c", "sandbox_mode=workspace-write",
        task["prompt"],
    ]
    return subprocess.run(
        cmd, cwd=WORKSPACE, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=TASK_TIMEOUT_S,
    )


RUNNERS = {"claude": run_claude, "codex": run_codex}


# ------------------------------------------------------------------- main

def main():
    harness = sys.argv[1] if len(sys.argv) > 1 else "claude"
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))

    if harness not in RUNNERS:
        sys.exit(f"unknown harness: {harness}")
    run = RUNNERS[harness]
    if harness == "claude":
        env = {**os.environ.copy(), **CLAUDE_ENV}
    else:
        env = {**os.environ.copy(), **COMMON_ENV}

    results = []
    for task in TASKS:
        if only and task["id"] not in only:
            continue
        # reset artifacts that the task itself writes
        for f in ("out_T2.json", "out_T5.json", "broken_T6.json"):
            p = WORKSPACE / f
            if p.exists():
                p.unlink()
        if task["id"] == "T6":
            (WORKSPACE / "broken_T6.json").write_text(
                '{"papers": 3, "topic": "scaling laws", "tags": ["lm", "compute",]}'
                ' "note": "missing comma above"}',
                encoding="utf-8",
            )
        print(f"[{harness}] {task['id']} running...", flush=True)
        t0 = time.time()
        rec = {
            "harness": harness,
            "task": task["id"],
            "timeout_s": TASK_TIMEOUT_S,
        }
        try:
            proc = run(task, env)
            rec["returncode"] = proc.returncode
            rec["elapsed_s"] = round(time.time() - t0, 1)
            out = proc.stdout or ""
            result_text = ""
            if harness == "claude":
                try:
                    envelope = json.loads(out.strip().splitlines()[-1])
                    result_text = envelope.get("result", "")
                    rec["num_turns"] = envelope.get("num_turns")
                    rec["is_error"] = envelope.get("is_error")
                    rec["session_cost"] = envelope.get("total_cost_usd")
                except Exception:
                    result_text = out
                    rec["envelope_parse"] = "failed"
            else:
                result_text = out
            rec["result_text_head"] = result_text[:800]
            rec["stderr_tail"] = (proc.stderr or "")[-800:]
            ok, why = task["validate"](result_text)
            rec["pass"] = bool(ok)
            rec["fail_reason"] = why
        except subprocess.TimeoutExpired:
            rec["timeout"] = True
            rec["elapsed_s"] = round(time.time() - t0, 1)
            rec["pass"] = False
            rec["fail_reason"] = "timeout"
        except Exception as e:
            rec["pass"] = False
            rec["fail_reason"] = f"runner error: {e}"
            rec["elapsed_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(
            f"[{harness}] {task['id']} -> "
            f"{'PASS' if rec.get('pass') else 'FAIL'} "
            f"({rec.get('fail_reason', '')}) {rec.get('elapsed_s', '?')}s",
            flush=True,
        )

    out_path = RESULTS_DIR / f"{harness}_results.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    npass = sum(1 for r in results if r.get("pass"))
    print(f"\n=== {harness}: {npass}/{len(results)} PASS ===")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
