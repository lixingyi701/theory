"""Commands are separate so setup never launches a benchmark or model download."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from .audit import audit_manifest, verify_audit
from .dataset import load_manifest, make_manifest
from .env import ReplayEnvironment
from .io import digest, file_hash, git_revision, read_json, write_json, run_lock
from .llm import endpoint_from_config
from .runner import run_episode, validate_config
from .react import run_react_episode, validate_react_config


def backend(root: Path, expert: bool = False):
    from .alfworld import ALFWorldBackend
    return ReplayEnvironment(ALFWorldBackend(root, expert=expert))


def provenance() -> dict:
    packages = {}
    for name in ["agent-reliability-theory", "alfworld", "textworld", "numpy", "fast-downward-textworld"]:
        try:
            distribution = importlib.metadata.distribution(name)
            direct = distribution.read_text("direct_url.json")
            packages[name] = {"version": distribution.version,
                              "direct_url": json.loads(direct) if direct else None}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    native = {}
    try:
        library = Path(importlib.metadata.distribution("fast-downward-textworld").locate_file("fast_downward/libdownward.so"))
        if library.exists():
            native["fast_downward/libdownward.so"] = file_hash(library)
    except importlib.metadata.PackageNotFoundError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libc": platform.libc_ver(), "git": git_revision(), "packages": packages,
            "source_hashes": source, "native_hashes": native}


def summarize(directory: Path) -> dict:
    results = [read_json(p) for p in directory.glob("episodes/*/result.json")]
    conditions = {}
    run_metadata = read_json(directory / "run.json") if (directory / "run.json").exists() else {}
    for condition in run_metadata.get("conditions", ["clean", "sham"]):
        rows = [r for r in results if r["condition"] == condition]
        conditions[condition] = {
            "completed_episodes": len(rows), "successes": sum(r["success"] for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "sham_applied": sum(r["sham_applied"] for r in rows),
            "env_steps": sum(r["env_steps"] for r in rows),
            "dispatched_calls": sum(r["dispatched_calls"] for r in rows),
            "input_tokens_known": sum(r["input_tokens_known"] for r in rows),
            "output_tokens_known": sum(r["output_tokens_known"] for r in rows),
            "uncertain_or_unrecorded_attempts": sum(r["uncertain_or_unrecorded_attempts"] for r in rows),
            "format_invalid_responses": sum(r["format_invalid_responses"] for r in rows),
            "truncated_responses": sum(r["truncated_responses"] for r in rows),
        }
    expected = None
    if (directory / "run.json").exists():
        expected = read_json(directory / "run.json")["expected_episodes"]
    report = {"schema_version": 1, "expected_episodes": expected, "completed_episodes": len(results),
              "complete": expected is not None and len(results) == expected,
              "conditions": conditions,
              "scope": "Engineering clean/sham checks; not residual-error evidence"}
    write_json(directory / "summary.json", report)
    return report


def run_command(args) -> dict:
    manifest = load_manifest(args.manifest)
    config, models = read_json(args.config), read_json(args.models)
    is_react = args.command == "react-run"
    (validate_react_config if is_react else validate_config)(config)
    model = models[config["model_key"]]
    audit = read_json(args.audit_report)
    verify_audit(audit, manifest)
    current = provenance()
    for name in ["alfworld.py", "env.py", "audit.py"]:
        if current["source_hashes"][name] != audit.get("provenance", {}).get("source_hashes", {}).get(name):
            raise ValueError(f"{name} changed after replay audit; rerun the audit")
    # Guard the ALFWorld source installation, not only its version string.
    alf = current["packages"].get("alfworld") or {}
    commit = ((alf.get("direct_url") or {}).get("vcs_info") or {}).get("commit_id")
    from .alfworld import ALFWORLD_COMMIT
    if commit != ALFWORLD_COMMIT:
        raise ValueError("ALFWorld must be installed from the pinned Git commit (see requirements/runner.txt)")
    probe = backend(args.data_root)
    try:
        actual_identity = probe.backend.identity
        recorded_identity = {**audit["backend"], "expert": False}
        if actual_identity != recorded_identity:
            raise ValueError("Environment versions/config changed after replay audit")
    finally:
        probe.close()
    transport = endpoint_from_config(config)
    health = transport.health(model["served_name"])
    identity = {"manifest_sha256": manifest["sha256"], "config": config, "model": model,
                "audit_sha256": audit["sha256"], "source_hashes": current["source_hashes"]}
    run_file = args.output / "run.json"
    if run_file.exists():
        if not args.resume or read_json(run_file)["identity_sha256"] != digest(identity):
            raise ValueError("Existing run/config mismatch; use --resume or a new directory")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Output directory is nonempty without run metadata")
        write_json(run_file, {**identity, "identity_sha256": digest(identity), "provenance": current,
                              "endpoint": health, "conditions": config["conditions"],
                              "expected_episodes": len(manifest["tasks"]) * len(config["seeds"]) * len(config["conditions"])})
    for task in manifest["tasks"]:
        for seed in config["seeds"]:
            for condition in config["conditions"]:
                episode = args.output / "episodes" / f"{task['task_id']}-s{seed}-{condition}"
                env = backend(args.data_root)
                try:
                    execute = run_react_episode if is_react else run_episode
                    result = execute(env, transport, task, config, model, seed, condition,
                                         episode, args.resume, args.retry_uncertain)
                finally:
                    env.close()
                print(json.dumps({k: result[k] for k in ["task_id", "seed", "condition", "status", "env_steps"]}), flush=True)
                stop_after = config.get("early_stop_zero_successes_after")
                if is_react and condition == "base" and stop_after:
                    rows = [read_json(p) for p in args.output.glob("episodes/*-base/result.json")]
                    if len(rows) >= stop_after and not any(r["success"] for r in rows):
                        print("Predeclared zero-success stop; remaining episodes are missing", flush=True)
                        return summarize(args.output)
    return summarize(args.output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect installed packages; no download or model call")
    doctor.add_argument("--output", type=Path)
    manifest = commands.add_parser("manifest", help="Freeze a balanced local-data task list")
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"], default="train")
    manifest.add_argument("--per-type", type=int, default=4)
    manifest.add_argument("--seed", type=int, default=20261015)
    manifest.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit", help="No-LLM replay audit, 10 tasks / 40 restores by default")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--tasks", type=int, default=10)
    audit.add_argument("--output", type=Path, required=True)
    for name, default in [("run", "clean_sham.json"), ("react-run", "react_base.json")]:
        run = commands.add_parser(name, help="Run a frozen, replay-audited task list")
        run.add_argument("--data-root", type=Path, required=True)
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--audit-report", type=Path, required=True)
        run.add_argument("--config", type=Path, default=Path("configs") / default)
        run.add_argument("--models", type=Path, default=Path("configs/models.json"))
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--resume", action="store_true")
        run.add_argument("--allow-legacy-agent", action="store_true",
                         help="Explicit historical reproduction only; use reference-run for new baselines")
        run.add_argument("--retry-uncertain", action="store_true",
                         help="Explicitly redispatch unresolved calls, counting extra attempts")
    reference = commands.add_parser("reference-run", help="Run the author's unchanged ReAct ALFWorld functions")
    reference.add_argument("--data-root", type=Path, required=True)
    reference.add_argument("--manifest", type=Path, required=True)
    reference.add_argument("--audit-report", type=Path, required=True)
    reference.add_argument("--environment-report", type=Path, required=True)
    reference.add_argument("--config", type=Path, default=Path("configs/reference_react_8b.json"))
    reference.add_argument("--models", type=Path, default=Path("configs/models.json"))
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--resume", action="store_true")
    reference.add_argument("--retry-uncertain", action="store_true")
    reference.add_argument("--stop-after-episodes", type=int, help="Pause at an episode boundary; retain full manifest denominator")
    summary = commands.add_parser("summarize")
    summary.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = provenance()
            if args.output:
                write_json(args.output, result)
        elif args.command == "manifest":
            result = make_manifest(args.data_root, args.output, args.split, args.per_type, args.seed)
            result = {k: v for k, v in result.items() if k != "tasks"} | {"selected": len(result["tasks"])}
        elif args.command == "audit":
            if args.output.exists():
                raise FileExistsError("Audit output exists; choose a new name to retain the failed evidence")
            result = audit_manifest(lambda: backend(args.data_root, expert=False),
                                    load_manifest(args.manifest), args.tasks,
                                    reference_factory=lambda: backend(args.data_root, expert=True))
            result["provenance"] = provenance()
            result.pop("sha256")
            result["sha256"] = digest(result)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["passed"] else 1
        elif args.command in {"run", "react-run"}:
            if not args.allow_legacy_agent:
                raise ValueError("Self-built agents are historical only; use reference-run or explicitly --allow-legacy-agent")
            with run_lock(args.output):
                result = run_command(args)
        elif args.command == "reference-run":
            from .reference_run import run_reference_command
            with run_lock(args.output):
                result = run_reference_command(args)
        else:
            run_meta = read_json(args.directory / "run.json") if (args.directory / "run.json").exists() else {}
            if run_meta.get("config", {}).get("method") == "author_react_alfworld":
                from .reference_run import summarize_reference
                result = summarize_reference(args.directory)
            else:
                result = summarize(args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
