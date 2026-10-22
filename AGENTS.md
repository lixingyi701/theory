# Instructions for Codex and other coding agents

## Project and research scope

- Read `docs/IMPLEMENTATION_STATUS.md` and `docs/LINUX_RUNBOOK.md` before executing research jobs. The research protocol is under `docs/research/`.
- This is a standalone agent-reliability project. Do not import unrelated QuRO/VECTOR code or results.
- Preserve the distinction between code tests, real-environment replay checks, and LLM experiments. Never report fake-provider/backend outcomes as benchmark measurements.
- The reference baseline executes fixed upstream ReAct functions through `reference-run`. Current Reflexion stages use `scripts/run_reflexion_tool_stage.py`, the finite controller `scripts/continue_reflexion_normalization.py`, and paired `scripts/run_reflexion_pilot_forks.py`, with pinned author functions and explicit 8B serving/decoding identities. Read `docs/experiments/interface-normalization-20261023.md` and the Reflexion protocol/results. The dated raw `run_reflexion.py` / `run_reflexion_forks.py` entries preserve earlier runs. Custom `run` / `react-run` paths require `--allow-legacy-agent` and are historical evidence only. Mechanism calibration, the original R0-R3 matrix and task-length extrapolation remain future work.
- Execute research on the server: project, runner, data, model, journals and evaluator are server-local. Keep the workstation as a source/control client; do not recreate a local experiment environment by default. Read `docs/experiments/reference-alignment-20261022.md` first.

## Running work

- Downloads and model launch scripts default to plan-only. Execute them when the active user instruction authorizes deployment; do not start a long job solely because an installation command finished.
- Once execution is authorized, follow the documented stage order. A failed replay audit blocks the dependent clean/sham job. Fix the concrete failure without weakening the audit or silently dropping failed tasks.
- Use the fixed model revisions and ALFWorld commit. Save runtime versions, task/data hashes, prompts and configurations. Do not silently switch models, quantization, token limits or environment versions.
- Never pass environment facts, planner commands or evaluator-only annotations into policy or memory-updater prompts.
- Keep manifests immutable. Same base task, scene, seeds and future intervention branches must remain in the same data partition.
- No automatic network retry of an uncertain model request. Inspect the saved request and use `--retry-uncertain` only when the extra dispatch is intended; count it in costs.
- Keep success, environment failure, agent format failure, budget exhaustion and infrastructure errors distinct.
- Apply the frozen model/tool contract in `configs/reflexion_qwen3_8b_protocol.json`. One optional action-line marker and `put X in/on Y` / `in Y` / `on Y` are representation aliases; preserve chosen objects, targets and indices. Keep raw responses, decoded actions and submitted commands. Do not guess semantic repairs from environment facts.
- Record the decoding profile explicitly. Author retry-temperature decoding and Qwen recommended non-thinking sampling are separate configurations. New serving responses include effective sampling values; audit them with `scripts/audit_reflexion_protocol.py` before freezing a research interface.

## Code and data

- Run `bash scripts/offline_checks.sh` for changes to critical execution paths. Add checks only for a concrete uncovered failure.
- Put weights in `models/`, benchmark data in `data/`, manifests in `manifests/`, and logs/results in `runs/`; these are ignored by git.
- Do not commit API keys, `.env` files, full model responses, downloaded datasets or upstream PDFs. Do not print secrets in error reports.
- The implementation uses per-step JSON artifacts plus a committed checkpoint. Do not reconstruct progress from the mere presence of uncommitted step files.
- Publish factual summaries with exact run/config/task coverage. Missing data are missing, not zeros or negative results.
- Do not overwrite earlier failed audit evidence; save a new report after a fix. Document which code/config changed.
- Keep operational instructions executable from the repository root on Linux.
