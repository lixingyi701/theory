# Instructions for Codex and other coding agents

## Project and research scope

- Read `docs/IMPLEMENTATION_STATUS.md` and `docs/LINUX_RUNBOOK.md` before executing research jobs. The research protocol is under `docs/research/`.
- This is a standalone agent-reliability project. Do not import unrelated QuRO/VECTOR code or results.
- The current paper is theory led: prioritize the process model, assumptions and derivations in `docs/research/recovery_theory_framework.md`, and the finite-window propositions in `docs/research/finite_memory_recovery_analysis.md`, followed by `docs/experiments/finite-memory-simulation-design.md` and targeted real-Agent validation. Every new LLM batch must identify the proposition, unknown parameter or model assumption it addresses; the earlier EC/CE expansion is conditional on that need. `scripts/check_recovery_analysis.py` checks algebra and remaps existing evidence; its constructed examples are not ALFWorld measurements or completed simulation evidence.
- Use explicit terminology: an error generated after an error-absent start differs from restating an already visible error. Reserve "reappearance" for a documented disappearance followed by a later occurrence. Name local proposition correction and whole-reflection deletion separately. Distinguish finite-budget success gain, expected attempts saved and eventual success. Empty slots are not neutral reflection entries; deletion changes alignment and eviction times. `scripts/check_finite_memory_analysis.py` performs exact algebra checks, not Monte Carlo or Agent experiments.
- Preserve the distinction between code tests, real-environment replay checks, and LLM experiments. Never report fake-provider/backend outcomes as benchmark measurements.
- The reference baseline executes fixed upstream ReAct functions through `reference-run`. Reflexion normalization uses `scripts/run_reflexion_tool_stage.py` and paired `scripts/run_reflexion_pilot_forks.py`. The subsequent procedural-error study uses `scripts/run_reflexion_precondition_forks.py`; controlled window collection, immutable fitting and evaluation use `scripts/run_reflexion_state_prediction.py`, `scripts/fit_reflexion_state_prediction.py` and `scripts/evaluate_reflexion_state_prediction.py`. Read the interface protocol, `docs/experiments/precondition-pilot-20261023.md` and `docs/experiments/state-prediction-20261023.md`. Keep the pinned author functions and explicit 8B serving/decoding identities. Dated raw runners and custom `run` / `react-run` paths preserve historical evidence; custom agents require `--allow-legacy-agent`. Formal transition calibration, independent-task prediction, the original R0-R3 matrix and task-length extrapolation remain future work.
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
- Audit executed output shapes separately with `scripts/audit_reflexion_action_shapes.py`. A passing wiring audit does not establish that every model action is valid. Keep observation sentences, unmatched command forms, and no-effect commands separate; a recognized command shape does not establish correct entities or satisfied preconditions. Never turn an observation sentence into a guessed action.

## Code and data

- Run `bash scripts/offline_checks.sh` for changes to critical execution paths. Add checks only for a concrete uncovered failure.
- Put weights in `models/`, benchmark data in `data/`, manifests in `manifests/`, and logs/results in `runs/`; these are ignored by git.
- Do not commit API keys, `.env` files, full model responses, downloaded datasets or upstream PDFs. Do not print secrets in error reports.
- The implementation uses per-step JSON artifacts plus a committed checkpoint. Do not reconstruct progress from the mere presence of uncommitted step files.
- Publish factual summaries with exact run/config/task coverage. Missing data are missing, not zeros or negative results.
- The first controlled-window study has four known development tasks, 28 homogeneous calibration cells (12 reused, 16 new), and eight prospective mixed-state cells. It does not establish a prediction advantage: all mixed cells failed and the count/slot forecasts scored worse than the constant baseline. Preserve its parameter/prediction freezes and distinguish deliberate memory duplication from naturally generated history. New formal validation uses new base tasks; do not refit against this study and call its old outcomes held out.
- Do not overwrite earlier failed audit evidence; save a new report after a fix. Document which code/config changed.
- Keep operational instructions executable from the repository root on Linux.
