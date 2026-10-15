"""Enumerate eligible official games; freeze deterministic, scene-aware manifests."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io import digest, file_hash, read_json, write_json

TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]


def enumerate_tasks(root: Path, split: str) -> tuple[list[dict], dict]:
    root = root.resolve()
    tasks, excluded = [], Counter()
    for path in sorted(root.rglob("traj_data.json")):
        relative = path.relative_to(root)
        if split not in relative.parts:
            continue
        if "movable" in str(relative) or "Sliced" in str(relative):
            excluded["unsupported_movable_or_slice"] += 1
            continue
        game = path.with_name("game.tw-pddl")
        if not game.exists():
            excluded["missing_game"] += 1
            continue
        trajectory, game_data = read_json(path), read_json(game)
        if trajectory.get("task_type") not in TASK_TYPES:
            excluded["unsupported_task_type"] += 1
            continue
        if not game_data.get("solvable", False):
            excluded["not_marked_solvable"] += 1
            continue
        scene = trajectory.get("scene", {}).get("floor_plan")
        if scene is None:
            excluded["missing_scene_id"] += 1
            continue
        gamefile = game.relative_to(root).as_posix()
        tasks.append({"task_id": digest(gamefile)[:16], "scene_id": str(scene),
                      "task_type": trajectory["task_type"], "split": split, "gamefile": gamefile,
                      "game_sha256": file_hash(game), "trajectory_sha256": file_hash(path)})
    return tasks, dict(excluded)


def make_manifest(root: Path, output: Path, split: str, per_type: int, seed: int) -> dict:
    if per_type < 1:
        raise ValueError("per_type must be positive")
    if output.exists():
        raise FileExistsError("Manifests are immutable; choose a new path")
    tasks, excluded = enumerate_tasks(root, split)
    selected, used_scenes = [], set()
    # Prefer distinct scenes globally but do not silently drop a task category.
    pools = {kind: sorted([t for t in tasks if t["task_type"] == kind],
                          key=lambda t: digest([seed, t["task_id"]])) for kind in TASK_TYPES}
    for _ in range(per_type):
        for kind in TASK_TYPES:
            available = [t for t in pools[kind] if t["scene_id"] not in used_scenes]
            if not available:
                raise ValueError(f"Not enough distinct scenes for {kind}; inspect dataset coverage")
            task = available[0]
            selected.append(task)
            used_scenes.add(task["scene_id"])
    payload = {"schema_version": 1, "split": split, "selection_seed": seed,
               "eligible_count": len(tasks), "excluded": excluded,
               "eligible_by_type": dict(Counter(t["task_type"] for t in tasks)),
               "scene_grouping": "unique_floor_plan", "tasks": selected}
    result = {**payload, "sha256": digest(payload)}
    write_json(output, result)
    return result


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    if digest(payload) != manifest.get("sha256"):
        raise ValueError("Manifest integrity mismatch")
    return manifest
