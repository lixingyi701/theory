"""Explicit upsert memory with provenance, deterministic eviction and sham hooks."""
from __future__ import annotations

from dataclasses import asdict, dataclass


class FormatError(ValueError):
    pass


def short_text(value, field: str, limit: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise FormatError(f"{field} must be a nonempty string of at most {limit} characters")
    return value.strip()


@dataclass
class Fact:
    entity: str
    attribute: str
    value: str
    source_step: int
    last_updated: int


class Memory:
    def __init__(self, capacity: int = 16, entries: list[dict] | None = None):
        if capacity < 1:
            raise ValueError("Memory capacity must be positive")
        self.capacity = capacity
        self.entries = [Fact(**f) for f in entries or []]
        if len(self.entries) > capacity:
            raise ValueError("Checkpoint exceeds memory capacity")

    def export(self) -> list[dict]:
        return [asdict(f) for f in self.entries]

    def update(self, payload: dict, step: int) -> dict:
        updates = payload.get("updates") if isinstance(payload, dict) else None
        if not isinstance(updates, list) or len(updates) > self.capacity:
            raise FormatError("updates must be a list no longer than memory capacity")
        # Validate the entire delta before any mutation; a retry cannot partly apply.
        validated = []
        for item in updates:
            if not isinstance(item, dict):
                raise FormatError("Each update must be an object")
            validated.append(Fact(short_text(item.get("entity"), "entity"),
                                  short_text(item.get("attribute"), "attribute"),
                                  short_text(item.get("value"), "value"), step, step))
        evicted = []
        for fact in validated:
            self.entries = [f for f in self.entries if (f.entity, f.attribute) != (fact.entity, fact.attribute)]
            self.entries.append(fact)
            if len(self.entries) > self.capacity:
                evicted.append(asdict(self.entries.pop(0)))
        return {"updates": len(validated), "evicted": evicted}

    def sham_verified_location(self, truth: list[dict] | None) -> dict | None:
        """Equal-content replacement of a currently true location fact.

        This is an instrumentation sham, not an LLM paraphrase intervention.
        Truth is used only to choose the evaluator hook, never to correct memory.
        """
        for index, fact in enumerate(self.entries):
            if fact.attribute.lower() not in {"location", "in", "on", "at"}:
                continue
            for proposition in truth or []:
                args = proposition.get("arguments", [])
                # ALFWorld uses inreceptacle for object -> named receptacle.
                # objectatlocation instead points to an internal location ID.
                if proposition["name"] in {"in", "on", "at", "inreceptacle"} and len(args) == 2:
                    if args[0]["name"].strip() == fact.entity and args[1]["name"].strip() == fact.value:
                        before = asdict(fact)
                        self.entries[index] = Fact(**before)
                        return {"applied": True, "kind": "verified_equal_content_replace",
                                "before": before, "after": asdict(self.entries[index]),
                                "semantic_change": False}
        return None
