"""Resume/journaling for Workflow runs — cache agent() calls for replay."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .agent_runtime import TokenUsage

_MAX_ITEM_CAP = 4096
_MAX_AGENT_CALLS = 1000


def default_journal_dir() -> Path:
    base = os.environ.get("DIALECTICA_WORKFLOW_JOURNAL_DIR", ".dialectica/workflows")
    return Path(base)


def script_fingerprint(script: Any, *, registry_name: str | None = None) -> str:
    if registry_name:
        return f"registry:{registry_name}"
    module = getattr(script, "__module__", "")
    qualname = getattr(script, "__qualname__", repr(script))
    return hashlib.sha256(f"{module}:{qualname}".encode()).hexdigest()[:16]


def args_fingerprint(args: Any) -> str:
    try:
        payload = json.dumps(args, sort_keys=True, default=repr)
    except TypeError:
        payload = repr(args)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def agent_cache_key(
    prompt: str,
    *,
    schema: type[BaseModel] | None,
    tools: list[Any] | None,
    instructions: str,
    label: str | None,
    phase: str | None,
    model: str | None,
    isolation: str | None,
    agent_type: str | None,
    sees: list[str] | None = None,
) -> str:
    tool_ids = tuple(id(t) for t in (tools or ()))
    schema_name = schema.__name__ if schema is not None else ""
    parts = (
        prompt,
        schema_name,
        tool_ids,
        instructions,
        label or "",
        phase or "",
        model or "",
        isolation or "",
        agent_type or "",
        tuple(sees or ()),
    )
    payload = json.dumps(parts, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class AgentJournalEntry:
    sequence: int
    cache_key: str
    prompt: str
    result_kind: str  # "text" | "schema" | "none"
    result_text: str | None = None
    schema_name: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    def to_json(self) -> str:
        data = asdict(self)
        data["usage"] = asdict(self.usage)
        return json.dumps(data)

    @classmethod
    def from_json(cls, line: str) -> AgentJournalEntry:
        data = json.loads(line)
        usage = TokenUsage(**data.pop("usage"))
        return cls(usage=usage, **data)


@dataclass
class RunJournal:
    run_id: str
    script_fingerprint: str
    args_fingerprint: str
    entries: list[AgentJournalEntry] = field(default_factory=list)
    _live_from: int | None = field(default=None, repr=False)

    def lookup(self, sequence: int, cache_key: str) -> AgentJournalEntry | None:
        if self._live_from is not None and sequence >= self._live_from:
            return None
        if sequence >= len(self.entries):
            return None
        entry = self.entries[sequence]
        if entry.cache_key != cache_key:
            self._live_from = sequence
            return None
        return entry

    def append(self, entry: AgentJournalEntry) -> None:
        self.entries.append(entry)

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "script_fingerprint": self.script_fingerprint,
            "args_fingerprint": self.args_fingerprint,
        }
        (path.parent / "meta.json").write_text(json.dumps(meta, indent=2))
        with path.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(entry.to_json() + "\n")

    @classmethod
    def load(cls, run_id: str, journal_dir: Path | None = None) -> RunJournal:
        root = (journal_dir or default_journal_dir()) / run_id
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        entries: list[AgentJournalEntry] = []
        journal_path = root / "journal.jsonl"
        if journal_path.exists():
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(AgentJournalEntry.from_json(line))
        return cls(
            run_id=meta["run_id"],
            script_fingerprint=meta["script_fingerprint"],
            args_fingerprint=meta["args_fingerprint"],
            entries=entries,
        )

    @classmethod
    def create(
        cls,
        *,
        script: Any,
        args: Any,
        registry_name: str | None = None,
        resume_run_id: str | None = None,
        journal_dir: Path | None = None,
    ) -> tuple[RunJournal, Path]:
        root = journal_dir or default_journal_dir()
        fp_script = script_fingerprint(script, registry_name=registry_name)
        fp_args = args_fingerprint(args)
        if resume_run_id:
            journal = cls.load(resume_run_id, root)
            if (
                journal.script_fingerprint != fp_script
                or journal.args_fingerprint != fp_args
            ):
                journal._live_from = 0
            return journal, root / journal.run_id / "journal.jsonl"
        run_id = str(uuid.uuid4())
        journal = cls(
            run_id=run_id,
            script_fingerprint=fp_script,
            args_fingerprint=fp_args,
        )
        return journal, root / run_id / "journal.jsonl"


def serialize_agent_result(
    result: Any, schema: type[BaseModel] | None
) -> AgentJournalEntry:
    if result is None:
        return AgentJournalEntry(
            sequence=-1,
            cache_key="",
            prompt="",
            result_kind="none",
        )
    if schema is not None and isinstance(result, BaseModel):
        return AgentJournalEntry(
            sequence=-1,
            cache_key="",
            prompt="",
            result_kind="schema",
            result_text=result.model_dump_json(),
            schema_name=schema.__name__,
        )
    return AgentJournalEntry(
        sequence=-1,
        cache_key="",
        prompt="",
        result_kind="text",
        result_text=str(result),
    )


def deserialize_agent_result(
    entry: AgentJournalEntry, schema: type[BaseModel] | None
) -> Any:
    if entry.result_kind == "none":
        return None
    if entry.result_kind == "schema" and schema is not None:
        return schema.model_validate_json(entry.result_text or "{}")
    return entry.result_text
