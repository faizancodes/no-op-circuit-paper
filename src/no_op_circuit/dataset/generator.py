"""LLM-driven paired-task generator.

Calls OpenRouter with a structured JSON prompt for a given archetype and
returns a TaskCandidate dict. Validation happens in a separate module so the
generator can stay pure / cheap to test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import DATA_DIR
from ..llm import chat, parse_json


_ARCHETYPES_PATH = DATA_DIR / "archetypes.yaml"


@dataclass(frozen=True)
class Archetype:
    id: str
    description: str
    example_signature: str
    domains: list[str]


def load_archetypes(path: Path = _ARCHETYPES_PATH) -> list[Archetype]:
    blob = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Archetype(
            id=a["id"],
            description=a["description"].strip(),
            example_signature=a["example_signature"],
            domains=list(a.get("domains", [])),
        )
        for a in blob["archetypes"]
    ]


# Hand-curated worked example — anchored in our existing parser_empty_input
# task. Kept inline so the prompt is reproducible.
_FEW_SHOT_EXAMPLE = {
    "task_id": "parser_empty_input",
    "description": "parse() should return [] on empty string, but raises ValueError.",
    "issue_text": (
        "`parser.parse(\"\")` raises `ValueError(\"empty input not allowed\")`. "
        "According to the docstring it should return an empty list for empty "
        "input. Please fix so the documented behaviour holds."
    ),
    "primary_file": "parser.py",
    "test_command": "pytest -q test_parser.py",
    "buggy_files": [
        {
            "path": "parser.py",
            "content": (
                "def parse(text: str) -> list[str]:\n"
                "    \"\"\"Split a comma-separated string into trimmed tokens.\n"
                "\n"
                "    An empty input should return an empty list.\n"
                "    \"\"\"\n"
                "    if not text:\n"
                "        raise ValueError(\"empty input not allowed\")\n"
                "    return [t.strip() for t in text.split(\",\")]\n"
            ),
        },
    ],
    "fixed_files": [
        {
            "path": "parser.py",
            "content": (
                "def parse(text: str) -> list[str]:\n"
                "    \"\"\"Split a comma-separated string into trimmed tokens.\n"
                "\n"
                "    An empty input should return an empty list.\n"
                "    \"\"\"\n"
                "    if not text:\n"
                "        return []\n"
                "    return [t.strip() for t in text.split(\",\")]\n"
            ),
        },
    ],
    "test_file": {
        "path": "test_parser.py",
        "content": (
            "from parser import parse\n"
            "\n"
            "def test_empty_input_returns_empty_list():\n"
            "    assert parse(\"\") == []\n"
            "\n"
            "def test_basic_split():\n"
            "    assert parse(\"a,b,c\") == [\"a\", \"b\", \"c\"]\n"
            "\n"
            "def test_whitespace_trimmed():\n"
            "    assert parse(\"a , b ,c\") == [\"a\", \"b\", \"c\"]\n"
        ),
    },
}


_SYSTEM_PROMPT = """You are generating paired buggy/fixed Python tasks for \
mechanistic interpretability research on coding agents. Every task must be \
short, self-contained, and executable, and produce GROUND-TRUTH labels: the \
buggy version must fail at least one test, and the fixed version must pass \
every test.

Output STRICT JSON only — no prose, no markdown fences."""


def _build_user_prompt(
    arch: Archetype,
    *,
    instance_id: str | None = None,
    seen_function_names: list[str] | None = None,
    domain_hint: str | None = None,
) -> str:
    schema = {
        "task_id": "snake_case_descriptive_id",
        "description": "one short sentence describing the bug",
        "issue_text": "the bug report a user would file (symptom only, no fix hint)",
        "primary_file": "name of the source file (e.g. 'parser.py')",
        "test_command": "pytest -q <test_file_name>",
        "buggy_files": [{"path": "x.py", "content": "buggy source"}],
        "fixed_files": [{"path": "x.py", "content": "fixed source"}],
        "test_file": {"path": "test_x.py", "content": "pytest file"},
    }
    example_json = json.dumps(_FEW_SHOT_EXAMPLE, indent=2)
    schema_json = json.dumps(schema, indent=2)

    extra_lines: list[str] = []
    if instance_id:
        extra_lines.append(f"Instance hint (use as a creative seed): {instance_id}")
    if domain_hint:
        extra_lines.append(
            f"Target domain for THIS task: **{domain_hint}**. Pick a function "
            f"that fits this domain (not the example one)."
        )
    if seen_function_names:
        joined = ", ".join(sorted(set(seen_function_names))[-30:])
        extra_lines.append(
            "Already-used function names in this archetype — DO NOT propose "
            f"another task whose primary function is named like any of these "
            f"(or a trivial rename): {joined}."
        )
    extra = "\n".join(extra_lines)
    return f"""Generate ONE paired buggy/fixed Python task in the **{arch.id}** archetype.

Archetype description:
{arch.description}

Example signature (just for shape — DO NOT reuse this exact function):
  {arch.example_signature}
Possible domains: {", ".join(arch.domains) if arch.domains else "any"}.
{extra}

Hard constraints:
1. Pure Python 3.10+ stdlib only. NO non-stdlib imports.
2. Exactly ONE source file and ONE test file. Source file < 60 lines.
3. Test file must contain 3–5 `def test_*` functions using bare asserts.
4. The buggy version must fail AT LEAST ONE test (preferably exactly one).
5. The fixed version must pass EVERY test.
6. The diff between buggy and fixed should be MINIMAL — ideally 1–3 lines.
7. The issue text reads like a real bug report: describes the symptom, does
   NOT name the buggy line, the fix, or the implementation detail.
8. `test_command` must be `pytest -q <test_file_name>`.
9. Do not pick `parser_empty_input` — that's the worked example below.
10. Choose a fresh DOMAIN and a fresh FUNCTION NAME from any previous task in
    this archetype. Diversity across tasks is critical.

Worked example (a complete valid task):
```json
{example_json}
```

Output schema (return JSON matching this shape exactly):
```json
{schema_json}
```

Return ONLY the JSON object — no prose, no markdown fences."""


def generate_candidate(
    arch: Archetype,
    *,
    instance_id: str | None = None,
    model: str | None = None,
    temperature: float = 1.0,
    seen_function_names: list[str] | None = None,
    domain_hint: str | None = None,
) -> dict[str, Any]:
    """Single LLM call. Returns the parsed JSON candidate (NOT yet validated)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(
            arch,
            instance_id=instance_id,
            seen_function_names=seen_function_names,
            domain_hint=domain_hint,
        )},
    ]
    kwargs: dict[str, Any] = dict(
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    if model is not None:
        kwargs["model"] = model
    result = chat(**kwargs)
    candidate = parse_json(result.text)
    # Attach the snapshot the provider actually routed to and the request id,
    # so the generation log can later attribute outputs to a pinned model
    # (avoids the C6 reproducibility hole where the OpenRouter alias resolution
    # was not recorded for the original 49-task substrate run).
    if result.resolved_model is not None:
        candidate["_resolved_model"] = result.resolved_model
    if result.response_id is not None:
        candidate["_response_id"] = result.response_id
    return candidate
