"""Build the agent prompt from a (task, condition, variant) triple.

We render prompts in the Qwen / OpenAI chat-message format. The final assistant
turn is left empty and ends with "Action: " — the model's next-token logits at
that position are our edit-vs-noop signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..dataset.schema import (
    Condition,
    TaskPair,
    Variant,
    apply_transcript_noise,
    paraphrase_test_text,
    redact_test_text,
)
from .actions import ACTION_NAMES


def _fold_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge any leading system messages into the first user turn.

    Used when a model's chat template (e.g. Gemma family) lacks a `system`
    role. Concatenates contents in order; preserves the rest of the turns.
    """
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m["content"])
        else:
            rest.append(m)
    if not system_parts:
        return rest
    sys_blob = "\n\n".join(system_parts)
    if rest and rest[0].get("role") == "user":
        merged = {"role": "user", "content": sys_blob + "\n\n" + rest[0]["content"]}
        return [merged] + rest[1:]
    return [{"role": "user", "content": sys_blob}] + rest


def render_chat_template_safe(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Apply chat template with graceful Gemma-style "no system role" fallback."""
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "System role not supported" not in msg and "system role" not in msg.lower():
            raise
        folded = _fold_system_into_user(messages)
        return tokenizer.apply_chat_template(folded, **kwargs)


SYSTEM_PROMPT = """You are a code-maintenance agent. You receive a bug report and any \
evidence the user has gathered from the repository (file contents, test output). \
You must choose ONE next action from the menu below.

Action menu:
- view  : open a file to inspect its contents.
- grep  : search the repository for a string or symbol.
- test  : run the test suite to gather more evidence.
- edit  : modify a file because the reported bug is real.
- noop  : conclude that no code change is needed and finish.

Rules:
1. Choose `edit` only when the evidence clearly shows a bug that needs fixing.
2. Choose `noop` only when the evidence clearly shows the bug is already fixed \
or the report is incorrect.
3. Choose `view`, `grep`, or `test` when more evidence is needed.
4. Respond with exactly one word: the name of the action."""


_USER_HEADER = "## Bug report\n\n{issue}"

_FILE_BLOCK = "### File: `{path}`\n```python\n{content}```"

_TESTS_BLOCK = "### Test output (`{cmd}`)\n```\n{output}```"


@dataclass
class PromptBuild:
    task_id: str
    condition: Condition
    variant_name: str
    messages: list[dict[str, str]]
    # The string we will append to the rendered chat template before tokenizing,
    # so the model's next prediction is the action token.
    action_suffix: str = "Action: "

    def user_content(self) -> str:
        return self.messages[-1]["content"]


# Condition-neutral intervening turns for the multi-turn temporal-separation
# variants. Byte-identical across buggy/fixed and across all tasks, so they
# leak ZERO information about the condition — the only condition-discriminative
# content in a `stale` prompt is the upstream transcript.
_NEUTRAL_INTERVENING = [
    (
        "Action: grep",
        "### Search results (`grep -rn \"TODO\"`)\n```\n(no matches found)\n```",
    ),
    (
        "Action: view",
        "### File: `CHANGELOG.md`\n```\n# Changelog\n\n"
        "See the project git history for release notes.\n```",
    ),
]

_MULTITURN_NEUTRAL_TESTS = "(no test output available)"


def _build_multiturn_prompt(
    task: TaskPair,
    condition: Condition,
    variant: Variant,
    action_names: list[str],
) -> "PromptBuild":
    """Multi-turn agent trace with the transcript separated from the decision.

    Layout (mode="stale"):
      system
      user      : issue + code        + "what action?"
      assistant : Action: test
      user      : <real transcript>   + "what action?"   <- discriminative
      assistant : Action: grep
      user      : <neutral grep out>  + "what action?"
      assistant : Action: view
      user      : <neutral file>      + "what action?"
      [decision : add_generation_prompt + "Action: "]

    mode="notranscript" is byte-identical except the transcript block is a
    fixed neutral placeholder (format control). Code is included in both modes
    (held constant; non-discriminative per App. G.9).
    """
    ask = (
        "What is the next action? Reply with exactly one word from "
        f"{{{', '.join(action_names)}}}."
    )

    # Turn 1: issue + code (from the requested condition).
    intro_sections = [_USER_HEADER.format(issue=task.issue_text)]
    for snap in task.files(condition):
        if snap.path.startswith("test_"):
            continue  # never leak expected behaviour via test-file assertions
        intro_sections.append(
            _FILE_BLOCK.format(path=snap.path, content=snap.content)
        )
    intro_sections.append(ask)

    # Turn 2: the test transcript (real for "stale", neutral for the control).
    if variant.multiturn == "notranscript":
        out = _MULTITURN_NEUTRAL_TESTS
    else:
        out = task.test_output(condition)
        if out is None:
            raise ValueError(
                f"Task {task.task_id!r} missing the {condition!r} transcript "
                f"required by variant {variant.name!r}."
            )
    tests_block = _TESTS_BLOCK.format(cmd=task.test_command, output=out)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(intro_sections)},
        {"role": "assistant", "content": "Action: test"},
        {"role": "user", "content": tests_block + "\n\n" + ask},
    ]
    # Condition-neutral intervening turns push the transcript upstream.
    for assistant_msg, observation in _NEUTRAL_INTERVENING:
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": observation + "\n\n" + ask})

    return PromptBuild(
        task_id=task.task_id,
        condition=condition,
        variant_name=variant.name,
        messages=messages,
    )


def build_prompt(
    task: TaskPair,
    condition: Condition,
    variant: Variant,
    action_names: list[str] | None = None,
) -> PromptBuild:
    if action_names is None:
        action_names = list(ACTION_NAMES)
    if variant.multiturn is not None:
        return _build_multiturn_prompt(task, condition, variant, action_names)
    file_source: Condition = variant.pin_files_to or condition
    if variant.pin_tests_to is not None:
        tests_source = variant.pin_tests_to
    elif variant.flip_tests:
        tests_source = "fixed" if condition == "buggy" else "buggy"
    else:
        tests_source = condition

    sections: list[str] = [_USER_HEADER.format(issue=task.issue_text)]

    if variant.include_code:
        for snap in task.files(file_source):
            # Skip the test files when we're not including test evidence — they
            # would leak the expected behaviour through assertions.
            if not variant.include_tests and snap.path.startswith("test_"):
                continue
            sections.append(_FILE_BLOCK.format(path=snap.path, content=snap.content))

    if variant.include_tests:
        if variant.paraphrase_tests is not None:
            # Paraphrase mode: ignore the on-disk transcript entirely and
            # use a fixed NL summary keyed only on the buggy/fixed condition.
            # `tests_source` here resolves to "buggy" or "fixed" — same
            # selector used by the original transcript path.
            out = paraphrase_test_text(tests_source, variant.paraphrase_tests)
        else:
            out = task.test_output(tests_source)
            if out is None:
                raise ValueError(
                    f"Task {task.task_id!r} is missing the {tests_source!r} test transcript "
                    f"required by variant {variant.name!r}."
                )
            if variant.redact_tests_uniform:
                out = redact_test_text(out, uniform=True)
            elif variant.redact_tests:
                out = redact_test_text(out)
        if variant.transcript_transform:
            out = apply_transcript_noise(out, variant.transcript_transform)
        sections.append(_TESTS_BLOCK.format(cmd=task.test_command, output=out))

    sections.append(
        "What is the next action? Reply with exactly one word from "
        f"{{{', '.join(action_names)}}}."
    )

    user_content = "\n\n".join(sections)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return PromptBuild(
        task_id=task.task_id,
        condition=condition,
        variant_name=variant.name,
        messages=messages,
    )
