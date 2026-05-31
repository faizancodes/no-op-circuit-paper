"""Action-order, abstract-label, and binary action-menu controls (Phase 2).

These build *position-balanced* prompt variants to test whether the 0%
explicit-abstention (``noop``) rate under the canonical fixed menu is an
artifact of ``noop`` always being listed last, rather than a property of the
abstention content itself.

STATUS: the prompt-construction code path is implemented and unit-tested
(``python -m no_op_circuit.agent.action_order`` runs a self-check). The model
evaluation is **UNRUN** in this revision (it needs GPU/Modal). No results from
this module are claimed in the manuscript; see paper Sec. 6.3.

Driver scaffold: ``scripts/run_action_order_control.py``.
"""

from __future__ import annotations

from ..dataset.schema import Condition, TaskPair, Variant
from .actions import ACTION_NAMES
from . import prompt as _prompt

# Exact menu descriptions from the canonical SYSTEM_PROMPT so a reordered menu
# changes ONLY the order, never the wording (keeps the control clean).
_MENU_DESC: dict[str, str] = {
    "view": "open a file to inspect its contents.",
    "grep": "search the repository for a string or symbol.",
    "test": "run the test suite to gather more evidence.",
    "edit": "modify a file because the reported bug is real.",
    "noop": "conclude that no code change is needed and finish.",
    # Single-token DeepSeek synonyms (see results/tokenization/): find=grep, done=noop.
    "find": "search the repository for a string or symbol.",
    "done": "conclude that no code change is needed and finish.",
}

_INTRO = (
    "You are a code-maintenance agent. You receive a bug report and any "
    "evidence the user has gathered from the repository (file contents, test "
    "output). You must choose ONE next action from the menu below."
)

_RULES = (
    "Rules:\n"
    "1. Choose `edit` only when the evidence clearly shows a bug that needs fixing.\n"
    "2. Choose `noop` only when the evidence clearly shows the bug is already fixed "
    "or the report is incorrect.\n"
    "3. Choose `view`, `grep`, or `test` when more evidence is needed.\n"
    "4. Respond with exactly one word: the name of the action."
)


def cyclic_action_orders(action_names: list[str] | None = None) -> list[list[str]]:
    """``n`` cyclic rotations of the action list.

    With ``n`` actions this yields ``n`` orderings in which every action --
    crucially ``noop`` -- occupies each of the ``n`` list positions exactly
    once, giving an exactly position-balanced design.
    """
    names = list(action_names or ACTION_NAMES)
    n = len(names)
    return [[names[(i + r) % n] for i in range(n)] for r in range(n)]


def system_prompt_for_order(order: list[str]) -> str:
    """Canonical SYSTEM_PROMPT with the action menu rendered in ``order``."""
    menu = "\n".join(f"- {name}  : {_MENU_DESC[name]}" for name in order)
    return f"{_INTRO}\n\nAction menu:\n{menu}\n\n{_RULES}"


def build_custom_order_prompts(task, condition, variant, action_words, abstain_word):
    """Position-balanced action-order prompts with a CUSTOM single-token vocab.

    Used for the DeepSeek single-token rerun ({view, find, test, edit, done}).
    Returns ``(order_id, order, abstain_position, PromptBuild)``.
    """
    out = []
    for k, order in enumerate(cyclic_action_orders(action_words)):
        pb = _prompt.build_prompt(task, condition, variant, action_names=order)
        pb.messages[0] = {"role": "system", "content": system_prompt_for_order(order)}
        pb.variant_name = f"{variant.name}__custom{k}"
        out.append((k, order, order.index(abstain_word), pb))
    return out


def build_action_order_prompts(
    task: TaskPair,
    condition: Condition,
    variant: Variant,
    orders: list[list[str]] | None = None,
) -> list[tuple[int, list[str], int, "_prompt.PromptBuild"]]:
    """One prompt per ordering, with BOTH the menu and the inline action list
    permuted. Returns ``(order_id, order, noop_position, PromptBuild)`` tuples.
    """
    orders = orders or cyclic_action_orders()
    out = []
    for k, order in enumerate(orders):
        pb = _prompt.build_prompt(task, condition, variant, action_names=order)
        pb.messages[0] = {"role": "system", "content": system_prompt_for_order(order)}
        pb.variant_name = f"{variant.name}__order{k}"
        out.append((k, order, order.index("noop"), pb))
    return out


# --- Experiment C: binary edit/noop menu --------------------------------------

def binary_orders() -> list[list[str]]:
    """The two single-token binary menus: ``[edit, noop]`` and ``[noop, edit]``."""
    return [["edit", "noop"], ["noop", "edit"]]


# --- Experiment B: abstract labels A-E (scaffold) -----------------------------
# Position-balanced abstract-label mappings (A..E) where the abstention action
# maps to each label-position once. Single-tokenness of A..E after "Action: "
# must be verified per-tokenizer at run time (see driver). Implemented as a
# generator here; full prompt rendering with an in-prompt mapping table and the
# decode-back step live in the (UNRUN) driver scaffold.

_ABSTRACT_LABELS = ["A", "B", "C", "D", "E"]


def abstract_label_mappings(action_names: list[str] | None = None):
    """Yield ``(label_to_action, noop_label_index)`` for each cyclic order."""
    for order in cyclic_action_orders(action_names):
        mapping = dict(zip(_ABSTRACT_LABELS, order))
        noop_idx = order.index("noop")
        yield mapping, noop_idx


def to_job(pb: "_prompt.PromptBuild", action_names: list[str], **extra) -> dict:
    """Convert a PromptBuild into a cache_activations-style job dict."""
    job = {
        "task_id": pb.task_id,
        "condition": pb.condition,
        "variant_name": pb.variant_name,
        "messages": pb.messages,
        "action_suffix": pb.action_suffix,
        "action_names": list(action_names),
    }
    job.update(extra)
    return job


def abstract_label_system_prompt(mapping: dict[str, str]) -> str:
    table = "\n".join(f"{lab} = {mapping[lab]}" for lab in _ABSTRACT_LABELS)
    return (
        f"{_INTRO}\n\nEach letter maps to one action:\n{table}\n\n"
        "Rules:\n"
        "1. Choose the letter for `edit` only when the evidence clearly shows a bug that needs fixing.\n"
        "2. Choose the letter for `noop` only when the evidence clearly shows the bug is already fixed "
        "or the report is incorrect.\n"
        "3. Choose the letter for `view`, `grep`, or `test` when more evidence is needed.\n"
        "4. Respond with exactly one letter: A, B, C, D, or E."
    )


def build_abstract_label_prompts(task, condition, variant):
    """One prompt per cyclic label->action mapping. action_names to score are A-E.
    Returns ``(order_id, mapping, noop_label_idx, PromptBuild)``.
    """
    out = []
    for k, (mapping, noop_idx) in enumerate(abstract_label_mappings()):
        pb = _prompt.build_prompt(task, condition, variant, action_names=list(_ABSTRACT_LABELS))
        pb.messages[0] = {"role": "system", "content": abstract_label_system_prompt(mapping)}
        pb.variant_name = f"{variant.name}__label{k}"
        out.append((k, mapping, noop_idx, pb))
    return out


_BINARY_DESC = {
    "edit": "modify a file because the reported bug is real.",
    "noop": "conclude that no code change is needed and finish.",
}


def binary_system_prompt(order: list[str]) -> str:
    menu = "\n".join(f"- {name}  : {_BINARY_DESC[name]}" for name in order)
    return (
        f"{_INTRO}\n\nAction menu:\n{menu}\n\n"
        "Rules:\n"
        "1. Choose `edit` only when the evidence clearly shows a bug that needs fixing.\n"
        "2. Choose `noop` only when the evidence clearly shows the bug is already fixed "
        "or the report is incorrect.\n"
        "3. Respond with exactly one word: edit or noop."
    )


def build_binary_prompts(task, condition, variant):
    """One prompt per binary menu order ([edit,noop] and [noop,edit])."""
    out = []
    for k, order in enumerate(binary_orders()):
        pb = _prompt.build_prompt(task, condition, variant, action_names=order)
        pb.messages[0] = {"role": "system", "content": binary_system_prompt(order)}
        pb.variant_name = f"{variant.name}__bin{k}"
        out.append((k, order, order.index("noop"), pb))
    return out


def letter_only_system_prompt() -> str:
    """Content-free letter-choice prompt: A-E carry NO action meaning.

    Used to test whether the abstract-label `B`-dominance is an intrinsic
    surface-form prior (the model defaults to one letter regardless of meaning)
    rather than a property of the action-mapping format.
    """
    return (
        "You will be shown a bug report and any evidence gathered from the "
        "repository. Then choose one option, labeled A, B, C, D, or E. The "
        "letters carry no inherent meaning.\n\n"
        "Respond with exactly one letter: A, B, C, D, or E."
    )


def build_letter_only_prompts(task, condition, variant):
    """One content-free prompt per (task, condition): task evidence + bare A-E."""
    pb = _prompt.build_prompt(task, condition, variant, action_names=list(_ABSTRACT_LABELS))
    pb.messages[0] = {"role": "system", "content": letter_only_system_prompt()}
    pb.variant_name = f"{variant.name}__letteronly"
    return [(0, None, None, pb)]


if __name__ == "__main__":  # lightweight self-check (no model / no GPU)
    orders = cyclic_action_orders()
    n = len(ACTION_NAMES)
    # Position balance: noop appears in every slot exactly once.
    noop_positions = sorted(o.index("noop") for o in orders)
    assert noop_positions == list(range(n)), noop_positions
    # Every action is position-balanced, not just noop.
    for pos in range(n):
        col = sorted(o[pos] for o in orders)
        assert col == sorted(ACTION_NAMES), (pos, col)
    assert binary_orders() == [["edit", "noop"], ["noop", "edit"]]
    assert [m["E"] for m, _ in [(dict(zip(_ABSTRACT_LABELS, o)), 0) for o in orders]]
    print(f"OK: {n} position-balanced orders; noop slots {noop_positions}")
    print("sample menu (order 1):")
    print(system_prompt_for_order(orders[1]))
