"""Task-pair schema.

Each task ships a buggy and a fixed copy of the same code, plus a single issue
text and per-condition test transcripts. Variants control which pieces of
evidence the agent gets to see in its prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Condition = Literal["buggy", "fixed"]
TestSource = Literal["buggy", "fixed", "fixed_flaky"]


@dataclass(frozen=True)
class Variant:
    name: str
    include_code: bool
    include_tests: bool
    # When set, the variant overrides the file/test source for the leg of a
    # pair (e.g. stale_misleading always shows the fixed code, even to the
    # "buggy" condition — used for one-sided probes).
    pin_files_to: Condition | None = None
    pin_tests_to: TestSource | None = None
    # When True, swap the test transcript to the OPPOSITE buggy/fixed condition
    # while keeping the code from the requested condition. Used for the
    # contradictory-transcript control (App. ?). Mutually exclusive with
    # pin_tests_to: if both are set, pin_tests_to wins (build_prompt enforces).
    flip_tests: bool = False
    # When True, post-process the test transcript through `redact_test_text`
    # to strip overt pass/fail tokens (FAILED, passed, AssertionError, ...)
    # while preserving structural shape. Used for the lexical-redaction
    # control. Applied AFTER pin_tests_to / flip_tests pick the source.
    redact_tests: bool = False
    # As above, but uses the case-uniform OUTCOME replacement to avoid
    # the redaction artefacts (case/shape) of the standard variant.
    redact_tests_uniform: bool = False
    # Replace the test transcript entirely with a fixed natural-language
    # paraphrase keyed only on the buggy/fixed condition. Used for the
    # paraphrased-transcript control (App. G.15). Two styles:
    #   "minimal"   — one-sentence absent/present contrast (~49 chars).
    #   "realistic" — 2-3 sentence prose mimicking a code-reviewer summary.
    # The structural prompt wrapper (code block + filename) is preserved.
    paraphrase_tests: str | None = None  # None | "minimal" | "realistic"
    # Temporally-separated / multi-turn transcript variants (App. G.15). When
    # set, build_prompt emits a multi-turn agent trace where the test
    # transcript (if any) appears at an EARLY turn and is separated from the
    # final decision point by condition-neutral intervening turns (grep/view
    # with byte-identical content across buggy/fixed). Tests whether the
    # pass/fail direction carries the verdict forward to a decision point where
    # the transcript is positionally distant.
    #   "stale"        — real transcript shown upstream, then neutral turns.
    #   "notranscript" — identical scaffold but the transcript block is a
    #                    neutral placeholder (format control).
    multiturn: str | None = None  # None | "stale" | "notranscript"
    # Phase 4 transcript-robustness transforms applied to the transcript text
    # AFTER source selection/redaction: flaky | many_passing | truncated |
    # summary_only. Same perturbation applied to both conditions, so the
    # buggy/fixed signal must survive it.
    transcript_transform: str | None = None
    description: str = ""


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant(
            name="issue_only",
            include_code=False,
            include_tests=False,
            description="Issue text only — no code, no tests. Control for model prior.",
        ),
        Variant(
            name="code",
            include_code=True,
            include_tests=False,
            description="Issue + code. No test evidence.",
        ),
        Variant(
            name="code_tests",
            include_code=True,
            include_tests=True,
            description="Issue + code + test output. The main causal contrast.",
        ),
        # --- Phase 4 transcript-robustness variants (noisy/mixed transcripts) ---
        Variant(name="code_tests_noisy_flaky", include_code=True, include_tests=True,
                transcript_transform="flaky",
                description="code_tests + one unrelated FAILED line appended to both conditions."),
        Variant(name="code_tests_many_passing", include_code=True, include_tests=True,
                transcript_transform="many_passing",
                description="code_tests + many unrelated PASSED lines prepended to both conditions."),
        Variant(name="code_tests_truncated", include_code=True, include_tests=True,
                transcript_transform="truncated",
                description="code_tests with the transcript truncated to its first lines."),
        Variant(name="code_tests_summary_only", include_code=True, include_tests=True,
                transcript_transform="summary_only",
                description="code_tests with only the final pytest summary line kept."),
        Variant(
            name="stale_misleading",
            include_code=True,
            include_tests=True,
            pin_files_to="fixed",
            pin_tests_to="fixed",
            description=(
                "Issue claims a bug but code is already fixed and all tests pass. "
                "Probes whether the model overrides issue text when evidence disagrees."
            ),
        ),
        Variant(
            name="stale_flaky",
            include_code=True,
            include_tests=True,
            pin_files_to="fixed",
            # tests_output is the fixed-with-one-flaky transcript, supplied
            # separately by the task author when present.
            pin_tests_to="fixed_flaky",
            description=(
                "Like stale_misleading, but a single unrelated test is failing "
                "(flaky). Probes false-positive edits triggered by noise."
            ),
        ),
        Variant(
            name="code_tests_swapped",
            include_code=True,
            include_tests=True,
            flip_tests=True,
            description=(
                "Contradictory-transcript control. The buggy condition shows "
                "the BUGGY code + the FIXED (passing) transcript; the fixed "
                "condition shows the FIXED code + the BUGGY (failing) "
                "transcript. Distinguishes a code-driven monitor from a "
                "pass/fail-transcript-text detector."
            ),
        ),
        Variant(
            name="code_tests_lex_redacted",
            include_code=True,
            include_tests=True,
            redact_tests=True,
            description=(
                "Lexical-redaction control: same as code_tests but the pytest "
                "transcript has its overt pass/fail tokens redacted "
                "(FAILED, passed, failed, AssertionError, FAILURES, OK, "
                "Traceback) while preserving structural shape (line counts, "
                "test-name paths, summary-line skeleton). Tests whether the "
                "monitor depends on literal pass/fail tokens or on a more "
                "abstract pass/fail representation."
            ),
        ),
        Variant(
            name="code_tests_lex_redacted_uniform",
            include_code=True,
            include_tests=True,
            redact_tests_uniform=True,
            description=(
                "Stricter lexical-redaction control. All pass/fail vocabulary "
                "(FAILED, passed, failed, AssertionError, FAILURES, OK, "
                "Traceback) maps to a single case-uniform token `OUTCOME`. "
                "Eliminates the redaction artefacts (case/shape differences "
                "between REDACT vs redact) that the standard variant left in "
                "place. Buggy and fixed transcripts still differ in structural "
                "shape (failure-section length, summary-line count) but no "
                "longer in lexical content."
            ),
        ),
        Variant(
            name="code_tests_paraphrased_minimal",
            include_code=True,
            include_tests=True,
            paraphrase_tests="minimal",
            description=(
                "Paraphrased-transcript control (minimal). Replace the entire "
                "transcript with a single short sentence keyed only on the "
                "buggy/fixed condition (`The expected behavior was "
                "absent/present in the output.`). Strips BOTH literal pytest "
                "tokens AND structural cues (line counts, FAILURES sections, "
                "summary lines). Tests whether the residual reads a semantic "
                "pass/fail signal or is bound to the literal pytest format."
            ),
        ),
        Variant(
            name="code_tests_paraphrased_realistic",
            include_code=True,
            include_tests=True,
            paraphrase_tests="realistic",
            description=(
                "Paraphrased-transcript control (realistic). Same as the "
                "minimal variant but the paraphrase is a 2-3 sentence prose "
                "summary that mimics what a developer or code reviewer might "
                "write in a PR comment. Buggy and fixed prose are parallel in "
                "structure and near-equal in length, so the only discriminative "
                "signal is the semantic polarity."
            ),
        ),
        Variant(
            name="code_tests_stale_multiturn",
            include_code=True,
            include_tests=True,
            multiturn="stale",
            description=(
                "Temporally-separated transcript (App. G.15). A multi-turn "
                "agent trace: turn 1 shows issue + code, turn 2 runs tests and "
                "shows the real pytest transcript, then two condition-neutral "
                "intervening turns (grep + view, byte-identical across "
                "buggy/fixed) push the transcript upstream before the final "
                "decision point. The decision-point's local context contains no "
                "pass/fail tokens; only the residual carries the verdict "
                "forward. Tests whether the L24/pos-1 direction reads pass/fail "
                "when the transcript is positionally distant from the decision."
            ),
        ),
        Variant(
            name="code_multiturn_notranscript",
            include_code=True,
            include_tests=True,
            multiturn="notranscript",
            description=(
                "Format control for code_tests_stale_multiturn. Identical "
                "multi-turn scaffold (same turns, same neutral intervening "
                "content, same code) but the transcript block is replaced by a "
                "neutral placeholder identical across buggy/fixed. If the "
                "residual still discriminates here, the multi-turn FORMAT — not "
                "the carried-forward transcript — is the driver. Expected: "
                "chance (cf. App. G.9 single-turn no-transcript)."
            ),
        ),
    ]
}


_REDACTION_RULES = [
    # Order matters: longest matches first to avoid eating substrings.
    ("AssertionError", "RedactedErr___"),
    ("Traceback",      "Trace________"),
    ("FAILURES",       "REDACTED"),
    ("FAILED",         "REDACT"),
    ("passed",         "redact"),
    ("failed",         "redact"),
    (" OK\n",          " ___\n"),
]


# A stricter redaction: map ALL pass/fail-related vocabulary to a single
# case-uniform neutral token. This eliminates the redaction artefacts of
# the standard rules (REDACT vs redact differing in case/length/shape)
# that a reviewer correctly flagged as still class-leaking.
_UNIFORM_REDACTION_RULES = [
    ("AssertionError", "OUTCOME"),
    ("Traceback",      "OUTCOME"),
    ("FAILURES",       "OUTCOME"),
    ("FAILED",         "OUTCOME"),
    ("passed",         "OUTCOME"),
    ("failed",         "OUTCOME"),
    ("OK",             "OUTCOME"),
]


def redact_test_text(text: str, *, uniform: bool = False) -> str:
    """Strip overt pass/fail tokens while preserving structural shape.

    Used by the `code_tests_lex_redacted` (uniform=False) and
    `code_tests_lex_redacted_uniform` (uniform=True) variants.

    With uniform=False, replacements are chosen so each token's character
    count is preserved (within 1–3), so prompt lengths don't shift
    materially. Buggy and fixed transcripts still differ slightly in
    capitalisation and replacement shape, which a reviewer flagged as
    a potential confound.

    With uniform=True, ALL pass/fail/error vocabulary collapses to a
    single token `OUTCOME` (case-uniform), eliminating those artefacts.
    Buggy and fixed transcripts still differ in line count and
    structural shape (failure section vs single summary line) but no
    longer in lexical pass/fail content.
    """
    rules = _UNIFORM_REDACTION_RULES if uniform else _REDACTION_RULES
    out = text
    for needle, repl in rules:
        out = out.replace(needle, repl)
    return out


# Natural-language paraphrases of the test transcript. The point is to
# strip BOTH the literal pytest tokens (FAILED / passed / AssertionError /
# Traceback / OK) AND the structural pattern (FAILURES section, summary
# line, line-count differences) while preserving the semantic pass/fail
# signal. Two styles parameterise how rich the NL is:
#
#   "minimal"   — one short sentence per condition with absent/present
#                 contrast. The cleanest scientific test of "is the
#                 residual reading semantic pass/fail at all?".
#   "realistic" — 2-3 sentence prose mimicking what a developer or code
#                 reviewer might write in a PR comment.
#
# Both variants hold prompt-format structure constant across buggy and
# fixed (same wrapper, parallel sentence structure, near-equal length),
# so any residual discriminative signal must come from the *semantic
# content* of the paraphrase.
_PARAPHRASE_MINIMAL = {
    "buggy": "The expected behavior was absent in the output.",
    "fixed": "The expected behavior was present in the output.",
}

_PARAPHRASE_REALISTIC = {
    "buggy": (
        "After running the project's test suite on this code, the assertions "
        "did not match the behavior described in the issue. Several test "
        "cases produced output that differed from what the documentation "
        "indicated should occur, suggesting the implementation does not yet "
        "satisfy the specification."
    ),
    "fixed": (
        "After running the project's test suite on this code, the assertions "
        "matched the behavior described in the issue. All test cases produced "
        "output that aligned with what the documentation indicated should "
        "occur, suggesting the implementation satisfies the specification."
    ),
}


def paraphrase_test_text(condition: str, style: str) -> str:
    """Return the fixed NL paraphrase for the given condition and style.

    The paraphrase is independent of the original transcript content —
    only the buggy/fixed *polarity* matters. This isolates "does the
    residual read semantic pass/fail" from "does the residual read
    literal pytest format".
    """
    if style == "minimal":
        table = _PARAPHRASE_MINIMAL
    elif style == "realistic":
        table = _PARAPHRASE_REALISTIC
    else:
        raise ValueError(f"unknown paraphrase style: {style!r}")
    if condition not in table:
        raise ValueError(f"paraphrase undefined for condition {condition!r}")
    return table[condition] + "\n"


# Phase 4 transcript-robustness transforms. Each is a fixed perturbation applied
# identically to buggy and fixed transcripts, so the pass/fail signal must
# survive it. Designed to break naive transcript parsers (distractor failures /
# passes) or to remove detail (truncation / summary-only).
_FLAKY_TAIL = (
    "=================================== FAILURES ===================================\n"
    "______________________________ test_network_retry _____________________________\n"
    "tests/test_integration.py::test_network_retry FAILED\n"
    "E   ConnectionError: transient timeout (unrelated to this change)\n"
)
_MANY_PASS_HEAD = "".join(
    f"tests/test_suite.py::test_case_{i:02d} PASSED\n" for i in range(1, 21)
)


def apply_transcript_noise(text: str, kind: str) -> str:
    """Apply a Phase-4 robustness transform to a transcript string."""
    if kind == "flaky":  # append one unrelated failing test (distractor failure)
        return text.rstrip("\n") + "\n" + _FLAKY_TAIL
    if kind == "many_passing":  # prepend many passing tests (distractor passes)
        return _MANY_PASS_HEAD + text
    if kind == "truncated":  # keep only the first 6 lines
        return "\n".join(text.splitlines()[:6]) + "\n"
    if kind == "summary_only":  # keep only the final pytest summary line
        lines = [ln for ln in text.splitlines() if ln.strip()]
        summary = next(
            (ln for ln in reversed(lines)
             if any(w in ln.lower() for w in ("passed", "failed", "error"))),
            lines[-1] if lines else "",
        )
        return summary + "\n"
    raise ValueError(f"unknown transcript transform: {kind!r}")


@dataclass
class FileSnapshot:
    path: str
    content: str


@dataclass
class TaskPair:
    task_id: str
    description: str
    issue_text: str
    primary_file: str
    test_command: str
    buggy_files: list[FileSnapshot] = field(default_factory=list)
    fixed_files: list[FileSnapshot] = field(default_factory=list)
    buggy_test_output: str = ""
    fixed_test_output: str = ""
    # Optional: a flaky-test transcript using the fixed code; populated only
    # when the task author has authored one. Used by the `stale_flaky` variant.
    fixed_flaky_test_output: str | None = None

    def files(self, condition: Condition) -> list[FileSnapshot]:
        return self.buggy_files if condition == "buggy" else self.fixed_files

    def test_output(self, source: TestSource) -> str | None:
        if source == "buggy":
            return self.buggy_test_output
        if source == "fixed":
            return self.fixed_test_output
        if source == "fixed_flaky":
            return self.fixed_flaky_test_output
        raise ValueError(f"unknown test-output source: {source!r}")
