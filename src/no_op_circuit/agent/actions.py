"""Action vocabulary for the no-op-circuit agent.

Five canonical actions. Chosen as short, lowercase English words to maximize
the chance of being tokenized as a single BPE token by most modern code
tokenizers. We verify single-tokenness at model load time.

We deliberately use single short words rather than tool-call JSON because:
  (a) Mech interp analysis is much cleaner at a single action token.
  (b) Logit margin "edit minus noop" becomes a one-number summary of the
      model's edit-vs-abstain decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    description: str


ACTIONS: list[Action] = [
    Action("view", "Open a file in the repo to inspect its contents."),
    Action("grep", "Search the repo for a string or symbol."),
    Action("test", "Run the test suite and read the output."),
    Action("edit", "Modify a file to fix the bug."),
    Action("noop", "Conclude that no code change is needed; finish without editing."),
]

ACTION_NAMES: list[str] = [a.name for a in ACTIONS]
