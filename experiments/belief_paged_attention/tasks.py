# SPDX-License-Identifier: Apache-2.0
"""Synthetic long-context tasks for the belief-gated experiment.

Each task is built from **labeled segments** so the harness can map the critical
evidence to exact token positions (hence to logical KV pages) for the oracle and
keyword policies — without leaking any future information into the belief policy.

A task exposes:
  * ``segments``: ordered (text, is_critical, keywords) tuples. Concatenated they
    form the prompt. ``is_critical`` marks the evidence pages the answer depends
    on (used by ``oracle_belief_pages`` — prefix-derived, no future info).
  * ``answer``: the teacher-forced continuation (what the full-cache model should
    produce). Tasks are accepted only if the full-cache model actually solves
    them (checked in the harness), per the spec.
  * ``keywords`` / ``entity_vocab``: surface signals for the keyword baseline and
    the belief inferencer.

The six families stress different failure modes (needle recall, entity-state,
incremental constraints, belief revision/contradiction, delayed disambiguation,
distractor-heavy). The contradiction and delayed-disambiguation families are the
falsifier-#7 stressors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

Segment = tuple[str, bool, tuple[str, ...]]  # (text, is_critical, keywords)

_FILLER = [
    "The weather today is mild with a gentle breeze over the quiet hills.",
    "Gardeners often discuss the merits of compost and seasonal rotation.",
    "A distant train sounded its horn as the afternoon light faded slowly.",
    "Many people enjoy a warm cup of tea while reading by the window.",
    "The museum exhibit featured pottery and woven baskets from the coast.",
    "Clouds drifted lazily while children played in the open green park.",
    "Old maps show trade routes that crossed the wide and dusty plains.",
    "The bakery on the corner sells bread that smells of rosemary and salt.",
]


@dataclass
class TaskInstance:
    name: str
    task_type: str
    segments: list[Segment]
    answer: str
    keywords: tuple[str, ...]
    entity_vocab: frozenset[str]
    meta: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return "".join(s[0] for s in self.segments)


def _filler(rng: random.Random, n: int) -> list[Segment]:
    return [(" " + rng.choice(_FILLER), False, ()) for _ in range(n)]


def _interleave(rng: random.Random, criticals: list[Segment], n_filler: int) -> list[Segment]:
    """Scatter critical segments among filler so evidence is non-local."""
    fill = _filler(rng, n_filler)
    # place each critical at a random slot, keeping critical order
    slots = sorted(rng.sample(range(n_filler + len(criticals)),
                              len(criticals)))
    out: list[Segment] = []
    ci = 0
    fi = 0
    for pos in range(n_filler + len(criticals)):
        if ci < len(slots) and pos == slots[ci]:
            out.append(criticals[ci]); ci += 1
        else:
            out.append(fill[fi]); fi += 1
    return out


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------


def needle_recall(rng: random.Random, n_filler: int) -> TaskInstance:
    code = rng.randint(1000, 9999)
    crit: list[Segment] = [(f" The secret access code is {code}.", True, ("code", "secret"))]
    segs = [(" Read the passage and remember the code.", False, ())]
    segs += _interleave(rng, crit, n_filler)
    segs += [(" The secret access code is", False, ())]
    return TaskInstance("needle_recall", "needle", segs, f" {code}",
                        ("code", "secret", "access"), frozenset({"code", "secret"}))


def entity_state(rng: random.Random, n_filler: int) -> TaskInstance:
    rooms = ["kitchen", "garden", "library", "attic", "cellar"]
    a, b = rng.sample(rooms, 2)
    crit = [
        (f" Marcus is in the {a}.", True, ("Marcus",)),
        (f" Marcus moves to the {b}.", True, ("Marcus",)),
    ]
    segs = [(" Track where Marcus is.", False, ())]
    segs += _interleave(rng, crit, n_filler)
    segs += [(" Right now Marcus is in the", False, ())]
    return TaskInstance("entity_state", "entity", segs, f" {b}",
                        ("Marcus",), frozenset({"marcus"}), {"final": b})


def incremental_constraint(rng: random.Random, n_filler: int) -> TaskInstance:
    color = rng.choice(["blue", "green", "amber", "violet"])
    crit = [
        (" Constraint: the box must always be painted a single color.", True,
         ("constraint", "must")),
        (f" The chosen color is {color}.", True, ("color",)),
    ]
    segs = [(" Follow the painting constraints.", False, ())]
    segs += _interleave(rng, crit, n_filler)
    segs += [(" Following the constraint, the box color is", False, ())]
    return TaskInstance("incremental_constraint", "constraint", segs, f" {color}",
                        ("constraint", "must", "color"),
                        frozenset({"constraint", "must", "color"}))


def belief_revision(rng: random.Random, n_filler: int) -> TaskInstance:
    a, b = rng.sample(["Anna", "Diego", "Priya", "Omar"], 2)
    crit = [
        (f" The team leader is {a}.", True, ("leader",)),
        (f" Correction: actually the team leader is now {b}.", True,
         ("leader", "correction", "actually")),
    ]
    segs = [(" Note who currently leads the team.", False, ())]
    segs += _interleave(rng, crit, n_filler)
    segs += [(" The current team leader is", False, ())]
    return TaskInstance("belief_revision", "revision", segs, f" {b}",
                        ("leader", "correction", "actually"),
                        frozenset({"leader", "correction", "actually", a.lower(), b.lower()}),
                        {"old": a, "new": b})


def delayed_disambiguation(rng: random.Random, n_filler: int) -> TaskInstance:
    # Ambiguous pronoun resolved only later.
    val = rng.choice(["the red key", "the brass key", "the iron key"])
    crit = [
        (" Someone left a key on the table; it is unclear which one.", True,
         ("key",)),
        (f" Later it becomes clear the key is {val}.", True, ("key", "clear")),
    ]
    segs = [(" Determine which key was left.", False, ())]
    segs += _interleave(rng, crit, n_filler)
    segs += [(" The key on the table is", False, ())]
    return TaskInstance("delayed_disambiguation", "ambiguous", segs, f" {val}",
                        ("key", "clear"), frozenset({"key", "clear"}))


def distractor_heavy(rng: random.Random, n_filler: int) -> TaskInstance:
    # Many similar-looking numbers; only one is the answer.
    target = rng.randint(100, 999)
    distractors = [
        (f" A decoy figure mentioned in passing is {rng.randint(100,999)}.", False,
         ("figure",))
        for _ in range(3)
    ]
    crit = [(f" The official total of record is {target}.", True, ("official", "total"))]
    segs = [(" Find the official total among distractors.", False, ())]
    mixed = crit + distractors
    rng.shuffle(mixed)
    segs += _interleave(rng, mixed, n_filler)
    segs += [(" The official total of record is", False, ())]
    return TaskInstance("distractor_heavy", "distractor", segs, f" {target}",
                        ("official", "total", "record"),
                        frozenset({"official", "total", "record"}))


FAMILIES = {
    "needle_recall": needle_recall,
    "entity_state": entity_state,
    "incremental_constraint": incremental_constraint,
    "belief_revision": belief_revision,
    "delayed_disambiguation": delayed_disambiguation,
    "distractor_heavy": distractor_heavy,
}


def make_task(family: str, seed: int, n_filler: int) -> TaskInstance:
    return FAMILIES[family](random.Random(seed), n_filler)
