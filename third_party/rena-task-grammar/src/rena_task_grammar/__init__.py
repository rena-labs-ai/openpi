# Author: Mili Das

"""The canonical task-string grammar, shared by every backend producer and reader.

The forms, and nothing else:

    pick <object> and place in|on <location>
    pick <object>
    place <object> in|on <location>
    open <location>
    close <location>
    failed to pick <object>
    failed to place <object> in|on <location>
    failed to open <location>
    failed to close <location>

The ``failed to`` forms are segment labels, never an operator's input.

``compose`` is the only writer of a canonical string and ``parse`` the only reader,
so the grammar exists in one place instead of once per repo.

**Its own package, with zero dependencies, and the code lives in this ``__init__``
rather than a submodule.** Both are deliberate. The grammar is consumed by a zip
Lambda, two container images, the training boxes and the policy server, so anything
it imports has to be installable in all of them -- and a module is only as
dependency-free as the package enclosing it. Keeping the code in this ``__init__``
makes that structural: there is no parent package that could grow a dependency.

**So: never add an import to this file that is not stdlib.** If the grammar ever needs
one, it belongs in the caller.

**Shape is enforced; vocabulary is listed, never enforced.** The classes a task names
its object and location by live here, and ``to_class`` maps a word onto them. But
``compose`` and ``parse`` accept any word: ``pick teddy and place in thing by the
door`` is valid, because refusing an off-list word at ingest would drop a recording;
it trains under its own word, less a leading article, instead. A location's class only
picks the preposition before it.
``unprintable`` is the one refusal about content, and it is about bytes rather than
words: a control character is not text an operator can retype at inference time, so
``compose`` refuses to emit one and ``parse`` refuses to read one.

**Readers accept legacy phrasings too.** Raw recordings still carry the
pre-convention wording, so ``parse`` matches both grammars and reports which one it
saw. ``is_canonical`` is the check a migration gate wants; ``classify`` deliberately
answers the looser question.

Pattern order in ``_PATTERNS`` is load-bearing, in two ways that were each verified
against the live corpus rather than reasoned about:

  * **Loosest last.** Canonical ``pick (.+)`` matches any string starting with
    "pick ", including a whole legacy ``pick up the X and place it in the Y`` -- which
    would classify a pick-and-place recording as pick-only and segment it wrong.
  * **Legacy before its canonical counterpart.** Both grammars match
    ``place the X in the Y`` (an operator may legitimately type "the X" as the
    object), and only the legacy split extracts the object the existing ``_ch``
    labels were built from.
"""

from __future__ import annotations

import re
import unicodedata
from string import Formatter
from typing import NamedTuple

FORM_PICK_AND_PLACE = "pick_and_place"
FORM_PICK = "pick"
FORM_PLACE = "place"
FORM_OPEN = "open"
FORM_CLOSE = "close"
FORM_FAIL_PICK = "fail_pick"
FORM_FAIL_PLACE = "fail_place"
FORM_FAIL_OPEN = "fail_open"
FORM_FAIL_CLOSE = "fail_close"

# What a *recording* can be, and the actions it is made of, in order. The fail forms are
# only ever emitted as segment labels, so a source recording claiming one is not a task
# mode -- see ``classify_task_mode``.
FORM_ACTIONS = {
    FORM_PICK_AND_PLACE: (FORM_PICK, FORM_PLACE),
    FORM_PICK: (FORM_PICK,),
    FORM_PLACE: (FORM_PLACE,),
    FORM_OPEN: (FORM_OPEN,),
    FORM_CLOSE: (FORM_CLOSE,),
}

FORM_FAIL = {
    FORM_PICK: FORM_FAIL_PICK,
    FORM_PLACE: FORM_FAIL_PLACE,
    FORM_OPEN: FORM_FAIL_OPEN,
    FORM_CLOSE: FORM_FAIL_CLOSE,
}

# The one place the emitted shape is written down. ``compose`` formats these, ``parse``
# matches them, the docs quote them, and the data-alert notifier's advice is rendered
# from them.
CANONICAL_TEMPLATES = {
    FORM_PICK_AND_PLACE: "pick {object} and place {preposition} {location}",
    FORM_PICK: "pick {object}",
    FORM_PLACE: "place {object} {preposition} {location}",
    FORM_OPEN: "open {location}",
    FORM_CLOSE: "close {location}",
}
CANONICAL_TEMPLATES |= {
    fail: "failed to " + CANONICAL_TEMPLATES[action]
    for action, fail in FORM_FAIL.items()
}
# The fields a task must carry. The preposition is not one: it follows from the location.
_FIELDS = {
    form: tuple(
        name
        for _, name, _, _ in Formatter().parse(template)
        if name and name != "preposition"
    )
    for form, template in CANONICAL_TEMPLATES.items()
}

OBJECT_CLASSES = (
    "doll",
    "plush toy",
    "toy animal",
    "toy fruit",
    "lego",
    "toy",
    "clothing",
    "shoe",
    "trash",
    "plastic bottle",
    "can",
)

# Each location class with the preposition a task puts before it.
LOCATION_CLASSES = {
    "bin": "in",
    "trash can": "in",
    "basket": "in",
    "box": "in",
    "drawer": "in",
    "cabinet": "in",
    "mini fridge": "in",
    "shelf": "on",
    "table": "on",
}

# The location classes an open or close task can name.
OPENABLE_CLASSES = ("drawer", "cabinet", "mini fridge")

# Trained under their own names, outside the class list: never mapped onto a class.
KEPT_NAMES = ("fork", "marker", "tape", "clear bowl")

# Words operators use for a class in every session. A word whose meaning depends on the
# session is a per-recording override instead -- see ``to_classes``.
CLASS_ALIASES = {
    "dolls": "doll",
    "doll toy": "doll",
    "plush": "plush toy",
    "plush toys": "plush toy",
    "apple": "toy fruit",
    "red apple": "toy fruit",
    "banana": "toy fruit",
    "strawberry": "toy fruit",
    "legos": "lego",
    "car": "toy",
    "balloon": "toy",
    "boat": "toy",
    "block": "toy",
    "clothes": "clothing",
    "tshirt": "clothing",
    "shirt": "clothing",
    "sock": "clothing",
    "socks": "clothing",
    "cloth": "clothing",
    "shoes": "shoe",
    "slipper": "shoe",
    "garbage": "trash",
    "paper": "trash",
    "napkin": "trash",
    "plastic bottles": "plastic bottle",
    "cans": "can",
    "plastic bin": "bin",
    "plastic tray": "bin",
    "rack": "shelf",
}

# Index i is the stage head's logit i in every trained checkpoint: append, never renumber.
STAGES = (
    "picking",
    "picked_placing",
    "placed",
    "opening",
    "opened",
    "closing",
    "closed",
)

# Each action's working stage, then its done stage. Pick's done stage is place's working
# stage, so a pick-and-place recording climbs 0→1→2.
ACTION_STAGES = {
    FORM_PICK: (STAGES.index("picking"), STAGES.index("picked_placing")),
    FORM_PLACE: (STAGES.index("picked_placing"), STAGES.index("placed")),
    FORM_OPEN: (STAGES.index("opening"), STAGES.index("opened")),
    FORM_CLOSE: (STAGES.index("closing"), STAGES.index("closed")),
}

# IGNORECASE throughout: it is what the segmenter's own regexes did before they moved
# here, and a reader has no reason to be stricter than the reader it replaces.
_FLAGS = re.IGNORECASE


# Read as either preposition whatever the location, as the grammar accepts any word there.
_FIELD_PATTERNS = {"preposition": "|".join(sorted(set(LOCATION_CLASSES.values())))}


def _canonical_pattern(template: str) -> re.Pattern:
    """``template`` as a pattern: every free-text field lazy but the last, which takes
    the rest."""
    parts = list(Formatter().parse(template))
    last = max(i for i, (_, name, _, _) in enumerate(parts) if name)

    def field(i: int, name: str) -> str:
        free_text = ".+" if i == last else ".+?"
        return f"(?P<{name}>{_FIELD_PATTERNS.get(name, free_text)})"

    return re.compile(
        "".join(
            re.escape(literal) + (field(i, name) if name else "")
            for i, (literal, name, _, _) in enumerate(parts)
        ),
        _FLAGS,
    )


_CANONICAL_RE = {
    form: _canonical_pattern(template) for form, template in CANONICAL_TEMPLATES.items()
}

# Legacy phrasings, matching exactly what the segmenter matched, greedy groups
# included -- a "tidier" rewrite would silently re-split an object containing
# " in the " and change labels the existing ``_ch`` datasets were built from.
_LEGACY_PICK_AND_PLACE_RE = re.compile(
    r"pick up the (?P<object>.+?) and (?:place|put) (?:it )?in(?:to)? the (?P<location>.+)",
    _FLAGS,
)
_LEGACY_FAIL_PICK_RE = re.compile(r"failed to pick up the (?P<object>.+)", _FLAGS)
_LEGACY_PLACE_RE = re.compile(
    r"place the (?P<object>.+) in(?:to)? the (?P<location>.+)", _FLAGS
)
_LEGACY_PICK_RE = re.compile(r"pick up the (?P<object>.+)", _FLAGS)

# The legacy place template interpolated no object at all, so these strings name a
# destination and nothing to put there. Unconvertible by decision, not by omission:
# guessing the object would fabricate training data. Must be tested before the
# canonical place form, which would otherwise read the object as "it" before either
# preposition.
_LEGACY_OBJECTLESS_PLACE_RE = re.compile(
    r"place (?:it )?(?:in(?:to)?|on) the (?P<location>.+)", _FLAGS
)

# A pick-and-place task cut off after its preposition. Canonical ``pick (.+)`` would
# otherwise read "shoe and place on" as the object of a pick-only task.
_TRUNCATED_PICK_AND_PLACE_RE = _canonical_pattern(
    CANONICAL_TEMPLATES[FORM_PICK_AND_PLACE].removesuffix(" {location}")
)

# Written by the segmenter when a dataset carries no task metadata at all. It has to
# stay outside the grammar: canonical ``pick (.+)`` would otherwise read it as a pick
# of an object called "and place", turning every missing-metadata episode into a
# silently mis-segmented one instead of an excluded one.
NO_TASK_SENTINEL = "pick and place"

# The articles no object or location keeps; the review page drops them by this pattern too.
LEADING_ARTICLES = re.compile(r"^(?:(?:the|a|an) +)+", _FLAGS)
_WHITESPACE = re.compile(r"\s+")

# (form, legacy, pattern), in match order. See the module docstring -- both the
# loosest-last and the legacy-first orderings are load-bearing.
_PATTERNS = (
    (FORM_PICK_AND_PLACE, True, _LEGACY_PICK_AND_PLACE_RE),
    (FORM_FAIL_PICK, True, _LEGACY_FAIL_PICK_RE),
    (FORM_PLACE, True, _LEGACY_PLACE_RE),
    (FORM_PICK_AND_PLACE, False, _CANONICAL_RE[FORM_PICK_AND_PLACE]),
    (FORM_FAIL_PICK, False, _CANONICAL_RE[FORM_FAIL_PICK]),
    (FORM_FAIL_PLACE, False, _CANONICAL_RE[FORM_FAIL_PLACE]),
    (FORM_FAIL_OPEN, False, _CANONICAL_RE[FORM_FAIL_OPEN]),
    (FORM_FAIL_CLOSE, False, _CANONICAL_RE[FORM_FAIL_CLOSE]),
    (FORM_PLACE, False, _CANONICAL_RE[FORM_PLACE]),
    (FORM_OPEN, False, _CANONICAL_RE[FORM_OPEN]),
    (FORM_CLOSE, False, _CANONICAL_RE[FORM_CLOSE]),
    (FORM_PICK, True, _LEGACY_PICK_RE),
    (FORM_PICK, False, _CANONICAL_RE[FORM_PICK]),
)


class TaskGrammarError(ValueError):
    """A field set that cannot be expressed in the grammar (unknown form, blank
    required field). Mirrors ``rena_rec.task.TaskError`` on the recorder side."""

    def __init__(self, message: str, missing_field: str = ""):
        super().__init__(message)
        self.missing_field = missing_field


class TaskParts(NamedTuple):
    """A parsed task string. ``legacy`` says which grammar matched, which is the
    only way to tell a string that needs migrating from one that is already done."""

    form: str
    obj: str
    location: str
    legacy: bool


def unprintable(text: str) -> str:
    """The first non-printable character in ``text``, or ``""``."""
    return next((c for c in text or "" if not c.isprintable()), "")


def normalize(text: str) -> str:
    """The key under which two spellings of one object or location name match."""
    cleaned = unicodedata.normalize("NFKC", text or "").strip().strip("\"'“”")
    cleaned = _WHITESPACE.sub(" ", cleaned.casefold())
    return LEADING_ARTICLES.sub("", cleaned).strip()


def form_name(task_type: str) -> str:
    """The form a recorder's ``task_type`` names: case and padding are not part of it."""
    return (task_type or "").strip().lower()


def compose(task_type: str, obj: str, location: str = "") -> str:
    """The canonical string for ``task_type``, from free-text object and location.

    ``task_type`` is a form name, as the recorder writes it.

    Raises ``TaskGrammarError`` for an unknown type, a blank field the form needs, or a
    control character in a field. Refusing beats emitting ``place  in box``, and the
    caller is better placed to decide whether that is an operator's omission or a bug.
    A field the form does not name is ignored. A field's leading articles are dropped, so
    an operator's "the cup" composes to ``place cup in box``, not to legacy wording.
    """
    form = form_name(task_type)
    if form not in CANONICAL_TEMPLATES:
        raise TaskGrammarError(
            f"unknown task_type {task_type!r}; "
            f"expected one of {sorted(CANONICAL_TEMPLATES)}"
        )
    fields = {
        name: LEADING_ARTICLES.sub("", (value or "").strip())
        for name, value in (("object", obj), ("location", location))
    }
    fields["preposition"] = preposition(fields["location"])
    missing = next((name for name in _FIELDS[form] if not fields[name]), None)
    if missing:
        raise TaskGrammarError(
            f"task_type {task_type!r} requires a non-blank {missing}", missing
        )
    bad = unprintable(fields["object"]) or unprintable(fields["location"])
    if bad:
        raise TaskGrammarError(
            f"task field contains U+{ord(bad):04X}, which is not text"
        )
    return CANONICAL_TEMPLATES[form].format(**fields)


def parse(task: str) -> TaskParts | None:
    """The form, object and location of ``task``, canonical or legacy, else None.

    None means "not a task string in either grammar" -- the sentinel, a legacy place
    form carrying no object, a control character, or anything else. Callers must treat
    it as a data error rather than defaulting to a form; guessing produces mislabelled
    training data.
    """
    t = (task or "").strip()
    if not t or t.lower() == NO_TASK_SENTINEL or unprintable(t):
        return None
    if _LEGACY_OBJECTLESS_PLACE_RE.fullmatch(t):
        return None
    if _TRUNCATED_PICK_AND_PLACE_RE.fullmatch(t):
        return None
    for form, legacy, pattern in _PATTERNS:
        m = pattern.fullmatch(t)
        if m:
            fields = {name: value.strip() for name, value in m.groupdict().items()}
            if not all(fields.get(name) for name in _FIELDS[form]):
                # A field the form needs is whitespace -- refuse the whole string
                # rather than trying looser patterns: on ``pick up the   `` the legacy
                # form matches with an empty object, while canonical ``pick (.+)``
                # would match with the object "up the", fabricating a noun out of the
                # legacy prefix. Refusing on exactly the fields ``compose`` requires
                # is what makes ``compose(*parse(t))`` total, so callers on the
                # segmenter's per-episode loop cannot raise.
                return None
            return TaskParts(
                form=form,
                obj=fields.get("object", ""),
                location=fields.get("location", ""),
                legacy=legacy,
            )
    return None


def classify(task: str) -> str | None:
    """Which form ``task`` is in, accepting legacy phrasings, else None.

    Tolerant by design, so it answers "is this a task at all" -- not "is this
    canonical". Use ``is_canonical`` for the latter; a migration gate that checks
    this one instead would pass an entirely unmigrated dataset.
    """
    parts = parse(task)
    return parts.form if parts else None


def is_canonical(task: str) -> bool:
    """True only for a string already in the canonical grammar."""
    parts = parse(task)
    return bool(parts) and not parts.legacy


def is_objectless_place(task: str) -> bool:
    """True for a legacy place string naming a destination but no object.

    Its own predicate because it is the one unconvertible input with an operator
    fix: ``translate_legacy`` returns None both for this and for a phrasing we
    simply cannot map, and only the caller can route those to different people.
    """
    return bool(_LEGACY_OBJECTLESS_PLACE_RE.fullmatch((task or "").strip()))


def translate_legacy(task: str) -> str | None:
    """``task`` rewritten canonically, or None if it is not a legacy string.

    None covers both "already canonical" (nothing to do) and "untranslatable" --
    distinguish them with ``is_canonical`` and ``is_objectless_place``.
    """
    parts = parse(task)
    if not parts or not parts.legacy:
        return None
    return compose(parts.form, parts.obj, parts.location)


def to_class(word: str) -> str:
    """The class or kept name ``word`` stands for, else ``word`` unchanged."""
    key = normalize(word)
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if key in OBJECT_CLASSES or key in LOCATION_CLASSES or key in KEPT_NAMES:
        return key
    return word


def preposition(location: str) -> str:
    """The preposition before ``location``: its class's, else "in"."""
    return LOCATION_CLASSES.get(to_class(location), "in")


def to_classes(task: str) -> str | None:
    """``task`` canonical, with its object and location named by class; None where
    ``parse`` refuses it."""
    parts = parse(task)
    if parts is None:
        return None
    return compose(
        parts.form,
        to_class(parts.obj) if parts.obj else "",
        to_class(parts.location) if parts.location else "",
    )
