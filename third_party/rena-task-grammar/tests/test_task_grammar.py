# Author: Mili Das

"""Tests for the canonical task grammar.

Three things carry the weight here. The **round trip** (``compose`` -> ``classify``)
pins every form; the **ordering** tests pin why ``_PATTERNS`` is in the order
it is, since three of its four hazards silently return the *wrong* form rather than None,
which is how a mislabelled dataset gets built without anything erroring; and the
**zero-dependency** test pins the reason this is its own package at all.
"""

import ast
import pathlib
import sys

import pytest

import rena_task_grammar
from rena_task_grammar import (
    ACTION_STAGES,
    CANONICAL_TEMPLATES,
    CLASS_ALIASES,
    FORM_ACTIONS,
    FORM_CLOSE,
    FORM_FAIL,
    FORM_FAIL_CLOSE,
    FORM_FAIL_OPEN,
    FORM_FAIL_PICK,
    FORM_FAIL_PLACE,
    FORM_OPEN,
    FORM_PICK,
    FORM_PICK_AND_PLACE,
    FORM_PLACE,
    KEPT_NAMES,
    LOCATION_CLASSES,
    NO_TASK_SENTINEL,
    OBJECT_CLASSES,
    OPENABLE_CLASSES,
    STAGES,
    TaskGrammarError,
    TaskParts,
    classify,
    compose,
    is_canonical,
    is_objectless_place,
    parse,
    to_class,
    to_classes,
    translate_legacy,
    unprintable,
)

# ── compose / classify round trip ───────────────────────────────────────────────
CANONICAL = {
    FORM_PICK_AND_PLACE: "pick lego and place in bin",
    FORM_PICK: "pick lego",
    FORM_PLACE: "place lego in bin",
    FORM_FAIL_PICK: "failed to pick lego",
    FORM_FAIL_PLACE: "failed to place lego in bin",
}
ACTION = {
    FORM_OPEN: "open drawer",
    FORM_CLOSE: "close drawer",
    FORM_FAIL_OPEN: "failed to open drawer",
    FORM_FAIL_CLOSE: "failed to close drawer",
}


@pytest.mark.parametrize("form,expected", sorted(CANONICAL.items()))
def test_compose_emits_the_canonical_form(form, expected):
    assert compose(form, "lego", "bin") == expected


@pytest.mark.parametrize("form,task", sorted(CANONICAL.items()))
def test_classify_round_trips_every_canonical_form(form, task):
    assert classify(task) == form
    assert is_canonical(task)


@pytest.mark.parametrize("form,task", sorted(CANONICAL.items()))
def test_parse_recovers_object_and_location(form, task):
    parts = parse(task)
    assert (parts.form, parts.obj, parts.legacy) == (form, "lego", False)
    assert parts.location == (
        "bin" if form in (FORM_PICK_AND_PLACE, FORM_PLACE, FORM_FAIL_PLACE) else ""
    )


def test_every_form_has_a_template():
    assert set(CANONICAL_TEMPLATES) == set(CANONICAL) | set(ACTION)


def test_recording_forms_exclude_the_fail_labels():
    """A recording is never *of* a failure; the segmenter labels one."""
    assert not set(FORM_FAIL.values()) & set(FORM_ACTIONS)
    assert set(FORM_ACTIONS) == {
        FORM_PICK_AND_PLACE,
        FORM_PICK,
        FORM_PLACE,
        FORM_OPEN,
        FORM_CLOSE,
    }


@pytest.mark.parametrize(
    ("task_type", "task"),
    [
        ("pick", "pick lego"),
        ("open", "open drawer"),
        ("close", "close drawer"),
        ("fail_open", "failed to open drawer"),
    ],
)
def test_compose_ignores_a_field_the_form_does_not_name(task_type, task):
    assert compose(task_type, "lego", "drawer") == task


@pytest.mark.parametrize("task", ["open", "close  ", "open d\x7frawer"])
def test_an_action_form_without_a_readable_location_does_not_parse(task):
    assert parse(task) is None


def test_action_stages_name_the_working_then_the_done_stage():
    named = {form: tuple(STAGES[i] for i in ids) for form, ids in ACTION_STAGES.items()}
    assert named == {
        FORM_PICK: ("picking", "picked_placing"),
        FORM_PLACE: ("picked_placing", "placed"),
        FORM_OPEN: ("opening", "opened"),
        FORM_CLOSE: ("closing", "closed"),
    }


def test_every_action_a_form_is_made_of_has_its_stages():
    actions = {action for form in FORM_ACTIONS.values() for action in form}
    assert actions <= set(ACTION_STAGES)


def test_every_action_has_a_fail_form_prefixing_failed_to():
    assert set(FORM_FAIL) == set(ACTION_STAGES)
    for action, fail in FORM_FAIL.items():
        assert CANONICAL_TEMPLATES[fail] == "failed to " + CANONICAL_TEMPLATES[action]


@pytest.mark.parametrize("form", list(CANONICAL_TEMPLATES))
def test_parse_reads_back_what_compose_writes_for_every_template(form):
    template = CANONICAL_TEMPLATES[form]
    obj = "red block" if "{object}" in template else ""
    location = "box" if "{location}" in template else ""
    assert parse(compose(form, obj, location)) == TaskParts(form, obj, location, False)


# ── ordering hazards: each of these returns a WRONG form, not None, if mis-ordered
def test_pick_and_place_wins_over_the_looser_pick_form():
    """``pick (.+)`` matches the whole pick-and-place string, taking "cube and place
    in bin" as the object -- a pick-and-place recording segmented as pick-only."""
    assert classify("pick cube and place in bin") == FORM_PICK_AND_PLACE
    assert parse("pick cube and place in bin").obj == "cube"


def test_a_legacy_pick_and_place_string_is_not_read_as_a_canonical_pick():
    """The sharpest ordering hazard: canonical ``pick (.+)`` happily matches
    ``pick up the block and place it in the box`` with object "up the block ...".
    Every legacy pattern must therefore be tried before it."""
    task = "pick up the block and place it in the box"
    parts = parse(task)
    assert parts.form == FORM_PICK_AND_PLACE
    assert (parts.obj, parts.location) == ("block", "box")
    assert parts.legacy


def test_the_no_task_sentinel_never_classifies():
    """``pick (.+)`` reads the sentinel as a pick of an object called "and place",
    which would segment every missing-metadata episode instead of excluding it."""
    assert classify(NO_TASK_SENTINEL) is None
    assert parse(NO_TASK_SENTINEL) is None


def test_a_legacy_place_string_keeps_its_legacy_object_split():
    """Both grammars match this one -- the canonical form would read the object as
    "the block", and the existing ``_ch`` labels were built from the legacy split."""
    parts = parse("place the block in the box")
    assert (parts.obj, parts.location, parts.legacy) == ("block", "box", True)


def test_a_legacy_object_containing_in_the_keeps_its_greedy_split():
    """The legacy place object group is greedy; re-splitting it would rename labels
    in datasets already segmented."""
    parts = parse("place the box in the corner in the shed")
    assert (parts.obj, parts.location) == ("box in the corner", "shed")


@pytest.mark.parametrize(
    "task,fail",
    [
        ("failed to pick lego", FORM_FAIL_PICK),
        ("failed to place lego in bin", FORM_FAIL_PLACE),
        ("failed to open drawer", FORM_FAIL_OPEN),
        ("failed to close drawer", FORM_FAIL_CLOSE),
    ],
)
def test_a_fail_form_is_never_read_as_its_action(task, fail):
    """A failed attempt is never labelled as the action succeeding."""
    assert classify(task) == fail


# ── legacy tolerance and translation ───────────────────────────────────────────
LEGACY = {
    "pick up the plush toy and place it in the box": (
        FORM_PICK_AND_PLACE,
        "pick plush toy and place in box",
    ),
    "pick up the napkin and put it in the box": (
        FORM_PICK_AND_PLACE,
        "pick napkin and place in box",
    ),
    "pick up the red cube and put it into the bin": (
        FORM_PICK_AND_PLACE,
        "pick red cube and place in bin",
    ),
    "pick up the plush toy": (FORM_PICK, "pick plush toy"),
    "place the plush toy in the box": (FORM_PLACE, "place plush toy in box"),
    "failed to pick up the plush toy": (FORM_FAIL_PICK, "failed to pick plush toy"),
}


@pytest.mark.parametrize("task,expected", sorted(LEGACY.items()))
def test_classify_accepts_every_legacy_phrasing(task, expected):
    """Tolerance is the point: every dataset in S3 still holds these."""
    assert classify(task) == expected[0]
    assert not is_canonical(task)


@pytest.mark.parametrize("task,expected", sorted(LEGACY.items()))
def test_translate_legacy_maps_every_legacy_phrasing(task, expected):
    translated = translate_legacy(task)
    assert translated == expected[1]
    assert is_canonical(translated)
    assert classify(translated) == expected[0]


def test_translate_legacy_is_a_no_op_on_canonical_strings():
    """None means "nothing to translate", not "bad" -- so a migration re-run over an
    already-migrated dataset rewrites nothing."""
    for task in CANONICAL.values():
        assert translate_legacy(task) is None


def test_is_canonical_is_the_check_a_migration_gate_needs():
    """``classify`` accepts both grammars by design, so a gate asserting only that
    would pass an entirely unmigrated dataset."""
    legacy = "pick up the block and place it in the box"
    assert classify(legacy) is not None
    assert not is_canonical(legacy)


# ── the unconvertible cases (D3) ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "task",
    [
        "place it in the box",  # the legacy place template, uninterpolated
        "place in the box",
        "place it into the box",
    ],
)
def test_a_place_string_with_no_object_is_unconvertible(task):
    """Guessing the object would fabricate training data. Note the canonical place
    form matches "place it in the box" with object "it", so this must be tested
    first -- otherwise D3 is silently defeated."""
    assert parse(task) is None
    assert classify(task) is None
    assert is_objectless_place(task)


def test_is_objectless_place_separates_the_two_untranslatable_causes():
    """``translate_legacy`` returns None for both, and they go to different people:
    a blank object is an operator's fix, an unknown phrasing is an engineer's."""
    assert is_objectless_place("place it in the box")
    assert not is_objectless_place("shove the block towards the box")
    assert translate_legacy("shove the block towards the box") is None


@pytest.mark.parametrize("task", ["", "   ", "move the block to the box", "explore"])
def test_non_task_strings_classify_as_nothing(task):
    assert classify(task) is None
    assert parse(task) is None


# ── compose refuses rather than emitting a malformed string ─────────────────────
def test_compose_takes_the_recorders_task_type_as_the_form_name():
    assert compose("pick_and_place", "lego", "bin") == "pick lego and place in bin"
    assert compose("PICK", "lego") == "pick lego"


@pytest.mark.parametrize("task_type", ["explore", "pnp", "full", "fail"])
def test_compose_rejects_an_unknown_task_type(task_type):
    with pytest.raises(TaskGrammarError, match="pick_and_place"):
        compose(task_type, "lego", "bin")


@pytest.mark.parametrize(
    "task_type", ["pick", "place", "pick_and_place", "fail_pick", "fail_place"]
)
def test_compose_rejects_a_blank_object(task_type):
    """D3, enforced where the string is built: ``place  in box`` must never exist."""
    with pytest.raises(TaskGrammarError) as e:
        compose(task_type, "   ", "bin")
    assert e.value.missing_field == "object"


@pytest.mark.parametrize(
    "task_type,obj",
    [
        ("place", "lego"),
        ("pick_and_place", "lego"),
        ("open", ""),
        ("close", ""),
        ("fail_place", "lego"),
        ("fail_open", ""),
        ("fail_close", ""),
    ],
)
def test_compose_rejects_a_blank_location_where_the_form_needs_one(task_type, obj):
    with pytest.raises(TaskGrammarError) as e:
        compose(task_type, obj, "  ")
    assert e.value.missing_field == "location"


def test_compose_ignores_a_location_the_form_does_not_use():
    assert compose("pick", "lego", "bin") == "pick lego"


@pytest.mark.parametrize("article", ["the", "The", "a", "an", "the the", "a the"])
def test_compose_drops_leading_articles_so_the_cup_does_not_read_as_legacy(article):
    task = compose("place", f"{article} cup", f"{article} box")
    assert task == "place cup in box"
    assert is_canonical(task)


def test_compose_strips_operator_whitespace():
    assert compose("place", "  lego  ", "  bin ") == "place lego in bin"


def test_the_object_stays_free_text():
    """D8: the convention fixes the sentence shape, never the noun."""
    task = compose("pick_and_place", "tiger", "thing by the door")
    assert task == "pick tiger and place in thing by the door"
    assert classify(task) == FORM_PICK_AND_PLACE


# ── the reason this is its own package ──────────────────────────────────────────
def test_the_grammar_imports_nothing_but_stdlib():
    """The property the package exists for, and it is not self-evident from reading it.

    This grammar is consumed by a zip Lambda, two container images, the training boxes
    and the policy server. It lived in ``rena_s3sync`` until 2026-08-19, where the file
    itself was stdlib-only but ``import boto3`` at the top of that package's ``__init__``
    made ``from rena_s3sync.task_grammar import ...`` fail in the serving venv, which has
    no boto3 -- so the claim was true of the file and false of the import. One new
    dependency here silently un-deploys the serving boundary again.
    """
    roots: set[str] = set()
    for path in pathlib.Path(rena_task_grammar.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert roots <= sys.stdlib_module_names, (
        f"non-stdlib import in the task grammar: "
        f"{sorted(roots - sys.stdlib_module_names)}"
    )


def test_unprintable_finds_control_characters_but_accepts_ordinary_text():
    assert unprintable("pick sock and place in basket") == ""
    assert unprintable("pick the thing by the door") == ""
    assert unprintable("") == ""
    assert unprintable(None) == ""
    assert unprintable("p\x7fsock") == "\x7f"
    assert unprintable("sock\n") == "\n"
    assert unprintable("sock\tbin") == "\t"


def test_parse_rejects_a_control_character_anywhere_in_the_string():
    assert parse("pick p\x7fsock and place in basket") is None
    assert parse("pick sock and place in b\x7fasket") is None
    assert parse("pick sock\tand place in basket") is None
    assert classify("pick p\x7fsock and place in basket") is None
    assert not is_canonical("pick p\x7fsock and place in basket")
    assert parse("pick sock and place in basket") == TaskParts(
        FORM_PICK_AND_PLACE, "sock", "basket", False
    )


def test_compose_refuses_a_control_character_so_it_cannot_emit_what_parse_rejects():
    with pytest.raises(TaskGrammarError, match="U\\+007F"):
        compose("pick_and_place", "p\x7fsock", "basket")
    with pytest.raises(TaskGrammarError, match="U\\+0009"):
        compose("pick_and_place", "sock", "bas\tket")
    with pytest.raises(TaskGrammarError, match="U\\+0009"):
        compose("pick", "the\tsock")
    assert compose("pick_and_place", "thing by the door", "bin #2 (left)") == (
        "pick thing by the door and place in bin #2 (left)"
    )


def test_stage_ids_never_move():
    """Each index is a logit of every trained stage head."""
    assert STAGES == (
        "picking",
        "picked_placing",
        "placed",
        "opening",
        "opened",
        "closing",
        "closed",
    )


CLASSES = {*OBJECT_CLASSES, *LOCATION_CLASSES}

# Every object and location word in the corpus census, as operators wrote it.
CENSUS_TERMS = (
    "plush toy", "lego", "shoe", "toy animal", "car", "block", "clothes", "garbage",
    "fork", "toys", "paper", "balloon", "marker", "toy fruit", "legos", "tshirt", "tape",
    "napkin", "shoes", "banana", "shirt", "socks", "slipper", "apple", "boat", "sock",
    "red apple", "tiger", "alligator", "Shoes", "trash", "cloth", "plush", "doll toy",
    "strawberry", "box", "bin", "plastic bin", "basket", "plastic tray", "trash can",
    "Plastic tray", "Rack",
)  # fmt: skip
# Their class depends on the session, so each recording's comes from its frames.
NEEDS_FRAMES = {"toys", "tiger", "alligator"}


@pytest.mark.parametrize("term", CENSUS_TERMS)
def test_every_census_term_has_a_class_unless_its_frames_decide(term):
    if term in NEEDS_FRAMES:
        assert to_class(term) == term
    else:
        assert to_class(term) in CLASSES | set(KEPT_NAMES)


# Each name rena-mobile-react's collect catalog sends, with the class it means.
APP_CATALOG = {
    "Legos": "lego", "Plush Toys": "plush toy", "Dolls": "doll", "Clothes": "clothing",
    "Shoes": "shoe", "Trash": "trash", "Plastic Bottles": "plastic bottle", "Cans": "can",
    "Bin": "bin", "Basket": "basket", "Shelf": "shelf", "Drawer": "drawer",
    "Cabinet": "cabinet", "Mini Fridge": "mini fridge", "Table": "table",
    "Trash Can": "trash can",
}  # fmt: skip


@pytest.mark.parametrize("name, cls", APP_CATALOG.items())
def test_every_app_catalog_name_maps_to_the_class_the_app_means(name, cls):
    assert to_class(name) == cls


def test_every_alias_names_a_class():
    assert set(CLASS_ALIASES.values()) <= CLASSES
    assert not set(CLASS_ALIASES) & (CLASSES | set(KEPT_NAMES))


def test_kept_names_stay_outside_the_class_list():
    assert not set(KEPT_NAMES) & CLASSES


def test_every_openable_is_a_location_class():
    assert set(OPENABLE_CLASSES) <= set(LOCATION_CLASSES)


def test_to_class_ignores_case_articles_and_spacing():
    assert to_class(" The  Legos ") == "lego"


def test_to_class_passes_an_off_list_word_through_unchanged():
    assert to_class("Rug") == "Rug"


@pytest.mark.parametrize(
    "form,location,expected",
    [
        (FORM_PICK_AND_PLACE, "shelf", "pick shoe and place on shelf"),
        (FORM_PLACE, "table", "place shoe on table"),
        (FORM_PICK_AND_PLACE, "Rack", "pick shoe and place on Rack"),
        (FORM_PICK_AND_PLACE, "rug", "pick shoe and place in rug"),
    ],
)
def test_the_location_class_picks_the_preposition(form, location, expected):
    """The operator's word is kept; only its class decides in or on."""
    assert compose(form, "shoe", location) == expected


@pytest.mark.parametrize(
    "task", ["pick shoe and place in shelf", "pick shoe and place on box"]
)
def test_parse_reads_either_preposition_before_any_location(task):
    parts = parse(task)
    assert (parts.form, parts.obj) == (FORM_PICK_AND_PLACE, "shoe")


def test_no_other_preposition_splits_off_a_location():
    assert parse("pick shoe and place under table").form == FORM_PICK


@pytest.mark.parametrize("task", ["place it in the shelf", "place it on the shelf"])
def test_a_place_with_no_object_is_refused_before_either_preposition(task):
    assert parse(task) is None
    assert is_objectless_place(task)


@pytest.mark.parametrize("task", ["pick shoe and place in", "pick shoe and place on "])
def test_a_pick_and_place_task_cut_off_after_its_preposition_is_not_a_pick(task):
    assert parse(task) is None


def test_a_legacy_string_translates_with_its_class_preposition():
    task = "pick up the shoe and place it in the shelf"
    assert translate_legacy(task) == "pick shoe and place on shelf"


@pytest.mark.parametrize(
    "task,expected",
    [
        ("pick Shoes and place in Rack", "pick shoe and place on shelf"),
        ("failed to pick legos", "failed to pick lego"),
        (
            "pick up the legos and place it in the plastic bin",
            "pick lego and place in bin",
        ),
        ("pick toys", "pick toys"),
    ],
)
def test_to_classes_names_object_and_location_by_class(task, expected):
    assert to_classes(task) == expected


def test_to_classes_refuses_what_parse_refuses():
    assert to_classes(NO_TASK_SENTINEL) is None
