"""The vendored prompt sampler and task grammar are the commits deps/ pins, so a checkpoint's commit fixes its prompts."""

import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("dep", ["autolabel-prompts", "rena-task-grammar"])
def test_the_committed_copy_is_the_pinned_tree(dep):
    """A tree id hashes content alone, so a committed edit to the copy fails here."""
    pin = json.loads(_git("show", f"HEAD:deps/{dep}.pin.json"))
    assert (
        _git("rev-parse", f"HEAD:third_party/{dep}") == pin["tree"]
    ), f"third_party/{dep} differs from its pin: change it in {pin['repo']}, then run scripts/bump_rena_dep.sh"


def test_training_imports_the_vendored_copies():
    import autolabel_prompts
    import rena_task_grammar

    for module, dep in ((autolabel_prompts, "autolabel-prompts"), (rena_task_grammar, "rena-task-grammar")):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(ROOT / "third_party" / dep)
