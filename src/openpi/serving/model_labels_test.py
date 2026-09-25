import datetime
import json
import pathlib
import zoneinfo

import pytest

from openpi.serving import model_labels

UTC = datetime.UTC


def _nsecs(when: datetime.datetime) -> int:
    return int(when.timestamp() * 1e9)


def _write_meta(step_dir: pathlib.Path, meta: object) -> pathlib.Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / model_labels.STEP_METADATA_FILENAME).write_text(json.dumps(meta))
    return step_dir


def test_reads_the_day_orbax_committed_the_step(tmp_path):
    committed = datetime.datetime(2026, 9, 24, 6, 15, tzinfo=UTC)
    step = _write_meta(tmp_path / "e22" / "70000", {"commit_timestamp_nsecs": _nsecs(committed)})

    assert model_labels.checkpoint_trained_on(step, tz=UTC) == datetime.date(2026, 9, 24)


def test_uses_the_serving_zone_so_a_late_evening_run_keeps_its_day(tmp_path):
    # 02:30 UTC on the 25th is 22:30 on the 24th in New York. Reported in UTC
    # it would read as the next day's model.
    committed = datetime.datetime(2026, 9, 25, 2, 30, tzinfo=UTC)
    step = _write_meta(tmp_path / "step", {"commit_timestamp_nsecs": _nsecs(committed)})

    assert model_labels.checkpoint_trained_on(step, tz=zoneinfo.ZoneInfo("America/New_York")) == datetime.date(
        2026, 9, 24
    )
    assert model_labels.checkpoint_trained_on(step, tz=UTC) == datetime.date(2026, 9, 25)


def test_falls_back_to_when_the_step_began_writing(tmp_path):
    started = datetime.datetime(2026, 9, 23, 23, 0, tzinfo=UTC)
    step = _write_meta(tmp_path / "step", {"commit_timestamp_nsecs": None, "init_timestamp_nsecs": _nsecs(started)})

    assert model_labels.checkpoint_trained_on(step, tz=UTC) == datetime.date(2026, 9, 23)


def test_prefers_the_commit_time_when_both_are_present(tmp_path):
    step = _write_meta(
        tmp_path / "step",
        {
            "init_timestamp_nsecs": _nsecs(datetime.datetime(2026, 9, 23, 23, 59, tzinfo=UTC)),
            "commit_timestamp_nsecs": _nsecs(datetime.datetime(2026, 9, 24, 0, 1, tzinfo=UTC)),
        },
    )

    assert model_labels.checkpoint_trained_on(step, tz=UTC) == datetime.date(2026, 9, 24)


def test_ignores_the_directory_mtime_when_there_is_no_metadata(tmp_path):
    # A checkpoint synced onto the host has a fresh mtime. Showing that would
    # label an old model as last night's.
    step = tmp_path / "step"
    step.mkdir()
    (step / "params").mkdir()

    assert model_labels.checkpoint_trained_on(step, tz=UTC) is None


@pytest.mark.parametrize(
    "meta",
    [
        "not json at all",
        [],
        {},
        {"commit_timestamp_nsecs": None},
        {"commit_timestamp_nsecs": 0},
        {"commit_timestamp_nsecs": -5},
        {"commit_timestamp_nsecs": "1790000000000000000"},
        {"commit_timestamp_nsecs": True},
    ],
)
def test_never_guesses_a_date_from_metadata_it_cannot_read(tmp_path, meta):
    step = tmp_path / "step"
    step.mkdir()
    raw = meta if isinstance(meta, str) else json.dumps(meta)
    (step / model_labels.STEP_METADATA_FILENAME).write_text(raw)

    assert model_labels.checkpoint_trained_on(step, tz=UTC) is None


def test_a_missing_directory_is_no_date_rather_than_an_error(tmp_path):
    # Raising here would take the whole model set down with it at startup.
    assert model_labels.checkpoint_trained_on(tmp_path / "gone", tz=UTC) is None


def test_label_carries_the_date_and_not_the_time():
    assert model_labels.dated_label("v22", datetime.date(2026, 9, 24)) == "v22 · Sep 24"


def test_single_digit_days_are_not_zero_padded():
    assert model_labels.dated_label("v22", datetime.date(2026, 10, 4)) == "v22 · Oct 4"


def test_label_is_unchanged_without_a_date():
    assert model_labels.dated_label("v22", None) == "v22"


def test_an_id_standing_in_for_a_missing_label_is_dated_too():
    assert model_labels.dated_label("e13/70000", datetime.date(2026, 9, 21)) == "e13/70000 · Sep 21"
