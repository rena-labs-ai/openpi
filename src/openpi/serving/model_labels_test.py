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


@pytest.mark.parametrize("nsecs", [10**25, 10**30])
def test_an_out_of_range_timestamp_is_no_date_rather_than_a_crash(tmp_path, nsecs):
    # Positive and an int, so it passes the type checks, but datetime cannot
    # represent it — JSON has no int64 ceiling, and anything orbax itself wrote
    # (int64 nanoseconds, so at most April 2262) is in range. Raising here
    # stops the server from starting at all.
    step = _write_meta(tmp_path / "step", {"commit_timestamp_nsecs": nsecs})

    assert model_labels.checkpoint_trained_on(step, tz=UTC) is None


@pytest.mark.parametrize("bad_commit", ["not a number", -1, 0, True, 1.5])
def test_a_malformed_commit_time_is_not_papered_over_with_the_init_time(tmp_path, bad_commit):
    # init stands in only when commit is absent. A commit field that is there
    # but wrong means the file is not trustworthy, and a date from its other
    # field would be a guess presented as a fact.
    started = datetime.datetime(2026, 9, 23, 23, 0, tzinfo=UTC)
    step = _write_meta(
        tmp_path / "step",
        {"commit_timestamp_nsecs": bad_commit, "init_timestamp_nsecs": _nsecs(started)},
    )

    assert model_labels.checkpoint_trained_on(step, tz=UTC) is None


def test_init_stands_in_when_commit_is_missing_entirely(tmp_path):
    started = datetime.datetime(2026, 9, 23, 23, 0, tzinfo=UTC)
    step = _write_meta(tmp_path / "step", {"init_timestamp_nsecs": _nsecs(started)})

    assert model_labels.checkpoint_trained_on(step, tz=UTC) == datetime.date(2026, 9, 23)


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
