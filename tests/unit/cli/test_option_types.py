from datetime import datetime
from typing import Union

import freezegun
import pytest
import pytz
from click import BadParameter, Option

from dbt.cli.option_types import YAML, SampleType, UTCDateTime
from dbt.event_time.sample_window import SampleWindow


class TestYAML:
    @pytest.mark.parametrize(
        "raw_value,expected_converted_value",
        [
            ("{}", {}),
            ("{'test_var_key': 'test_var_value'}", {"test_var_key": "test_var_value"}),
        ],
    )
    def test_yaml_init(self, raw_value, expected_converted_value):
        converted_value = YAML().convert(raw_value, Option(["--vars"]), None)
        assert converted_value == expected_converted_value

    @pytest.mark.parametrize(
        "invalid_yaml_str",
        ["{", ""],
    )
    def test_yaml_init_invalid_yaml_str(self, invalid_yaml_str):
        with pytest.raises(BadParameter) as e:
            YAML().convert(invalid_yaml_str, Option(["--vars"]), None)
        assert "--vars" in e.value.format_message()


class TestUTCDateTime:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (
                "2026-09-03T16:00:00+08:00",
                datetime(2026, 9, 3, 8, 0, tzinfo=pytz.UTC),
            ),
            (
                "2026-09-03T08:00:00Z",
                datetime(2026, 9, 3, 8, 0, tzinfo=pytz.UTC),
            ),
            ("2026-09-03", datetime(2026, 9, 3, 0, 0, tzinfo=pytz.UTC)),
            (
                "2026-09-03T08:00:00",
                datetime(2026, 9, 3, 8, 0, tzinfo=pytz.UTC),
            ),
            (
                "2026-09-03 08:00:00",
                datetime(2026, 9, 3, 8, 0, tzinfo=pytz.UTC),
            ),
        ],
    )
    def test_convert_returns_utc_aware_datetime(self, value, expected):
        assert UTCDateTime().convert(value, Option(["--event-time-start"]), None) == expected

    def test_convert_rejects_named_timezone(self):
        with pytest.raises(BadParameter):
            UTCDateTime().convert(
                "2026-09-03T16:00:00 Asia/Shanghai",
                Option(["--event-time-start"]),
                None,
            )


class TestSampleType:
    @pytest.mark.parametrize(
        "input,expected_result",
        [
            (
                "{'start': '2025-01-24', 'end': '2025-01-27'}",
                SampleWindow(
                    start=datetime(2025, 1, 24, 0, 0, 0, 0, pytz.UTC),
                    end=datetime(2025, 1, 27, 0, 0, 0, 0, pytz.UTC),
                ),
            ),
            (
                "{'start': '2026-09-03T16:00:00+08:00', 'end': '2026-09-03T17:00:00+08:00'}",
                SampleWindow(
                    start=datetime(2026, 9, 3, 8, 0, 0, 0, pytz.UTC),
                    end=datetime(2026, 9, 3, 9, 0, 0, 0, pytz.UTC),
                ),
            ),
            (
                "{'tart': '2025-01-24', 'bend': '2025-01-27'}",
                BadParameter('Field "start" of type datetime is missing in SampleWindow instance'),
            ),
            (
                "{}",
                BadParameter('Field "start" of type datetime is missing in SampleWindow instance'),
            ),
            (
                "cats",
                BadParameter(
                    "Runtime Error\n  Cannot load SAMPLE_WINDOW from 'cats'. Must be of form 'DAYS_INT GRAIN_SIZE'."
                ),
            ),
        ],
    )
    def test_convert(self, input: str, expected_result: Union[SampleWindow, Exception]):
        try:
            result = SampleType().convert(input, Option(["--sample"]), None)
            assert result == expected_result
        except Exception as e:
            assert str(e) == str(expected_result)

    # this had to be a seprate test case because the @freezegun.freeze_time
    # was screwing up the instantiation of SampleWindow.from_dict calls for the
    # other test cases
    @freezegun.freeze_time("2025-01-28T02:03:0Z")
    def test_convert_relative(self):
        input = "3 days"
        expected_result = SampleWindow(
            start=datetime(2025, 1, 25, 2, 3, 0, 0, pytz.UTC),
            end=datetime(2025, 1, 28, 2, 3, 0, 0, pytz.UTC),
        )
        result = SampleType().convert(input, Option(["--sample"]), None)
        assert result == expected_result
