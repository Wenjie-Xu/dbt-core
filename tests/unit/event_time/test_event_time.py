from datetime import datetime, timedelta, timezone, tzinfo

import pytest
import pytz

from dbt.artifacts.resources.types import BatchSize
from dbt.event_time.event_time import normalize_datetime_to_utc, offset_timestamp


class NoneOffsetTimezone(tzinfo):
    def utcoffset(self, dt):
        return None


class TestEventTime:

    @pytest.mark.parametrize(
        "timestamp,expected_timestamp",
        [
            (
                datetime(2026, 9, 3, 8, 0),
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 16, 0, tzinfo=NoneOffsetTimezone()),
                datetime(2026, 9, 3, 16, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 16, 0, tzinfo=timezone(timedelta(hours=8))),
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 2, 0, tzinfo=timezone(-timedelta(hours=6))),
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
            ),
        ],
    )
    def test_normalize_datetime_to_utc(self, timestamp, expected_timestamp):
        assert normalize_datetime_to_utc(timestamp) == expected_timestamp

    @pytest.mark.parametrize(
        "timestamp,batch_size,offset,expected_timestamp",
        [
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.year,
                1,
                datetime(2025, 9, 5, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.year,
                -1,
                datetime(2023, 9, 5, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.month,
                1,
                datetime(2024, 10, 5, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.month,
                -1,
                datetime(2024, 8, 5, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.day,
                1,
                datetime(2024, 9, 6, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.day,
                -1,
                datetime(2024, 9, 4, 3, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.hour,
                1,
                datetime(2024, 9, 5, 4, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.hour,
                -1,
                datetime(2024, 9, 5, 2, 56, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.minute,
                20,
                datetime(2024, 9, 5, 4, 16, 1, 1, pytz.UTC),
            ),
            (
                datetime(2024, 1, 31, 16, 6, 0, 0, pytz.UTC),
                BatchSize.month,
                1,
                datetime(2024, 2, 29, 16, 6, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 2, 29, 16, 6, 0, 0, pytz.UTC),
                BatchSize.year,
                1,
                datetime(2025, 2, 28, 16, 6, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_offset_timestamp(self, timestamp, batch_size, offset, expected_timestamp):
        assert offset_timestamp(timestamp, batch_size, offset) == expected_timestamp
