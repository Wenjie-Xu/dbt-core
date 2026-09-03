from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
import pytz
from freezegun import freeze_time

from dbt.artifacts.resources import NodeConfig
from dbt.artifacts.resources.types import BatchSize
from dbt.exceptions import DbtRuntimeError
from dbt.materializations.incremental.microbatch import MicrobatchBuilder

MODEL_CONFIG_BEGIN = datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)


class TestMicrobatchBuilder:
    @pytest.fixture(scope="class")
    def microbatch_model(self):
        model = mock.Mock()
        model.config = mock.MagicMock(NodeConfig)
        model.config.materialized = "incremental"
        model.config.incremental_strategy = "microbatch"
        model.config.begin = MODEL_CONFIG_BEGIN
        model.config.batch_size = BatchSize.day

        return model

    @freeze_time("2024-09-05 08:56:00")
    @pytest.mark.parametrize(
        "is_incremental,event_time_end,expected_end_time",
        [
            (
                False,
                None,
                datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_build_end_time(
        self, microbatch_model, is_incremental, event_time_end, expected_end_time
    ):
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model,
            is_incremental=is_incremental,
            event_time_start=None,
            event_time_end=event_time_end,
        )

        assert microbatch_builder.build_end_time() == expected_end_time

    @pytest.mark.parametrize(
        "is_incremental,event_time_start,checkpoint,batch_size,lookback,expected_start_time",
        [
            (
                False,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                0,
                # is_incremental: False => model.config.begin
                MODEL_CONFIG_BEGIN,
            ),
            # BatchSize.year
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                0,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                # Offset not applied when event_time_start provided
                1,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                0,
                # is_incremental=False + no start_time -> model.config.begin
                MODEL_CONFIG_BEGIN,
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                0,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                1,
                datetime(2023, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            # BatchSize.month
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                0,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                # Offset not applied when event_time_start provided
                1,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                0,
                # is_incremental=False + no start_time -> model.config.begin
                MODEL_CONFIG_BEGIN,
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                0,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                1,
                datetime(2024, 8, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            # BatchSize.day
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                0,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                # Offset not applied when event_time_start provided
                1,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                0,
                # is_incremental=False + no start_time -> model.config.begin
                MODEL_CONFIG_BEGIN,
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                0,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                1,
                datetime(2024, 9, 4, 0, 0, 0, 0, pytz.UTC),
            ),
            # BatchSize.hour
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                0,
                datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                # Offset not applied when event_time_start provided
                1,
                datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC),
            ),
            (
                False,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                0,
                # is_incremental=False + no start_time -> model.config.begin
                MODEL_CONFIG_BEGIN,
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                0,
                datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                1,
                datetime(2024, 9, 5, 7, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                BatchSize.hour,
                0,
                datetime(2024, 9, 4, 23, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                BatchSize.hour,
                1,
                datetime(2024, 9, 4, 22, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                BatchSize.day,
                0,
                datetime(2024, 9, 4, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                BatchSize.day,
                1,
                datetime(2024, 9, 3, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.month,
                0,
                datetime(2024, 8, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.month,
                1,
                datetime(2024, 7, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.year,
                0,
                datetime(2023, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                True,
                None,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.year,
                1,
                datetime(2022, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_build_start_time(
        self,
        microbatch_model,
        is_incremental,
        event_time_start,
        checkpoint,
        batch_size,
        lookback,
        expected_start_time,
    ):
        microbatch_model.config.batch_size = batch_size
        microbatch_model.config.lookback = lookback
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model,
            is_incremental=is_incremental,
            event_time_start=event_time_start,
            event_time_end=None,
        )

        assert microbatch_builder.build_start_time(checkpoint) == expected_start_time

    @pytest.mark.parametrize(
        "start,end,batch_size,expected_batches",
        [
            # BatchSize.year
            (
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2026, 1, 7, 3, 56, 0, 0, pytz.UTC),
                BatchSize.year,
                [
                    (
                        datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2026, 1, 7, 3, 56, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            # BatchSize.month
            (
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 11, 7, 3, 56, 0, 0, pytz.UTC),
                BatchSize.month,
                [
                    (
                        datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 11, 7, 3, 56, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            # BatchSize.day
            (
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 9, 7, 3, 56, 0, 0, pytz.UTC),
                BatchSize.day,
                [
                    (
                        datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 7, 3, 56, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            # BatchSize.hour
            (
                datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 3, 56, 0, 0, pytz.UTC),
                BatchSize.hour,
                [
                    (
                        datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 5, 3, 56, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            # Test when event_time_end matches the truncated batch size
            (
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.year,
                [
                    (
                        datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            (
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.month,
                [
                    (
                        datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            (
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC),
                BatchSize.day,
                [
                    (
                        datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC),
                    ),
                ],
            ),
            (
                datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC),
                datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC),
                BatchSize.hour,
                [
                    (
                        datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC),
                    ),
                    (
                        datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC),
                        datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC),
                    ),
                ],
            ),
        ],
    )
    def test_build_batches(self, microbatch_model, start, end, batch_size, expected_batches):
        microbatch_model.config.batch_size = batch_size
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, is_incremental=True, event_time_start=None, event_time_end=None
        )

        actual_batches = microbatch_builder.build_batches(start, end)
        assert len(actual_batches) == len(expected_batches)
        assert actual_batches == expected_batches

    def test_build_batches_minute_batch_interval(self, microbatch_model):
        microbatch_model.config.batch_size = BatchSize.minute
        microbatch_model.config.batch_interval = 20
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, is_incremental=True, event_time_start=None, event_time_end=None
        )

        actual_batches = microbatch_builder.build_batches(
            datetime(2026, 9, 2, 10, 0, 0, 0, pytz.UTC),
            datetime(2026, 9, 2, 11, 0, 0, 0, pytz.UTC),
        )
        expected_batches = [
            (
                datetime(2026, 9, 2, 10, 0, 0, 0, pytz.UTC),
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
            ),
            (
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
                datetime(2026, 9, 2, 10, 40, 0, 0, pytz.UTC),
            ),
            (
                datetime(2026, 9, 2, 10, 40, 0, 0, pytz.UTC),
                datetime(2026, 9, 2, 11, 0, 0, 0, pytz.UTC),
            ),
        ]
        assert len(actual_batches) == len(expected_batches)
        assert actual_batches == expected_batches

    def test_builder_normalizes_explicit_offsets_to_utc(self, microbatch_model):
        offset = timezone(timedelta(hours=8))
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model,
            is_incremental=True,
            event_time_start=datetime(2026, 9, 3, 16, 0, tzinfo=offset),
            event_time_end=datetime(2026, 9, 3, 17, 0, tzinfo=offset),
            default_end_time=datetime(2026, 9, 3, 18, 0, tzinfo=offset),
        )

        assert microbatch_builder.event_time_start == datetime(
            2026, 9, 3, 8, 0, tzinfo=timezone.utc
        )
        assert microbatch_builder.event_time_end == datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
        assert microbatch_builder.default_end_time == datetime(
            2026, 9, 3, 10, 0, tzinfo=timezone.utc
        )

    def test_build_start_time_truncates_offset_begin_in_utc(self, microbatch_model):
        microbatch_model.config.batch_size = BatchSize.minute
        microbatch_model.config.batch_interval = 20
        microbatch_model.config.begin = datetime.fromisoformat("2026-09-03T16:07:00+08:00")
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model,
            is_incremental=False,
            event_time_start=None,
            event_time_end=None,
        )

        assert microbatch_builder.build_start_time(None) == datetime(
            2026, 9, 3, 8, 0, tzinfo=timezone.utc
        )

    def test_build_end_time_ceilings_offset_in_utc(self, microbatch_model):
        microbatch_model.config.batch_size = BatchSize.minute
        microbatch_model.config.batch_interval = 20
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model,
            is_incremental=True,
            event_time_start=None,
            event_time_end=datetime.fromisoformat("2026-09-03T16:07:00+08:00"),
        )

        assert microbatch_builder.build_end_time() == datetime(
            2026, 9, 3, 8, 20, tzinfo=timezone.utc
        )

    def test_build_batches_normalizes_explicit_offsets_to_utc(self, microbatch_model):
        microbatch_model.config.batch_size = BatchSize.minute
        microbatch_model.config.batch_interval = 20
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, is_incremental=True, event_time_start=None, event_time_end=None
        )

        assert microbatch_builder.build_batches(
            datetime.fromisoformat("2026-09-03T16:00:00+08:00"),
            datetime.fromisoformat("2026-09-03T17:00:00+08:00"),
        ) == [
            (
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 3, 8, 20, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 8, 20, tzinfo=timezone.utc),
                datetime(2026, 9, 3, 8, 40, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 8, 40, tzinfo=timezone.utc),
                datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc),
            ),
        ]

    @pytest.mark.parametrize(
        "start,end",
        [
            (
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
            ),
        ],
    )
    def test_build_batches_rejects_non_increasing_range(self, microbatch_model, start, end):
        microbatch_model.config.batch_size = BatchSize.minute
        microbatch_model.config.batch_interval = 20
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, is_incremental=True, event_time_start=None, event_time_end=None
        )

        with pytest.raises(DbtRuntimeError, match="start.*less than.*end"):
            microbatch_builder.build_batches(start, end)

    @pytest.mark.parametrize(
        "batch_size,batch_interval,start,end,expected_batches",
        [
            (
                BatchSize.hour,
                2,
                datetime(2026, 9, 3, 0, 0, tzinfo=pytz.UTC),
                datetime(2026, 9, 3, 6, 0, tzinfo=pytz.UTC),
                [
                    (
                        datetime(2026, 9, 3, 0, 0, tzinfo=pytz.UTC),
                        datetime(2026, 9, 3, 2, 0, tzinfo=pytz.UTC),
                    ),
                    (
                        datetime(2026, 9, 3, 2, 0, tzinfo=pytz.UTC),
                        datetime(2026, 9, 3, 4, 0, tzinfo=pytz.UTC),
                    ),
                    (
                        datetime(2026, 9, 3, 4, 0, tzinfo=pytz.UTC),
                        datetime(2026, 9, 3, 6, 0, tzinfo=pytz.UTC),
                    ),
                ],
            ),
            (
                BatchSize.day,
                2,
                datetime(2026, 9, 1, tzinfo=pytz.UTC),
                datetime(2026, 9, 5, tzinfo=pytz.UTC),
                [
                    (
                        datetime(2026, 9, 1, tzinfo=pytz.UTC),
                        datetime(2026, 9, 3, tzinfo=pytz.UTC),
                    ),
                    (
                        datetime(2026, 9, 3, tzinfo=pytz.UTC),
                        datetime(2026, 9, 5, tzinfo=pytz.UTC),
                    ),
                ],
            ),
            (
                BatchSize.month,
                2,
                datetime(2026, 1, 1, tzinfo=pytz.UTC),
                datetime(2026, 5, 1, tzinfo=pytz.UTC),
                [
                    (
                        datetime(2026, 1, 1, tzinfo=pytz.UTC),
                        datetime(2026, 3, 1, tzinfo=pytz.UTC),
                    ),
                    (
                        datetime(2026, 3, 1, tzinfo=pytz.UTC),
                        datetime(2026, 5, 1, tzinfo=pytz.UTC),
                    ),
                ],
            ),
            (
                BatchSize.year,
                2,
                datetime(2020, 1, 1, tzinfo=pytz.UTC),
                datetime(2024, 1, 1, tzinfo=pytz.UTC),
                [
                    (
                        datetime(2020, 1, 1, tzinfo=pytz.UTC),
                        datetime(2022, 1, 1, tzinfo=pytz.UTC),
                    ),
                    (
                        datetime(2022, 1, 1, tzinfo=pytz.UTC),
                        datetime(2024, 1, 1, tzinfo=pytz.UTC),
                    ),
                ],
            ),
        ],
    )
    def test_build_batches_with_batch_interval(
        self, microbatch_model, batch_size, batch_interval, start, end, expected_batches
    ):
        microbatch_model.config.batch_size = batch_size
        microbatch_model.config.batch_interval = batch_interval
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, is_incremental=True, event_time_start=None, event_time_end=None
        )

        actual_batches = microbatch_builder.build_batches(start, end)

        assert actual_batches == expected_batches

    def test_build_jinja_context_for_incremental_batch(self, microbatch_model):
        context = MicrobatchBuilder.build_jinja_context_for_batch(
            model=microbatch_model,
            incremental_batch=True,
        )

        assert context["model"] == microbatch_model.to_dict()
        assert context["sql"] == microbatch_model.compiled_code
        assert context["compiled_code"] == microbatch_model.compiled_code

        assert context["is_incremental"]() is True
        assert context["should_full_refresh"]() is False

    def test_build_jinja_context_for_incremental_batch_false(self, microbatch_model):
        context = MicrobatchBuilder.build_jinja_context_for_batch(
            model=microbatch_model,
            incremental_batch=False,
        )

        assert context["model"] == microbatch_model.to_dict()
        assert context["sql"] == microbatch_model.compiled_code
        assert context["compiled_code"] == microbatch_model.compiled_code

        # Only build is_incremental callables when not first batch
        assert "is_incremental" not in context
        assert "should_full_refresh" not in context

    @pytest.mark.parametrize(
        "timestamp,batch_size,offset,expected_timestamp",
        [
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.year,
                1,
                datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.year,
                -1,
                datetime(2023, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.month,
                1,
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.month,
                -1,
                datetime(2024, 8, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.day,
                1,
                datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.day,
                -1,
                datetime(2024, 9, 4, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.hour,
                1,
                datetime(2024, 9, 5, 4, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.hour,
                -1,
                datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_offset_timestamp(self, timestamp, batch_size, offset, expected_timestamp):
        assert (
            MicrobatchBuilder.offset_timestamp(timestamp, batch_size, offset) == expected_timestamp
        )

    @pytest.mark.parametrize(
        "timestamp,batch_size,offset,batch_interval,expected_timestamp",
        [
            (
                datetime(2026, 9, 2, 10, 0, 0, 0, pytz.UTC),
                BatchSize.minute,
                1,
                20,
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
            ),
            (
                datetime(2026, 9, 2, 10, 0, 0, 0, pytz.UTC),
                BatchSize.minute,
                -1,
                20,
                datetime(2026, 9, 2, 9, 40, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_offset_timestamp_with_batch_interval(
        self, timestamp, batch_size, offset, batch_interval, expected_timestamp
    ):
        assert (
            MicrobatchBuilder.offset_timestamp(
                timestamp, batch_size, offset, batch_interval=batch_interval
            )
            == expected_timestamp
        )

    @pytest.mark.parametrize(
        "timestamp,batch_size,expected_timestamp",
        [
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.year,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.month,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.day,
                datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 5, 3, 56, 1, 1, pytz.UTC),
                BatchSize.hour,
                datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_truncate_timestamp(self, timestamp, batch_size, expected_timestamp):
        assert MicrobatchBuilder.truncate_timestamp(timestamp, batch_size) == expected_timestamp

    @pytest.mark.parametrize(
        "timestamp,batch_size,batch_interval,expected_timestamp",
        [
            (
                datetime(2026, 9, 2, 10, 7, 0, 0, pytz.UTC),
                BatchSize.minute,
                20,
                datetime(2026, 9, 2, 10, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
                BatchSize.minute,
                20,
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_truncate_timestamp_with_batch_interval(
        self, timestamp, batch_size, batch_interval, expected_timestamp
    ):
        assert (
            MicrobatchBuilder.truncate_timestamp(timestamp, batch_size, batch_interval)
            == expected_timestamp
        )

    def test_truncate_timestamp_normalizes_explicit_offset_to_utc(self):
        assert MicrobatchBuilder.truncate_timestamp(
            datetime.fromisoformat("2026-09-03T16:07:00+08:00"), BatchSize.minute, 20
        ) == datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)

    def test_ceiling_timestamp_normalizes_explicit_offset_to_utc(self):
        assert MicrobatchBuilder.ceiling_timestamp(
            datetime.fromisoformat("2026-09-03T16:07:00+08:00"), BatchSize.minute, 20
        ) == datetime(2026, 9, 3, 8, 20, tzinfo=timezone.utc)

    @pytest.mark.parametrize(
        "batch_size,start_time,expected_formatted_start_time",
        [
            (BatchSize.year, datetime(2020, 1, 1, 1), "2020"),
            (BatchSize.month, datetime(2020, 1, 1, 1), "202001"),
            (BatchSize.day, datetime(2020, 1, 1, 1), "20200101"),
            (BatchSize.hour, datetime(2020, 1, 1, 1), "20200101T01"),
            (BatchSize.minute, datetime(2026, 9, 2, 10, 20), "20260902T1020"),
        ],
    )
    def test_batch_id(
        self, batch_size: BatchSize, start_time: datetime, expected_formatted_start_time: str
    ) -> None:
        assert MicrobatchBuilder.batch_id(start_time, batch_size) == expected_formatted_start_time

    @pytest.mark.parametrize(
        "batch_size,batch_start,expected_formatted_batch_start",
        [
            (BatchSize.year, datetime(2020, 1, 1, 1), "2020"),
            (BatchSize.month, datetime(2020, 1, 1, 1), "2020-01"),
            (BatchSize.day, datetime(2020, 1, 1, 1), "2020-01-01"),
            (BatchSize.hour, datetime(2020, 1, 1, 1), "2020-01-01T01"),
            (BatchSize.minute, datetime(2026, 9, 2, 10, 20), "2026-09-02T1020"),
        ],
    )
    def test_format_batch_start(
        self, batch_size: BatchSize, batch_start: datetime, expected_formatted_batch_start: str
    ) -> None:
        assert (
            MicrobatchBuilder.format_batch_start(batch_start, batch_size)
            == expected_formatted_batch_start
        )

    @pytest.mark.parametrize(
        "batch_size,batch_start,expected_formatted_batch_start",
        [
            (BatchSize.minute, datetime(2026, 9, 3, 8, tzinfo=pytz.UTC), "2026-09-03T1600"),
            (BatchSize.minute, datetime(2026, 9, 3, 16, tzinfo=pytz.UTC), "2026-09-04T0000"),
            (BatchSize.day, datetime(2026, 9, 3, 16, tzinfo=pytz.UTC), "2026-09-04"),
            (BatchSize.month, datetime(2026, 9, 3, 16, tzinfo=pytz.UTC), "2026-09"),
            (BatchSize.year, datetime(2026, 9, 3, 16, tzinfo=pytz.UTC), "2026"),
        ],
    )
    def test_format_batch_start_for_filename(
        self, batch_size: BatchSize, batch_start: datetime, expected_formatted_batch_start: str
    ) -> None:
        assert (
            MicrobatchBuilder.format_batch_start_for_filename(batch_start, batch_size)
            == expected_formatted_batch_start
        )

    def test_batch_id_keeps_utc_semantics(self):
        batch_start = datetime(2026, 9, 3, 16, tzinfo=pytz.UTC)

        assert MicrobatchBuilder.batch_id(batch_start, BatchSize.minute) == "20260903T1600"

    @pytest.mark.parametrize(
        "timestamp,batch_size,expected_datetime",
        [
            (
                datetime(2024, 9, 17, 16, 6, 0, 0, pytz.UTC),
                BatchSize.hour,
                datetime(2024, 9, 17, 17, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 17, 16, 0, 0, 0, pytz.UTC),
                BatchSize.hour,
                datetime(2024, 9, 17, 16, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 17, 16, 6, 0, 0, pytz.UTC),
                BatchSize.day,
                datetime(2024, 9, 18, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 17, 0, 0, 0, 0, pytz.UTC),
                BatchSize.day,
                datetime(2024, 9, 17, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 17, 16, 6, 0, 0, pytz.UTC),
                BatchSize.month,
                datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.month,
                datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 9, 17, 16, 6, 0, 0, pytz.UTC),
                BatchSize.year,
                datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
            (
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
                BatchSize.year,
                datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_ceiling_timestamp(
        self, timestamp: datetime, batch_size: BatchSize, expected_datetime: datetime
    ) -> None:
        ceilinged = MicrobatchBuilder.ceiling_timestamp(timestamp, batch_size)
        assert ceilinged == expected_datetime

    @pytest.mark.parametrize(
        "timestamp,batch_size,batch_interval,expected_datetime",
        [
            (
                datetime(2026, 9, 2, 10, 7, 0, 0, pytz.UTC),
                BatchSize.minute,
                20,
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
            ),
            (
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
                BatchSize.minute,
                20,
                datetime(2026, 9, 2, 10, 20, 0, 0, pytz.UTC),
            ),
        ],
    )
    def test_ceiling_timestamp_with_batch_interval(
        self,
        timestamp: datetime,
        batch_size: BatchSize,
        batch_interval: int,
        expected_datetime: datetime,
    ) -> None:
        ceilinged = MicrobatchBuilder.ceiling_timestamp(timestamp, batch_size, batch_interval)
        assert ceilinged == expected_datetime
