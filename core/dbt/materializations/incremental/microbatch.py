from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz

from dbt.artifacts.resources.types import BatchSize
from dbt.artifacts.schemas.batch_results import BatchType
from dbt.contracts.graph.nodes import ModelNode, NodeConfig
from dbt.event_time.event_time import normalize_datetime_to_utc
from dbt.exceptions import DbtInternalError, DbtRuntimeError


class MicrobatchBuilder:
    """A utility class for building microbatch definitions associated with a specific model"""

    def __init__(
        self,
        model: ModelNode,
        is_incremental: bool,
        event_time_start: Optional[datetime],
        event_time_end: Optional[datetime],
        default_end_time: Optional[datetime] = None,
    ):
        if model.config.incremental_strategy != "microbatch":
            raise DbtInternalError(
                f"Model '{model.name}' does not use 'microbatch' incremental_strategy."
            )
        self.model = model

        if self.model.config.batch_size is None:
            raise DbtRuntimeError(
                f"Microbatch model '{self.model.name}' does not have a 'batch_size' config (one of {[batch_size.value for batch_size in BatchSize]}) specificed."
            )

        self.is_incremental = is_incremental
        self.event_time_start = (
            normalize_datetime_to_utc(event_time_start) if event_time_start else None
        )
        self.event_time_end = normalize_datetime_to_utc(event_time_end) if event_time_end else None
        self.default_end_time = normalize_datetime_to_utc(
            default_end_time or datetime.now(pytz.UTC)
        )

    def _batch_interval(self) -> int:
        """Return the configured interval, defaulting to one unit."""
        batch_interval = self.model.config.batch_interval
        if (
            batch_interval is None
            or not isinstance(batch_interval, int)
            or isinstance(batch_interval, bool)
        ):
            return 1
        if batch_interval <= 0:
            raise DbtRuntimeError(
                f"Microbatch 'batch_interval' must be a positive int, but got: {batch_interval}."
            )
        return batch_interval

    def build_end_time(self):
        """Defaults the end_time to the current time in UTC unless a non `None` event_time_end was provided"""
        end_time = self.event_time_end or self.default_end_time
        return MicrobatchBuilder.ceiling_timestamp(
            end_time, self.model.config.batch_size, self._batch_interval()
        )

    def build_start_time(self, checkpoint: Optional[datetime]):
        """Create a start time based off the passed in checkpoint.

        If the checkpoint is `None`, or this is the first run of a microbatch model, then the
        model's configured `begin` value will be returned as a checkpoint is necessary
        to build a start time. This is because we build the start time relative to the checkpoint
        via the batchsize and offset, and we cannot offset a checkpoint if there is no checkpoint.
        """
        assert isinstance(self.model.config, NodeConfig)
        checkpoint = normalize_datetime_to_utc(checkpoint) if checkpoint else None
        batch_size = self.model.config.batch_size
        batch_interval = self._batch_interval()

        # Use event_time_start if it is provided.
        if self.event_time_start:
            return MicrobatchBuilder.truncate_timestamp(
                self.event_time_start, batch_size, batch_interval
            )

        # First run, use model's configured 'begin' as start.
        if not self.is_incremental or checkpoint is None:
            if not self.model.config.begin:
                raise DbtRuntimeError(
                    f"Microbatch model '{self.model.name}' requires a 'begin' configuration."
                )

            return MicrobatchBuilder.truncate_timestamp(
                self.model.config.begin, batch_size, batch_interval
            )

        lookback = self.model.config.lookback

        # If the checkpoint is equivalent to itself truncated then the checkpoint stradles
        # the batch line. In this case the last batch will end with the checkpoint, but start
        # should be the previous hour/day/month/year. Thus we need to increase the lookback by
        # 1 to get this affect properly.
        if checkpoint == MicrobatchBuilder.truncate_timestamp(
            checkpoint, batch_size, batch_interval
        ):
            lookback += 1

        return MicrobatchBuilder.offset_timestamp(
            checkpoint, batch_size, -1 * lookback, batch_interval
        )

    def build_batches(self, start: datetime, end: datetime) -> List[BatchType]:
        """
        Given a start and end datetime, builds a list of UTC-aware batches where each
        batch is the size of the model's batch_size (times the configured batch_interval).
        """
        start = normalize_datetime_to_utc(start)
        end = normalize_datetime_to_utc(end)
        if start >= end:
            raise DbtRuntimeError(
                f"Microbatch start time '{start}' must be less than end time '{end}'."
            )

        batch_size = self.model.config.batch_size
        batch_interval = self._batch_interval()
        curr_batch_start: datetime = start
        curr_batch_end: datetime = MicrobatchBuilder.offset_timestamp(
            curr_batch_start, batch_size, 1, batch_interval
        )

        batches: List[BatchType] = [(curr_batch_start, curr_batch_end)]
        while curr_batch_end < end:
            curr_batch_start = curr_batch_end
            curr_batch_end = MicrobatchBuilder.offset_timestamp(
                curr_batch_start, batch_size, 1, batch_interval
            )
            batches.append((curr_batch_start, curr_batch_end))

        # use exact end value as stop
        batches[-1] = (batches[-1][0], end)

        return batches

    @staticmethod
    def build_jinja_context_for_batch(model: ModelNode, incremental_batch: bool) -> Dict[str, Any]:
        """
        Create context with entries that reflect microbatch model + incremental execution state

        Assumes self.model has been (re)-compiled with necessary batch filters applied.
        """
        jinja_context: Dict[str, Any] = {}

        # Microbatch model properties
        jinja_context["model"] = model.to_dict()
        jinja_context["sql"] = model.compiled_code
        jinja_context["compiled_code"] = model.compiled_code

        # Add incremental context variables for batches running incrementally
        if incremental_batch:
            jinja_context["is_incremental"] = lambda: True
            jinja_context["should_full_refresh"] = lambda: False

        return jinja_context

    @staticmethod
    def offset_timestamp(
        timestamp: datetime, batch_size: BatchSize, offset: int, batch_interval: int = 1
    ) -> datetime:
        """Truncates the passed in timestamp based on the batch_size and then applies the offset by the batch_size.

        Note: It's important to understand that the offset applies to the truncated timestamp, not
        the origin timestamp. Thus being offset by a day isn't relative to the any given hour that day,
        but relative to the start of the day. So if the timestamp is the very end of a day, 2024-09-17 23:59:59,
        you have a batch size of a day, and an offset of +1, then the returned value ends up being only one
        second later, 2024-09-18 00:00:00.

        The offset moves by one configured `batch_interval` of the selected batch size.

        2024-09-17 16:06:00 + Batchsize.minute +1 (batch_interval=20) -> 2024-09-17 16:20:00
        2024-09-17 16:06:00 + Batchsize.hour -1 -> 2024-09-17 15:00:00
        2024-09-17 16:06:00 + Batchsize.hour +1 -> 2024-09-17 17:00:00
        2024-09-17 16:06:00 + Batchsize.day -1 -> 2024-09-16 00:00:00
        2024-09-17 16:06:00 + Batchsize.day +1 -> 2024-09-18 00:00:00
        2024-09-17 16:06:00 + Batchsize.month -1 -> 2024-08-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.month +1 -> 2024-10-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year -1 -> 2023-01-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year +1 -> 2025-01-01 00:00:00
        """
        truncated = MicrobatchBuilder.truncate_timestamp(timestamp, batch_size, batch_interval)

        offset_timestamp: datetime
        if batch_size == BatchSize.minute:
            offset_timestamp = truncated + timedelta(minutes=offset * batch_interval)
        elif batch_size == BatchSize.hour:
            offset_timestamp = truncated + timedelta(hours=offset * batch_interval)
        elif batch_size == BatchSize.day:
            offset_timestamp = truncated + timedelta(days=offset * batch_interval)
        elif batch_size == BatchSize.month:
            offset_timestamp = truncated
            for _ in range(abs(offset) * batch_interval):
                if offset < 0:
                    offset_timestamp = offset_timestamp - timedelta(days=1)
                else:
                    offset_timestamp = offset_timestamp + timedelta(days=31)
                offset_timestamp = MicrobatchBuilder.truncate_timestamp(
                    offset_timestamp, batch_size
                )
        elif batch_size == BatchSize.year:
            offset_timestamp = truncated.replace(year=truncated.year + offset * batch_interval)

        return offset_timestamp

    @staticmethod
    def truncate_timestamp(
        timestamp: datetime, batch_size: BatchSize, batch_interval: int = 1
    ) -> datetime:
        """Truncates the passed in timestamp in UTC based on the batch_size.

        The returned datetime is always UTC-aware. For `BatchSize.minute`, the timestamp
        is truncated to the nearest lower boundary
        aligned to `batch_interval` minutes within the hour (e.g. interval=20: 10:07 -> 10:00).
        Other batch sizes are truncated to their normal unit boundary; `batch_interval`
        controls the distance between those boundaries.

        2024-09-17 16:06:00 + Batchsize.minute (batch_interval=20) -> 2024-09-17 16:00:00
        2024-09-17 16:06:00 + Batchsize.hour -> 2024-09-17 16:00:00
        2024-09-17 16:06:00 + Batchsize.day -> 2024-09-17 00:00:00
        2024-09-17 16:06:00 + Batchsize.month -> 2024-09-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year -> 2024-01-01 00:00:00
        """
        timestamp = normalize_datetime_to_utc(timestamp)
        if batch_size == BatchSize.minute:
            if (
                not isinstance(batch_interval, int)
                or isinstance(batch_interval, bool)
                or batch_interval <= 0
            ):
                raise DbtRuntimeError(
                    f"Microbatch 'batch_interval' must be a positive int, but got: {batch_interval}."
                )
            truncated = datetime(
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
                timestamp.minute - (timestamp.minute % batch_interval),
                0,
                0,
                pytz.utc,
            )
        elif batch_size == BatchSize.hour:
            truncated = datetime(
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
                0,
                0,
                0,
                pytz.utc,
            )
        elif batch_size == BatchSize.day:
            truncated = datetime(
                timestamp.year, timestamp.month, timestamp.day, 0, 0, 0, 0, pytz.utc
            )
        elif batch_size == BatchSize.month:
            truncated = datetime(timestamp.year, timestamp.month, 1, 0, 0, 0, 0, pytz.utc)
        elif batch_size == BatchSize.year:
            truncated = datetime(timestamp.year, 1, 1, 0, 0, 0, 0, pytz.utc)

        return truncated

    @staticmethod
    def batch_id(start_time: datetime, batch_size: BatchSize) -> str:
        return MicrobatchBuilder.format_batch_start(start_time, batch_size).replace("-", "")

    @staticmethod
    def format_batch_start(batch_start: datetime, batch_size: BatchSize) -> str:
        """Format the passed in datetime based on the batch_size.

        2024-09-17 16:06:00 + Batchsize.minute -> 2024-09-17T1606
        2024-09-17 16:06:00 + Batchsize.hour  -> 2024-09-17T16
        2024-09-17 16:06:00 + Batchsize.day   -> 2024-09-17
        2024-09-17 16:06:00 + Batchsize.month -> 2024-09
        2024-09-17 16:06:00 + Batchsize.year  -> 2024
        """
        if batch_size == BatchSize.year:
            return batch_start.strftime("%Y")
        elif batch_size == BatchSize.month:
            return batch_start.strftime("%Y-%m")
        elif batch_size == BatchSize.day:
            return batch_start.strftime("%Y-%m-%d")
        elif batch_size == BatchSize.minute:
            return batch_start.strftime("%Y-%m-%dT%H%M")
        else:  # batch_size == BatchSize.hour
            return batch_start.strftime("%Y-%m-%dT%H")

    @staticmethod
    def ceiling_timestamp(
        timestamp: datetime, batch_size: BatchSize, batch_interval: int = 1
    ) -> datetime:
        """Move the timestamp to its UTC-aware ceiling for the given batch size.

        Note, if the timestamp is already the batch size ceiling, that is returned
        2024-09-17 16:06:00 + BatchSize.minute (batch_interval=20) -> 2024-09-17 16:20:00
        2024-09-17 16:06:00 + BatchSize.hour -> 2024-09-17 17:00:00
        2024-09-17 16:00:00 + BatchSize.hour -> 2024-09-17 16:00:00
        2024-09-17 16:06:00 + BatchSize.day -> 2024-09-18 00:00:00
        2024-09-17 00:00:00 + BatchSize.day -> 2024-09-17 00:00:00
        2024-09-17 16:06:00 + BatchSize.month -> 2024-10-01 00:00:00
        2024-09-01 00:00:00 + BatchSize.month -> 2024-09-01 00:00:00
        2024-09-17 16:06:00 + BatchSize.year -> 2025-01-01 00:00:00
        2024-01-01 00:00:00 + BatchSize.year -> 2024-01-01 00:00:00

        """
        timestamp = normalize_datetime_to_utc(timestamp)
        ceiling = truncated = MicrobatchBuilder.truncate_timestamp(
            timestamp, batch_size, batch_interval
        )
        if truncated != timestamp:
            ceiling = MicrobatchBuilder.offset_timestamp(truncated, batch_size, 1, batch_interval)
        return ceiling
