"""Shared sequential batch processing over (extent, source_layer_id) jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsTask,
)


def resolve_batch_source_layer(layer_id: str) -> QgsRasterLayer | None:
    """Return a valid raster for *layer_id*, or None if missing or invalid."""
    layer = QgsProject.instance().mapLayer(layer_id)
    if isinstance(layer, QgsRasterLayer) and layer.isValid():
        return layer
    return None


@dataclass
class BatchStepResult:
    """Outcome of starting one batch step."""

    task: QgsTask | None = None
    skip: bool = False
    """True: no task started; try the next job or finish (missing layer, skip metadata, etc.)."""
    abort: bool = False
    """True: stop the batch immediately (e.g. could not build mask/task)."""


class BatchExtentRunner:
    """Sequential QgsTask queue: one extent at a time, shared skip/finish/cleanup."""

    def __init__(
        self,
        iface,
        *,
        label: str,
        get_task: Callable[[], QgsTask | None],
        set_task: Callable[[QgsTask | None], None],
        is_sibling_task_running: Callable[[], bool],
        batch_already_running_msg: str,
        sibling_task_running_msg: str,
    ) -> None:
        self.iface = iface
        self._label = label
        self._get_task = get_task
        self._set_task = set_task
        self._is_sibling_task_running = is_sibling_task_running
        self._batch_already_running_msg = batch_already_running_msg
        self._sibling_task_running_msg = sibling_task_running_msg

        self._active = False
        self._queue: list[tuple[QgsRectangle, str]] = []
        self._index = 0
        self._total = 0
        self._on_step: (
            Callable[[QgsRectangle, str, QgsRasterLayer, int, int], BatchStepResult]
            | None
        ) = None
        self._on_after_step: Callable[[], None] | None = None
        self._step_done_handler: Callable[[], None] | None = None
        self._on_finalize: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        """True while a batch is prepared or running (queue not yet fully drained)."""
        return self._active

    @property
    def step_index(self) -> int:
        """Index of the step currently running or just completed (1-based)."""
        return self._index

    @property
    def total(self) -> int:
        """Number of extents in the batch when it started (unchanged until the batch ends)."""
        return self._total

    def try_begin_batch(self, jobs: list[tuple[QgsRectangle, str]]) -> bool:
        """Validate and prepare the batch. Returns False if batch should not start."""
        if not jobs:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "No extents to process.",
                level=Qgis.Warning,
                duration=3,
            )
            return False
        if self._active:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                self._batch_already_running_msg,
                level=Qgis.Warning,
                duration=4,
            )
            return False
        if self._is_sibling_task_running():
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                self._sibling_task_running_msg,
                level=Qgis.Warning,
                duration=4,
            )
            return False

        self._active = True
        self._queue = list(jobs)
        self._index = 0
        self._total = len(jobs)
        return True

    def prepare_without_try_begin(self, jobs: list[tuple[QgsRectangle, str]]) -> None:
        """Set queue state after custom validation (e.g. video dialog). Caller must ensure not already active."""
        self._active = True
        self._queue = list(jobs)
        self._index = 0
        self._total = len(jobs)

    def start(
        self,
        jobs: list[tuple[QgsRectangle, str]],
        on_step: Callable[
            [QgsRectangle, str, QgsRasterLayer, int, int], BatchStepResult
        ],
        *,
        on_after_each_step: Callable[[], None] | None = None,
        on_finalize: Callable[[], None] | None = None,
        start_log: str | None = None,
        start_message: str | None = None,
        start_message_duration: int = 4,
    ) -> bool:
        """Begin batch: try_begin_batch, optional log/message, then run first step."""
        if not self.try_begin_batch(jobs):
            return False
        if start_log:
            QgsMessageLog.logMessage(start_log, "ICEYE Toolbox", Qgis.Info)
        if start_message:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                start_message,
                level=Qgis.Info,
                duration=start_message_duration,
            )
        self._on_step = on_step
        self._on_after_step = on_after_each_step
        self._on_finalize = on_finalize
        self._run_next()
        return True

    def run_next_after_prepare(
        self,
        on_step: Callable[
            [QgsRectangle, str, QgsRasterLayer, int, int], BatchStepResult
        ],
        *,
        on_after_each_step: Callable[[], None] | None = None,
        on_finalize: Callable[[], None] | None = None,
    ) -> None:
        """Continue after prepare_without_try_begin (e.g. video after dialog)."""
        self._on_step = on_step
        self._on_after_step = on_after_each_step
        self._on_finalize = on_finalize
        self._run_next()

    def _run_next(self) -> None:
        if self._on_step is None:
            return
        if not self._queue:
            self._finish_batch()
            return

        extent, layer_id = self._queue.pop(0)
        self._index += 1

        source_layer = resolve_batch_source_layer(layer_id)
        if source_layer is None:
            self._log_skip_missing_layer(layer_id)
            self._after_skip_or_recurse()
            return

        self.iface.setActiveLayer(source_layer)

        result = self._on_step(extent, layer_id, source_layer, self._index, self._total)
        if result.abort:
            self._cleanup_state()
            return
        if result.skip:
            self._after_skip_or_recurse()
            return
        if result.task is None:
            self._after_skip_or_recurse()
            return

        task = result.task
        self._set_task(task)

        def _done() -> None:
            self._on_step_finished()

        self._step_done_handler = _done
        task.taskCompleted.connect(_done)
        task.taskTerminated.connect(_done)
        QgsApplication.taskManager().addTask(task)

    def _on_step_finished(self) -> None:
        if not self._active:
            return

        task = self._get_task()
        if task is not None:
            handler = self._step_done_handler
            if handler is not None:
                try:
                    task.taskCompleted.disconnect(handler)
                except (TypeError, RuntimeError):
                    pass
                try:
                    task.taskTerminated.disconnect(handler)
                except (TypeError, RuntimeError):
                    pass
        self._step_done_handler = None
        self._set_task(None)

        if self._on_after_step is not None:
            self._on_after_step()

        if self._queue:
            self._run_next()
        else:
            self._finish_batch()

    def _log_skip_missing_layer(self, layer_id: str) -> None:
        QgsMessageLog.logMessage(
            f"{self._label}: source layer {layer_id!r} missing or invalid; "
            f"skipping step {self._index}/{self._total}",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        self.iface.messageBar().pushMessage(
            "ICEYE Toolbox",
            f"Skipped step {self._index}: source layer no longer in the project.",
            level=Qgis.Warning,
            duration=5,
        )

    def _after_skip_or_recurse(self) -> None:
        if self._queue:
            self._run_next()
        else:
            self._finish_batch()

    def _finish_batch(self) -> None:
        n = self._total
        self._cleanup_state()
        self.iface.messageBar().pushMessage(
            "ICEYE Toolbox",
            f"{self._label} finished ({n} area(s)).",
            level=Qgis.Info,
            duration=5,
        )
        QgsMessageLog.logMessage(
            f"{self._label}: queue empty, batch finished",
            "ICEYE Toolbox",
            Qgis.Info,
        )

    def _cleanup_state(self) -> None:
        self._active = False
        self._queue = []
        self._total = 0
        self._index = 0
        self._on_step = None
        self._on_after_step = None
        self._step_done_handler = None
        self._set_task(None)
        if self._on_finalize is not None:
            self._on_finalize()
        self._on_finalize = None
