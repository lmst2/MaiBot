"""数据库迁移进度展示工具。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID
from rich.text import Text


def _format_duration(total_seconds: Optional[float]) -> str:
    """将秒数格式化为适合展示的耗时文本。

    Args:
        total_seconds: 总秒数；为空时表示暂不可用。

    Returns:
        str: 格式化后的耗时文本。
    """
    if total_seconds is None:
        return "--:--:--"
    safe_seconds = max(total_seconds, 0.0)
    return str(timedelta(seconds=int(safe_seconds)))


class MigrationSummaryColumn(ProgressColumn):
    """渲染数据库迁移总进度摘要列。"""

    def render(self, task: Task) -> Text:
        """渲染当前任务的总进度摘要。

        Args:
            task: 当前进度任务对象。

        Returns:
            Text: 渲染后的摘要文本。
        """
        display_total = task.fields.get("display_total", task.total)
        total_text = "?" if display_total is None else str(int(display_total))
        completed_text = str(int(task.completed))
        return Text(f"总迁移进度（{completed_text}/{total_text}）")


class MigrationSpeedColumn(ProgressColumn):
    """渲染数据库迁移速度列。"""

    def render(self, task: Task) -> Text:
        """渲染当前任务的速度信息。

        Args:
            task: 当前进度任务对象。

        Returns:
            Text: 渲染后的速度文本。
        """
        unit_name = str(task.fields.get("unit_name", "项"))
        if task.speed is None or task.speed <= 0:
            return Text(f"-- {unit_name}/s")
        return Text(f"{task.speed:.2f} {unit_name}/s")


class MigrationElapsedColumn(ProgressColumn):
    """渲染数据库迁移已用时间列。"""

    def render(self, task: Task) -> Text:
        """渲染当前任务的已用时间。

        Args:
            task: 当前进度任务对象。

        Returns:
            Text: 渲染后的已用时间文本。
        """
        return Text(f"已用时间 {_format_duration(task.elapsed)}")


class MigrationRemainingColumn(ProgressColumn):
    """渲染数据库迁移预估剩余时间列。"""

    def render(self, task: Task) -> Text:
        """渲染当前任务的预估剩余时间。

        Args:
            task: 当前进度任务对象。

        Returns:
            Text: 渲染后的预估剩余时间文本。
        """
        return Text(f"预估时间 {_format_duration(task.time_remaining)}")


class BaseMigrationProgressReporter(ABC):
    """数据库迁移进度上报器基类。"""

    def __enter__(self) -> "BaseMigrationProgressReporter":
        """进入进度上报上下文。

        Returns:
            BaseMigrationProgressReporter: 当前上报器实例。
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出进度上报上下文。

        Args:
            exc_type: 异常类型。
            exc_value: 异常实例。
            traceback: 异常追踪对象。
        """
        del exc_type, exc_value, traceback
        self.close()

    @abstractmethod
    def open(self) -> None:
        """打开进度上报资源。"""

    @abstractmethod
    def close(self) -> None:
        """关闭进度上报资源。"""

    @abstractmethod
    def start(
        self,
        total: int,
        description: str = "总迁移进度",
        unit_name: str = "表",
    ) -> None:
        """启动一个新的迁移进度任务。

        Args:
            total: 任务总数。
            description: 任务描述。
            unit_name: 进度单位名称。
        """

    @abstractmethod
    def advance(self, advance: int = 1, item_name: Optional[str] = None) -> None:
        """推进当前迁移进度任务。

        Args:
            advance: 本次推进的步数。
            item_name: 当前完成的项目名称。
        """


class NullMigrationProgressReporter(BaseMigrationProgressReporter):
    """不输出任何内容的空进度上报器。"""

    def close(self) -> None:
        """关闭空进度上报器。"""

    def start(
        self,
        total: int,
        description: str = "总迁移进度",
        unit_name: str = "表",
    ) -> None:
        """启动空进度任务。

        Args:
            total: 任务总数。
            description: 任务描述。
            unit_name: 进度单位名称。
        """
        del total, description, unit_name

    def advance(self, advance: int = 1, item_name: Optional[str] = None) -> None:
        """推进空进度任务。

        Args:
            advance: 本次推进的步数。
            item_name: 当前完成的项目名称。
        """
        del advance, item_name


class RichMigrationProgressReporter(BaseMigrationProgressReporter):
    """基于 ``rich`` 的数据库迁移进度上报器。"""

    def __init__(
        self,
        console: Optional[Console] = None,
        disable: Optional[bool] = None,
        refresh_per_second: int = 10,
    ) -> None:
        """初始化 ``rich`` 迁移进度上报器。

        Args:
            console: 输出使用的 ``rich`` 控制台。
            disable: 是否禁用进度条；为空时根据终端能力自动判断。
            refresh_per_second: 每秒刷新次数。
        """
        self.console = console or Console()
        self.disable = disable
        self.refresh_per_second = refresh_per_second
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def open(self) -> None:
        """打开 ``rich`` 进度条资源。"""
        effective_disable = not self.console.is_terminal if self.disable is None else self.disable
        self._progress = Progress(
            MigrationSummaryColumn(),
            BarColumn(),
            MigrationSpeedColumn(),
            MigrationElapsedColumn(),
            MigrationRemainingColumn(),
            console=self.console,
            transient=False,
            disable=effective_disable,
            refresh_per_second=self.refresh_per_second,
            expand=True,
        )
        self._progress.start()

    def close(self) -> None:
        """关闭 ``rich`` 进度条资源。"""
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None
        self._task_id = None

    def start(
        self,
        total: int,
        description: str = "总迁移进度",
        unit_name: str = "表",
    ) -> None:
        """启动一个新的 ``rich`` 迁移进度任务。

        Args:
            total: 任务总数。
            description: 任务描述。
            unit_name: 进度单位名称。
        """
        if self._progress is None:
            self.open()
        assert self._progress is not None
        effective_total = max(total, 1)
        self._task_id = self._progress.add_task(
            description,
            total=effective_total,
            display_total=total,
            unit_name=unit_name,
        )

    def advance(self, advance: int = 1, item_name: Optional[str] = None) -> None:
        """推进当前 ``rich`` 迁移进度任务。

        Args:
            advance: 本次推进的步数。
            item_name: 当前完成的项目名称。
        """
        del item_name
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, advance=advance)


def create_rich_migration_progress_reporter() -> BaseMigrationProgressReporter:
    """创建默认的 ``rich`` 迁移进度上报器。

    Returns:
        BaseMigrationProgressReporter: 默认迁移进度上报器实例。
    """
    return RichMigrationProgressReporter()
