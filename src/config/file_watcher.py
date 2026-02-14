from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from watchfiles import Change, awatch

import asyncio

from src.common.logger import get_logger


logger = get_logger("file_watcher")


@dataclass(frozen=True)
class FileChange:
    change_type: Change
    path: Path


ChangeCallback = Callable[[Sequence[FileChange]], Awaitable[None]]


class FileWatcher:
    def __init__(self, paths: Iterable[Path], debounce_ms: int = 600) -> None:
        self._paths = [path.resolve() for path in paths]
        self._debounce_ms = debounce_ms
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._running

    async def start(self, callback: ChangeCallback) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(callback))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            return

    async def _run(self, callback: ChangeCallback) -> None:
        try:
            async for changes in awatch(*self._paths, debounce=self._debounce_ms):
                if not self._running:
                    break
                try:
                    await callback(self._normalize_changes(changes))
                except Exception as exc:
                    logger.warning(f"文件变更回调执行失败: {exc}")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"文件监视器运行异常: {exc}")

    def _normalize_changes(self, changes: set[tuple[Change, str]]) -> list[FileChange]:
        return [FileChange(change_type=change, path=Path(path)) for change, path in changes]
