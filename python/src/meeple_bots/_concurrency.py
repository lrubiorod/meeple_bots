"""Internal helpers for bounded, reproducible match-level concurrency."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

WorkerSetting: TypeAlias = int | Literal["auto"]

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


def resolve_workers(workers: WorkerSetting) -> int:
    """Validate a worker setting and resolve ``auto`` to physical cores minus one."""

    if workers == "auto":
        return max(1, _physical_core_count() - 1)
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be 'auto' or an integer")
    if workers < 1:
        raise ValueError("workers must be 'auto' or greater than zero")
    return workers


def ordered_parallel_map(
    function: Callable[[_Item], _Result],
    items: Iterable[_Item],
    workers: int,
    on_submit: Callable[[_Item], None] | None = None,
) -> Iterator[tuple[_Item, _Result]]:
    """Run independent jobs concurrently and yield them in input order.

    At most ``workers`` jobs are submitted at once. This bounds retained match traces and keeps
    deterministic output ordering. Submission and result delivery happen on the caller thread.
    """

    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            if on_submit is not None:
                on_submit(item)
            yield item, function(item)
        return

    pending: deque[tuple[_Item, Future[_Result]]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="meeple-match",
    ) as executor:
        for _ in range(workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            if on_submit is not None:
                on_submit(item)
            pending.append((item, executor.submit(function, item)))

        while pending:
            item, future = pending.popleft()
            yield item, future.result()
            try:
                next_item = next(iterator)
            except StopIteration:
                continue
            if on_submit is not None:
                on_submit(next_item)
            pending.append((next_item, executor.submit(function, next_item)))


def _physical_core_count() -> int:
    available_cpus = _available_cpu_ids()
    topology_root = Path("/sys/devices/system/cpu")
    physical_cores: set[tuple[str, str]] = set()

    try:
        for cpu in available_cpus:
            topology = topology_root / f"cpu{cpu}" / "topology"
            package_id = (topology / "physical_package_id").read_text().strip()
            core_id = (topology / "core_id").read_text().strip()
            physical_cores.add((package_id, core_id))
    except (OSError, ValueError):
        physical_cores.clear()

    if physical_cores:
        return len(physical_cores)
    return max(1, len(available_cpus))


def _available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        return tuple(range(os.cpu_count() or 1))
    except OSError:
        return tuple(range(os.cpu_count() or 1))
