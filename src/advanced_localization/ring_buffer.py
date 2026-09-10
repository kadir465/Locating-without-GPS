from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class TimeIndexedRingBuffer(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self._capacity = int(capacity)
        self._times: list[float] = []
        self._items: list[T] = []

    def append(self, timestamp_s: float, item: T) -> None:
        if self._times and timestamp_s < self._times[-1]:
            index = bisect_right(self._times, timestamp_s)
            self._times.insert(index, timestamp_s)
            self._items.insert(index, item)
        else:
            self._times.append(timestamp_s)
            self._items.append(item)

        overflow = len(self._times) - self._capacity
        if overflow > 0:
            del self._times[:overflow]
            del self._items[:overflow]

    def latest(self) -> Optional[Tuple[float, T]]:
        if not self._times:
            return None
        return self._times[-1], self._items[-1]

    def at_or_before(self, timestamp_s: float) -> Optional[Tuple[float, T]]:
        if not self._times:
            return None
        index = bisect_right(self._times, timestamp_s) - 1
        if index < 0:
            return None
        return self._times[index], self._items[index]

    def at_nearest(self, timestamp_s: float) -> Optional[Tuple[float, T]]:
        if not self._times:
            return None

        right_index = bisect_left(self._times, timestamp_s)
        if right_index <= 0:
            best_index = 0
        elif right_index >= len(self._times):
            best_index = len(self._times) - 1
        else:
            left_index = right_index - 1
            left_delta = abs(self._times[left_index] - timestamp_s)
            right_delta = abs(self._times[right_index] - timestamp_s)
            best_index = left_index if left_delta <= right_delta else right_index
        return self._times[best_index], self._items[best_index]

    def after(self, timestamp_s: float) -> List[Tuple[float, T]]:
        start = bisect_right(self._times, timestamp_s)
        return list(zip(self._times[start:], self._items[start:]))

    def between(self, start_s: float, end_s: float) -> List[Tuple[float, T]]:
        start = bisect_left(self._times, start_s)
        end = bisect_right(self._times, end_s)
        return list(zip(self._times[start:end], self._items[start:end]))

    def clear_after(self, timestamp_s: float) -> None:
        keep = bisect_right(self._times, timestamp_s)
        del self._times[keep:]
        del self._items[keep:]

    def clear(self) -> None:
        self._times.clear()
        self._items.clear()

    def __len__(self) -> int:
        return len(self._times)
