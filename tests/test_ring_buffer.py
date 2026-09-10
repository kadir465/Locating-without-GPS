from __future__ import annotations

from advanced_localization.ring_buffer import TimeIndexedRingBuffer


def test_ring_buffer_time_queries_handle_sorted_and_inserted_items() -> None:
    buffer = TimeIndexedRingBuffer[str](capacity=4)
    buffer.append(0.0, "a")
    buffer.append(0.3, "d")
    buffer.append(0.1, "b")
    buffer.append(0.2, "c")

    assert buffer.at_or_before(0.19) == (0.1, "b")
    assert buffer.at_nearest(0.24) == (0.2, "c")
    assert buffer.after(0.1) == [(0.2, "c"), (0.3, "d")]
    assert buffer.between(0.1, 0.2) == [(0.1, "b"), (0.2, "c")]


def test_ring_buffer_capacity_drops_oldest_by_time() -> None:
    buffer = TimeIndexedRingBuffer[str](capacity=3)
    buffer.append(0.0, "a")
    buffer.append(0.2, "c")
    buffer.append(0.1, "b")
    buffer.append(0.3, "d")

    assert len(buffer) == 3
    assert buffer.at_or_before(0.0) is None
    assert buffer.after(-1.0) == [(0.1, "b"), (0.2, "c"), (0.3, "d")]
