"""Порядок вызова агентов в ходу.

Порядок фиксирован (детерминизм):
    Лагерь → фокусные (кто говорит — первым) → пары → толпа → одиночки
    → преобразователь.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def build_call_order(
    *,
    focus: Iterable[str] = (),
    pairs: Sequence[tuple[str, str]] = (),
    singles: Iterable[str] = (),
    speaking: Sequence[str] = (),
    camp: str = "camp",
    crowd: str = "crowd",
    converter: str = "converter",
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def add(agent_id: str) -> None:
        if agent_id and agent_id not in seen:
            seen.add(agent_id)
            order.append(agent_id)

    add(camp)

    speak_rank = {name: i for i, name in enumerate(speaking)}
    for agent_id in sorted(
        focus,
        key=lambda a: (speak_rank.get(a, len(speak_rank)), a),
    ):
        add(agent_id)

    for left, right in pairs:
        first, second = (left, right) if speak_rank.get(left, 99) <= speak_rank.get(right, 99) else (right, left)
        add(first)
        add(second)

    add(crowd)

    for agent_id in sorted(singles):
        add(agent_id)

    add(converter)
    return order


def assign_priorities(order: Sequence[str], focus: Iterable[str]) -> dict[str, int]:
    """Приоритет для планировщика: фокусные входят в батч раньше остальных."""
    focal = set(focus)
    out: dict[str, int] = {}
    for index, agent_id in enumerate(order):
        if agent_id == "camp":
            out[agent_id] = 0
        elif agent_id == "converter":
            out[agent_id] = 90
        elif agent_id == "crowd":
            out[agent_id] = 30
        elif agent_id in focal:
            out[agent_id] = 10
        else:
            out[agent_id] = 40 + index % 5
    return out


def find_crossings(moves: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """Кого встретить повторно в том же ходу.

    moves: agent_id -> (откуда, куда). Встреча — общее ребро (в любую сторону)
    или обмен локациями.
    """
    agents = sorted(moves)
    found: set[tuple[str, str]] = set()
    for i, left in enumerate(agents):
        l_from, l_to = moves[left]
        for right in agents[i + 1 :]:
            r_from, r_to = moves[right]
            same_edge = {l_from, l_to} == {r_from, r_to}
            swap = l_from == r_to and r_from == l_to
            meet = l_to == r_to and l_to != l_from
            if same_edge or swap or meet:
                found.add((left, right))
    return sorted(found)
