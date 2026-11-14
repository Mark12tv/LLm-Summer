from __future__ import annotations

from engine.turn.order import assign_priorities, build_call_order, find_crossings


def test_camp_first_and_converter_last() -> None:
    order = build_call_order(focus={"lena"}, pairs=[("ulyana", "alisa")], singles={"miku"})
    assert order[0] == "camp"
    assert order[-1] == "converter"


def test_speaker_goes_first_inside_pair() -> None:
    order = build_call_order(focus={"lena"}, pairs=[("ulyana", "alisa")], speaking=["alisa"])
    assert order.index("alisa") < order.index("ulyana")


def test_focus_speaker_first() -> None:
    order = build_call_order(
        focus={"lena", "slavya"}, pairs=[], speaking=["slavya"]
    )
    assert order[1] == "slavya"


def test_crowd_before_singles() -> None:
    order = build_call_order(focus={"lena"}, pairs=[], singles={"zhenya", "boys"})
    assert order.index("crowd") < order.index("zhenya")
    assert order.index("crowd") < order.index("boys")


def test_no_duplicates_when_agent_is_in_two_buckets() -> None:
    order = build_call_order(focus={"lena"}, pairs=[("lena", "ulyana")], singles={"lena"})
    assert order.count("lena") == 1


def test_priority_focus_is_lower_number() -> None:
    order = build_call_order(focus={"lena"}, pairs=[("ulyana", "alisa")], singles={"miku"})
    priorities = assign_priorities(order, focus={"lena"})
    assert priorities["camp"] == 0
    assert priorities["converter"] == 90
    assert priorities["lena"] == 10
    assert priorities["lena"] < priorities["miku"]


def test_crossings_detect_same_edge_and_swap() -> None:
    moves = {
        "lena": ("domiki", "ploshchad"),
        "ulyana": ("ploshchad", "domiki"),   # обмен
        "alisa": ("stolovaya", "plyazh"),    # другое ребро
        "slavya": ("kostrovaya", "plyazh"),  # та же цель
    }
    assert ("lena", "ulyana") in find_crossings(moves)
    assert ("alisa", "slavya") in find_crossings(moves)
    assert ("lena", "alisa") not in find_crossings(moves)


def test_crossings_empty_without_moves() -> None:
    assert find_crossings({}) == []
