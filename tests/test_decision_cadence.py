"""AI-mode decision cadence and prompt-size levers (decision_interval_months, compact JSON)."""

from __future__ import annotations

import pytest

from engine.decision_engine import build_prompt, set_compact_json
from engine.game_loop import AILoopController
from engine.llm_provider import StubProvider


def _state(country_id: int, year: int, month: int, event: str | None = None) -> dict:
    state = {
        "version": "4.4.6",
        "year": year,
        "month": month,
        "country_id": country_id,
        "empire": {
            "name": f"Empire {country_id}",
            "ethics": ["Militarist"],
            "civics": [],
            "origin": "Prosperous Unification",
            "government": "Oligarchy",
        },
        "economy": {"energy": 100, "minerals": 200, "alloys": 30},
        "colonies": [f"Planet_{country_id}"],
        "known_empires": [],
        "fleets": [],
        "wars": [],
    }
    if event:
        state["event"] = event
    return state


def _controller(interval: int) -> AILoopController:
    return AILoopController(
        provider=StubProvider(),
        fast_decisions=False,
        decision_interval_months=interval,
    )


# --------------------------------------------------------------------- #
# Decision interval
# --------------------------------------------------------------------- #

def test_interval_zero_decides_every_save() -> None:
    controller = _controller(0)
    for month in (1, 2, 3):
        assert controller.process_states([_state(1, 2300, month)])[0] is not None
    assert controller.stats.decisions_made == 3
    assert controller.stats.decisions_skipped == 0


def test_interval_skips_until_due() -> None:
    controller = _controller(3)
    # Month 1: first decision.  Months 2, 3: skipped.  Month 4: due again.
    assert controller.process_states([_state(1, 2300, 1)])[0] is not None
    assert controller.process_states([_state(1, 2300, 2)])[0] is None
    assert controller.process_states([_state(1, 2300, 3)])[0] is None
    assert controller.process_states([_state(1, 2300, 4)])[0] is not None
    assert controller.stats.decisions_made == 2
    assert controller.stats.decisions_skipped == 2


def test_interval_spans_year_boundary() -> None:
    controller = _controller(3)
    assert controller.process_states([_state(1, 2300, 11)])[0] is not None
    assert controller.process_states([_state(1, 2300, 12)])[0] is None
    assert controller.process_states([_state(1, 2301, 1)])[0] is None
    assert controller.process_states([_state(1, 2301, 2)])[0] is not None


def test_event_bypasses_interval() -> None:
    controller = _controller(12)
    assert controller.process_states([_state(1, 2300, 1)])[0] is not None
    assert controller.process_states([_state(1, 2300, 2)])[0] is None
    assert controller.process_states([_state(1, 2300, 3, event="WAR_STARTED")])[0] is not None
    assert controller.stats.decisions_made == 2


def test_interval_is_per_empire() -> None:
    controller = _controller(3)
    controller.process_states([_state(1, 2300, 1)])
    # Empire 2 has never been decided: due immediately even though empire 1 is not
    results = controller.process_states([_state(1, 2300, 2), _state(2, 2300, 2)])
    assert results[0] is None
    assert results[1] is not None


def test_planner_still_runs_on_skipped_ticks() -> None:
    from engine.config import PlannerConfig

    controller = AILoopController(
        provider=StubProvider(),
        fast_decisions=False,
        decision_interval_months=6,
        planner_config=PlannerConfig(enabled=True, provider="none", interval_years=1),
        planner_provider=None,
    )
    controller.process_states([_state(1, 2300, 1)])
    assert controller.planners[1].context.year_generated == 2300
    # Next year, first month: decision skipped (interval 6 not reached from 2300.1? it is 12 months)
    # Use a tick that is skipped for decisions but due for the planner.
    controller.process_states([_state(1, 2300, 3)])
    assert controller.stats.decisions_skipped == 1
    # Force planner cadence: interval 1 year -> 2301.1 must re-plan even if decision were skipped
    controller.process_states([_state(1, 2301, 1)])
    assert controller.planners[1].context.year_generated == 2301


# --------------------------------------------------------------------- #
# Compact JSON + short reason
# --------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _restore_compact_json():
    yield
    set_compact_json(True)


def _state_block(prompt: str) -> str:
    start = prompt.index("CURRENT STATE:") + len("CURRENT STATE:")
    end = prompt.index("Respond in EXACTLY")
    return prompt[start:end].strip()


def test_state_json_is_minified_by_default() -> None:
    set_compact_json(True)
    block = _state_block(build_prompt({}, {}, _state(1, 2300, 1), None))
    assert "\n" not in block
    assert '"year":2300' in block


def test_state_json_indented_when_compact_disabled() -> None:
    set_compact_json(False)
    block = _state_block(build_prompt({}, {}, _state(1, 2300, 1), None))
    assert '"year": 2300' in block
    assert "\n  " in block


def test_prompt_asks_for_one_sentence_reason() -> None:
    prompt = build_prompt({}, {}, _state(1, 2300, 1), None)
    assert "REASON: <ONE sentence" in prompt
