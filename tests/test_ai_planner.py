"""Per-empire strategic planner in AI mode (AILoopController + StrategicPlanner)."""

from __future__ import annotations

from engine.config import PlannerConfig
from engine.decision_engine import build_prompt
from engine.game_loop import AILoopController
from engine.llm_provider import LLMProvider, LLMResponse, StubProvider
from engine.strategic_planner import StrategicContext

_PLAN = (
    "THREAT_LEVEL: high\n"
    "ECONOMY_HEALTH: fragile\n"
    "BOTTLENECK: alloys\n"
    "PRIORITY_1: BUILD_FLEET\n"
    "PRIORITY_2: IMPROVE_ECONOMY\n"
    "PRIORITY_3: DEFEND\n"
    "FOCUS: war preparation\n"
    "ARC: Rebuild the fleet before the neighbour strikes."
)


class _PlannerStub(LLMProvider):
    """Counts calls and answers in the planner format."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=_PLAN, model="planner-stub")

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "planner-stub"


def _state(country_id: int, year: int) -> dict:
    return {
        "version": "4.4.6",
        "year": year,
        "month": 1,
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


def _controller(planner: LLMProvider | None, interval: int = 5) -> AILoopController:
    return AILoopController(
        provider=StubProvider(),
        planner_config=PlannerConfig(enabled=True, provider="online", interval_years=interval),
        planner_provider=planner,
        fast_decisions=False,
    )


def test_planner_disabled_by_default() -> None:
    controller = AILoopController(provider=StubProvider(), fast_decisions=False)
    controller.process_states([_state(1, 2210)])
    assert controller.planners == {}


def test_one_planner_per_empire() -> None:
    planner = _PlannerStub()
    controller = _controller(planner)
    controller.process_states([_state(1, 2210), _state(2, 2210), _state(3, 2210)])

    assert sorted(controller.planners) == [1, 2, 3]
    assert planner.calls == 3
    ctx = controller.planners[1].context
    assert isinstance(ctx, StrategicContext)
    assert ctx.source == "llm"
    assert ctx.priorities == ["BUILD_FLEET", "IMPROVE_ECONOMY", "DEFEND"]
    assert ctx.recommended_focus == "war preparation"


def test_replans_every_interval_years() -> None:
    planner = _PlannerStub()
    controller = _controller(planner, interval=5)

    # 2210: first plan; 2212, 2214: cached; 2215: 5 years later -> re-plan
    for year in (2210, 2212, 2214):
        controller.process_states([_state(1, year)])
    assert planner.calls == 1

    controller.process_states([_state(1, 2215)])
    assert planner.calls == 2
    assert controller.planners[1].context.year_generated == 2215


def test_replans_on_phase_transition() -> None:
    planner = _PlannerStub()
    controller = _controller(planner, interval=50)

    controller.process_states([_state(1, 2228)])
    controller.process_states([_state(1, 2229)])
    assert planner.calls == 1
    # Early -> mid game boundary forces a new plan regardless of interval
    controller.process_states([_state(1, 2240)])
    assert planner.calls == 2


def test_planner_runs_even_when_fast_path_decides() -> None:
    planner = _PlannerStub()
    controller = AILoopController(
        provider=StubProvider(),
        planner_config=PlannerConfig(enabled=True, provider="online", interval_years=5),
        planner_provider=planner,
        fast_decisions=True,
    )
    # Year 2210 with a tiny fleet is handled by the code-only fast path...
    controller.process_states([_state(1, 2210)])
    assert controller.stats.decisions_made == 1
    # ...but the strategic plan is still produced on schedule.
    assert planner.calls == 1


def test_parallel_empires_get_separate_planners() -> None:
    planner = _PlannerStub()
    controller = AILoopController(
        provider=StubProvider(),
        planner_config=PlannerConfig(enabled=True, provider="online", interval_years=5),
        planner_provider=planner,
        fast_decisions=False,
        parallel_empires=True,
    )
    states = [_state(cid, 2210) for cid in range(1, 5)]
    controller.process_states(states)
    assert sorted(controller.planners) == [1, 2, 3, 4]
    assert planner.calls == 4


def test_code_only_planner_when_no_provider() -> None:
    controller = AILoopController(
        provider=StubProvider(),
        planner_config=PlannerConfig(enabled=True, provider="none", interval_years=5),
        planner_provider=None,
        fast_decisions=False,
    )
    controller.process_states([_state(1, 2210)])
    ctx = controller.planners[1].context
    assert ctx is not None
    assert ctx.source == "code"


def test_strategic_context_injected_into_prompt() -> None:
    ctx = StrategicContext(
        phase="early", year_generated=2210, source="llm",
        priorities=["BUILD_FLEET"], recommended_focus="war preparation",
        arc_summary="Rebuild the fleet.",
    )
    state = _state(1, 2212)
    with_ctx = build_prompt({}, {}, state, None, strategic_context=ctx)
    without = build_prompt({}, {}, state, None)
    assert ctx.to_prompt_block() in with_ctx
    assert ctx.to_prompt_block() not in without
