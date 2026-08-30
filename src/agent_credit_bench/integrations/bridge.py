"""Verbalized-MDP bridge: play suite environments as multi-turn chat games.

Turns any FiniteHorizonMDP into a text game an LLM can play: states render
to chat messages, model replies parse back to actions, and realized
transitions are recorded as ordinary suite Steps. Because the underlying MDP
stays known, logged actions can be projected onto a tabular Markov policy
conditioned on ``(timestep, state)``. Dynamic programming is exact for that
fitted projection, so every suite metric can be applied to the logged sample.
The projection has finite-sample error and deliberately collapses any extra
conversation-history dependence in the LLM policy; it is not an exact oracle
for the history-conditioned model itself.

Unparseable replies default to a deterministic, conservative mapping: choose
the legal action with the minimum expected episode return when all later
choices also minimize return. Callers can instead configure parse failures to
raise. Either way, the parsed-rate distinguishes model-selected actions from
environment-selected fallbacks.

No framework imports here; the verl recipe (recipes/verl_bridge/) builds on
these pieces, and any text-in/text-out callable works via play_episode.

States and actions must be strings (all suite environments comply) so
episodes serialize to JSON losslessly.
"""

import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_credit_bench.envs import DelayedEffectEnv, RecoveryEnv, VariableHorizonEnv
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.types import Action, State, Step, Trajectory

ENVS: dict[str, Callable[..., FiniteHorizonMDP]] = {
    "recovery": RecoveryEnv,
    "delayed_effect": DelayedEffectEnv,
    "variable_horizon": VariableHorizonEnv,
}

ParseFailurePolicy = Literal["minimum_return", "raise"]


def make_env(name: str, **params: Any) -> FiniteHorizonMDP:
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; available: {sorted(ENVS)}")
    return ENVS[name](**params)


def render_system_prompt(mdp: FiniteHorizonMDP) -> str:
    return (
        "You are playing a turn-based decision game. Each turn shows the "
        "current situation and the legal actions. Rewards may be delayed "
        "until the end of the game; maximize your total reward over at most "
        f"{mdp.horizon} turns. Reply with exactly one legal action name. "
        "You may think first, but end your reply with the chosen action."
    )


def render_state(
    timestep: int, state: State, actions: Sequence[Action]
) -> str:
    listed = ", ".join(str(action) for action in actions)
    return (
        f"Turn {timestep}. Current state: {state}. "
        f"Legal actions: {listed}. Which action do you take?"
    )


def parse_action(text: str, actions: Sequence[Action]) -> Action | None:
    """Match the last legal action named in the reply, case-insensitively.

    Last occurrence wins so models can reason before answering. Word-boundary
    matching keeps STOP from matching inside STOPPING. Returns None when no
    legal action appears.
    """
    best: Action | None = None
    best_position = -1
    for action in actions:
        pattern = rf"\b{re.escape(str(action))}\b"
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches and matches[-1].start() > best_position:
            best = action
            best_position = matches[-1].start()
    return best


def minimum_return_action(
    mdp: FiniteHorizonMDP,
    timestep: int,
    state: State,
    actions: Sequence[Action] | None = None,
) -> Action:
    """Choose a deterministic, pessimistic fallback for an unparseable reply.

    Backward induction minimizes expected undiscounted episode return, assuming
    future parse failures also take minimum-return actions. The first action in
    the MDP's declared order breaks exact value ties. This makes the fallback
    reproducible without silently rewarding malformed output merely because a
    favorable action happened to be listed first.
    """
    value_cache: dict[tuple[int, State], float] = {}

    def action_value(at: int, at_state: State, action: Action) -> float:
        total = 0.0
        for transition in mdp.transitions(at, at_state, action):
            terminal = transition.terminated or at + 1 >= mdp.horizon
            future = 0.0 if terminal else state_value(at + 1, transition.next_state)
            total += transition.probability * (transition.reward + future)
        return total

    def state_value(at: int, at_state: State) -> float:
        key = (at, at_state)
        if key not in value_cache:
            legal = list(mdp.actions(at, at_state))
            if not legal:
                raise ValueError(
                    f"no legal actions at (t={at}, state={at_state!r})"
                )
            value_cache[key] = min(
                action_value(at, at_state, action) for action in legal
            )
        return value_cache[key]

    source_actions = actions if actions is not None else mdp.actions(timestep, state)
    legal_actions = list(source_actions)
    if not legal_actions:
        raise ValueError(f"no legal actions at (t={timestep}, state={state!r})")
    return min(
        legal_actions,
        key=lambda action: action_value(timestep, state, action),
    )


@dataclass(frozen=True)
class BridgeTurn:
    observation: str
    reply: str
    parsed: bool
    step: Step


@dataclass
class BridgeSession:
    """One episode of a verbalized MDP, driven by text replies.

    Transition sampling mirrors sampling.sample_trajectories: seeded
    random.Random, rng.choices over transition probabilities, stop on a
    terminated transition or at the horizon. seed=None draws fresh entropy —
    the right default for training rollouts, where a shared per-prompt seed
    would correlate outcomes within a GRPO group.
    """

    mdp: FiniteHorizonMDP
    seed: int | None = None
    parse_failure_policy: ParseFailurePolicy = "minimum_return"
    _rng: random.Random = field(init=False, repr=False)
    _turns: list[BridgeTurn] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.parse_failure_policy not in ("minimum_return", "raise"):
            raise ValueError(
                "parse_failure_policy must be 'minimum_return' or 'raise'"
            )
        self._rng = random.Random(self.seed)
        self.state: State = self.mdp.initial_state
        self.timestep = 0
        self.done = False

    @property
    def turns(self) -> tuple[BridgeTurn, ...]:
        return tuple(self._turns)

    @property
    def available_actions(self) -> list[Action]:
        return list(self.mdp.actions(self.timestep, self.state))

    def observe(self) -> str:
        if self.done:
            raise RuntimeError("episode is over")
        return render_state(self.timestep, self.state, self.available_actions)

    def act(self, reply: str) -> Step:
        if self.done:
            raise RuntimeError("episode is over")
        actions = self.available_actions
        parsed = parse_action(reply, actions)
        if parsed is not None:
            action = parsed
        elif self.parse_failure_policy == "raise":
            raise ValueError(
                f"reply did not name a legal action from {actions!r}: {reply!r}"
            )
        else:
            action = minimum_return_action(
                self.mdp, self.timestep, self.state, actions
            )

        transitions = list(self.mdp.transitions(self.timestep, self.state, action))
        transition = self._rng.choices(
            transitions, weights=[tr.probability for tr in transitions]
        )[0]
        step = Step(
            timestep=self.timestep,
            state=self.state,
            action=action,
            reward=transition.reward,
            next_state=transition.next_state,
            terminated=transition.terminated,
        )
        self._turns.append(
            BridgeTurn(
                observation=self.observe(),
                reply=reply,
                parsed=parsed is not None,
                step=step,
            )
        )
        self.timestep += 1
        self.state = transition.next_state
        self.done = transition.terminated or self.timestep >= self.mdp.horizon
        return step

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(tuple(turn.step for turn in self._turns))

    @property
    def total_return(self) -> float:
        return sum(turn.step.reward for turn in self._turns)


def play_episode(
    mdp: FiniteHorizonMDP,
    respond: Callable[[list[dict[str, str]]], str],
    seed: int | None = None,
    parse_failure_policy: ParseFailurePolicy = "minimum_return",
) -> BridgeSession:
    """Drive one full episode with any messages -> reply-text callable."""
    session = BridgeSession(
        mdp, seed=seed, parse_failure_policy=parse_failure_policy
    )
    messages = [{"role": "system", "content": render_system_prompt(mdp)}]
    while not session.done:
        messages.append({"role": "user", "content": session.observe()})
        reply = respond(messages)
        messages.append({"role": "assistant", "content": reply})
        session.act(reply)
    return session


def episode_record(
    session: BridgeSession, env: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One finished episode as a JSON-serializable dict (one JSONL line)."""
    if not session.done:
        raise ValueError("episode is not finished")
    return {
        "env": env,
        "parse_failure_policy": session.parse_failure_policy,
        "total_return": session.total_return,
        "turns": [
            {
                "timestep": turn.step.timestep,
                "state": turn.step.state,
                "action": turn.step.action,
                "reward": turn.step.reward,
                "next_state": turn.step.next_state,
                "terminated": turn.step.terminated,
                "reply": turn.reply,
                "parsed": turn.parsed,
            }
            for turn in session.turns
        ],
        **(dict(extra) if extra else {}),
    }


def trajectory_from_record(record: Mapping[str, Any]) -> Trajectory:
    return Trajectory(
        tuple(
            Step(
                timestep=turn["timestep"],
                state=turn["state"],
                action=turn["action"],
                reward=turn["reward"],
                next_state=turn["next_state"],
                terminated=turn["terminated"],
            )
            for turn in record["turns"]
        )
    )


def load_episodes(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class EmpiricalTabularPolicy:
    """Finite-sample Markov projection from logged action counts.

    The table estimates ``pi_hat(action | timestep, state)``. It aggregates
    trajectories that reach the same key even if their prompts or earlier
    conversation histories differ. Exact MDP solutions using this object are
    therefore exact for the fitted Markov projection, not for a potentially
    history-conditioned LLM policy. The action frequencies also retain normal
    finite-sample estimation error.

    Unvisited states fall back to uniform: they carry no visitation mass in
    the logged data, but the exact solver still needs a distribution there.
    """

    counts: Mapping[tuple[int, State], Mapping[Action, int]]
    sample_size: int | None = None

    @classmethod
    def from_trajectories(
        cls, trajectories: Iterable[Trajectory]
    ) -> "EmpiricalTabularPolicy":
        counts: dict[tuple[int, State], dict[Action, int]] = {}
        sample_size = 0
        for trajectory in trajectories:
            sample_size += 1
            for step in trajectory.steps:
                key = (step.timestep, step.state)
                at_key = counts.setdefault(key, {})
                at_key[step.action] = at_key.get(step.action, 0) + 1
        return cls(counts=counts, sample_size=sample_size)

    def visit_count(self, timestep: int, state: State) -> int:
        return sum(self.counts.get((timestep, state), {}).values())

    def action_probabilities(
        self,
        timestep: int,
        state: State,
        actions: Sequence[Action],
    ) -> Mapping[Action, float]:
        at_key = self.counts.get((timestep, state))
        if not at_key:
            probability = 1.0 / len(actions)
            return {action: probability for action in actions}
        total = sum(at_key.values())
        return {action: at_key.get(action, 0) / total for action in actions}
