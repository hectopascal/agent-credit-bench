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
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from agent_credit_bench._numerics import finite_float
from agent_credit_bench._validation import (
    validate_positive_integer,
    validated_transitions,
)
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

    A complete exact reply takes precedence. Otherwise the last nonoverlapping
    mention wins, with longer labels taking precedence over contained labels.
    Alphanumeric boundaries keep STOP from matching inside STOPPING while
    allowing punctuation-bearing names such as C++. Returns None on failure.
    """
    stripped = text.strip()
    for action in actions:
        if stripped == str(action):
            return action
    for action in actions:
        if stripped.casefold() == str(action).casefold():
            return action
    mentions = []
    for action in actions:
        pattern = rf"(?<!\w){re.escape(str(action))}(?!\w)"
        mentions.extend(
            (match.start(), match.end(), action)
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
    mentions.sort(key=lambda mention: (mention[0], -mention[1]))
    best: Action | None = None
    end = -1
    for start, stop, action in mentions:
        if start >= end:
            best, end = action, stop
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
    source_actions = actions if actions is not None else mdp.actions(timestep, state)
    legal_actions = list(source_actions)
    if not legal_actions:
        raise ValueError(f"no legal actions at (t={timestep}, state={state!r})")
    values = _minimum_action_returns(mdp, timestep, state, worst_outcome=False)
    return min(legal_actions, key=values.__getitem__)


def minimum_episode_return(mdp: FiniteHorizonMDP) -> float:
    """Lowest supported complete return, over actions AND stochastic outcomes.

    Used as the truncation reward: abandoning an episode must not improve on
    any legal completion, including in environments with negative rewards.
    Zero-probability edges are excluded; early termination ends accumulation.
    """
    values = _minimum_action_returns(mdp, 0, mdp.initial_state, worst_outcome=True)
    return finite_float(min(values.values()))


def _minimum_action_returns(
    mdp: FiniteHorizonMDP, timestep: int, state: State, *, worst_outcome: bool
) -> dict[Action, Fraction]:
    validate_positive_integer(mdp.horizon, "mdp.horizon")
    if not 0 <= timestep < mdp.horizon:
        raise ValueError("timestep must be within the environment horizon")
    future_values: dict[State, Fraction] = {}
    for at in range(mdp.horizon - 1, timestep - 1, -1):
        state_values = {}
        for at_state in ([state] if at == timestep else mdp.states_at(at)):
            action_values = {}
            for action in mdp.actions(at, at_state):
                outcomes = []
                probability_sum = Fraction(0)
                for transition in validated_transitions(
                    mdp.transitions(at, at_state, action), at, at_state, action
                ):
                    if transition.probability == 0.0:
                        continue
                    terminal = transition.terminated or at + 1 == mdp.horizon
                    future = (
                        Fraction(0)
                        if terminal else future_values[transition.next_state]
                    )
                    probability = Fraction(transition.probability)
                    probability_sum += probability
                    value = Fraction(transition.reward) + future
                    outcomes.append(
                        value if worst_outcome else probability * value
                    )
                action_values[action] = (
                    min(outcomes) if worst_outcome else sum(outcomes) / probability_sum
                )
            if not action_values:
                raise ValueError(f"no legal actions at (t={at}, state={at_state!r})")
            state_values[at_state] = min(action_values.values())
        future_values = state_values
    return action_values


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

        transitions = validated_transitions(
            self.mdp.transitions(self.timestep, self.state, action),
            self.timestep, self.state, action,
        )
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
        return self.trajectory.total_return


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


def _record_env_params(
    mdp: FiniteHorizonMDP, env: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    supplied = metadata.pop("env_params", None)
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("env_params must be a parameter object")
    if type(mdp) in (RecoveryEnv, DelayedEffectEnv, VariableHorizonEnv):
        if ENVS.get(env) is not type(mdp):
            raise ValueError(f"environment name {env!r} does not match the session MDP")
        params = asdict(mdp)
        if supplied is not None:
            # Accept older callers' partial configurations when their defaults
            # reconstruct the actual MDP; always write the complete configuration.
            declared = asdict(make_env(env, **supplied))
            if json.loads(json.dumps(declared, allow_nan=False)) != json.loads(
                json.dumps(params, allow_nan=False)
            ):
                raise ValueError("env_params do not match the session MDP")
    else:
        if supplied is None:
            raise ValueError("custom environments require explicit env_params metadata")
        params = dict(supplied)
    # Canonical JSON containers also turn tuple-valued rewards into lists.
    return json.loads(json.dumps(params, allow_nan=False))


def episode_record(
    session: BridgeSession, env: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One finished episode as a JSON-serializable dict (one JSONL line).

    Built-in environments include their full actual constructor parameters.
    Custom MDPs require ``extra={"env_params": {...}}``. Extra metadata cannot
    replace recorded fields; supplied built-in parameters must match the MDP.
    """
    if not session.done:
        raise ValueError("episode is not finished")
    metadata = dict(extra or {})
    params = _record_env_params(session.mdp, env, metadata)
    reserved = {"env", "parse_failure_policy", "total_return", "turns"}.intersection(
        metadata
    )
    if reserved:
        raise ValueError(
            f"extra metadata cannot replace recorded fields: {sorted(reserved)}"
        )
    return {
        "env": env,
        "env_params": params,
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
        **metadata,
    }


def _validate_logged_trajectory(mdp: FiniteHorizonMDP, trajectory: Trajectory) -> None:
    validate_positive_integer(mdp.horizon, "mdp.horizon")
    if not trajectory.steps:
        raise ValueError("episode has no turns")
    expected_state = mdp.initial_state
    for index, step in enumerate(trajectory.steps):
        prefix = f"turn {index}: "
        if index >= mdp.horizon:
            raise ValueError(prefix + "exceeds the environment horizon")
        if type(step.timestep) is not int or step.timestep != index:
            raise ValueError(prefix + f"timestep must be {index}")
        if not isinstance(step.state, str) or step.state != expected_state:
            raise ValueError(prefix + f"state must be {expected_state!r}")
        if not isinstance(step.action, str) or step.action not in mdp.actions(
            index, step.state
        ):
            raise ValueError(prefix + f"illegal action {step.action!r}")
        if (
            isinstance(step.reward, bool)
            or not isinstance(step.reward, (int, float))
            or not math.isfinite(step.reward)
        ):
            raise ValueError(prefix + "reward must be finite")
        if type(step.terminated) is not bool:
            raise ValueError(prefix + "terminated must be a boolean")
        transitions = validated_transitions(
            mdp.transitions(index, step.state, step.action),
            index, step.state, step.action,
        )
        if not any(
            tr.probability > 0.0
            and tr.next_state == step.next_state
            and tr.reward == step.reward
            and tr.terminated == step.terminated
            for tr in transitions
        ):
            raise ValueError(
                prefix + "transition has no positive-probability MDP support"
            )
        if step.terminated and index != len(trajectory.steps) - 1:
            raise ValueError(prefix + "episode continues after termination")
        expected_state = step.next_state
    if not trajectory.steps[-1].terminated and len(trajectory.steps) < mdp.horizon:
        raise ValueError(f"turn {len(trajectory.steps) - 1}: episode is incomplete")


def trajectory_from_record(
    record: Mapping[str, Any], *, mdp: FiniteHorizonMDP | None = None
) -> Trajectory:
    """Deserialize a log, optionally validating it against its declared MDP.

    With mdp supplied, rewards, transitions, chronology, completion, and the
    recorded total must match stable summation or the legacy left-to-right sum.
    Individual rewards match MDP support exactly. Horizon exhaustion counts as
    completion even when the
    last transition is not marked terminated. Callers scoring logs should
    always supply mdp; omission retains the parsing-only compatibility path.
    """
    turns = record.get("turns")
    if not isinstance(turns, (list, tuple)):
        raise ValueError("turns must be a list")
    steps = []
    for index, turn in enumerate(turns):
        try:
            step = Step(
                timestep=turn["timestep"],
                state=turn["state"],
                action=turn["action"],
                reward=turn["reward"],
                next_state=turn["next_state"],
                terminated=turn["terminated"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"turn {index}: invalid step fields: {error}") from error
        steps.append(step)
    trajectory = Trajectory(tuple(steps))
    if mdp is not None:
        _validate_logged_trajectory(mdp, trajectory)
        total = record.get("total_return")
        # Python 3.12 changed built-in sum's float algorithm. Accept the old
        # explicitly reconstructed sum, plus a two-ulp allowance around fsum
        # for compensated producers; never use a reward-scale relative tolerance.
        stable_total = trajectory.total_return
        legacy_total = 0.0
        for step in trajectory.steps:
            legacy_total += step.reward
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(total)
            or not (
                total == legacy_total
                or math.isclose(
                    total, stable_total, rel_tol=0.0,
                    abs_tol=2 * math.ulp(stable_total),
                )
            )
        ):
            raise ValueError("total_return must be finite and equal the sum of rewards")
    return trajectory


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
