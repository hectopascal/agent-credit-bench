# AgentCreditBench — Project Plan

> **Historical design document.** This plan records the original `v0.1`
> sequence, hypotheses, and stopping conditions; the repository has since been
> implemented and extended beyond that scope. Its unchecked boxes,
> future-tense milestones, "expected qualitative results," M0 label, and
> immediate-next-action section are not current status or empirical evidence.
> Use the tests, committed result CSVs, and README validation notes for current
> behavior and supported claims.

## Project status

- **Repository:** `agent-credit-bench`
- **Python package:** `agent_credit_bench`
- **Phase at time of drafting:** M0 — exact-value oracle
- **Original target release:** `v0.1.0`

---

## 1. Project thesis

**AgentCreditBench is a CPU-first conformance-test suite for turn-level credit estimators in agentic reinforcement learning.**

It evaluates estimators on tiny finite-horizon MDPs where the exact on-policy advantage of every sampled action can be calculated using backward dynamic programming.

The benchmark asks:

> When an estimator assigns credit to each action in a trajectory, how closely does that credit match the action’s exact policy advantage?

The repo is intended as a **unit-test suite**, not as a realistic agent-training benchmark.

### Positioning

The niche this fills, and what it deliberately does not duplicate:

- **bsuite-style diagnostics** (umbrella chain, memory chain) use structurally similar environments but score *agents* through learning curves. This project scores *estimators* against exact advantages; no training loop is involved.
- **Method papers** (GRPO variants, GiGPO, Turn-PPO, TRACE) validate credit assignment indirectly through end-task success on agent benchmarks, or against approximate Monte-Carlo ground truth from extra rollouts. This project provides exact ground truth on small problems instead.
- The result is a conformance suite that method authors and RL-infrastructure implementers can run in seconds on CPU to check that an estimator does what its math claims.

The README must state this positioning explicitly (see M6) so the suite is not mistaken for either an agent benchmark or a training framework.

---

## 2. Research question

How do common trajectory-level and turn-level credit estimators behave when:

1. an important action has a delayed effect;
2. most actions are irrelevant distractors;
3. an early mistake is repaired by a later action;
4. actions affect episode length and termination?

The benchmark should expose recognizable failure modes such as:

- smearing terminal reward across irrelevant actions;
- rewarding a bad action merely because the trajectory eventually succeeded;
- failing to distinguish recovery from the mistake being recovered from;
- length-dependent bias under variable-horizon trajectories;
- high estimator variance under sparse or stochastic rewards.

A second, equally important question:

> When an estimator's credit values deviate from exact advantage, does the resulting policy gradient still point in the right direction, and at what variance cost?

Value error and gradient error are different failure axes. An estimator can be far from advantage as a value estimate yet induce an unbiased gradient (raw-return REINFORCE is the canonical example), or close in value yet systematically biased in direction. The benchmark must measure both axes, or it will misgrade valid-but-uncentered estimators as broken.

---

## 3. Ground-truth definition

For `v0.1`, “credit” means the exact on-policy advantage:

\[
A_t^\pi(s,a)=Q_t^\pi(s,a)-V_t^\pi(s)
\]

with discount factor:

\[
\gamma=1
\]

For a finite-horizon MDP:

\[
Q_t^\pi(s,a)
=
\sum_{s',r}
P(s',r\mid s,a)
\left[
r +
\mathbb{1}_{\text{not terminal}}
V_{t+1}^\pi(s')
\right]
\]

and:

\[
V_t^\pi(s)
=
\sum_a
\pi(a\mid s)
Q_t^\pi(s,a)
\]

The values are calculated exactly by backward induction.

### Scope of this definition

This project benchmarks **policy-gradient-compatible action credit**.

It does not claim that advantage is the only meaningful definition of credit. In particular, `v0.1` does not attempt to benchmark:

- semantic responsibility;
- token attribution;
- Shapley values;
- general causal attribution;
- counterfactual explanations;
- human judgments of which step “caused” an outcome.

### The baseline equivalence class

For policy-gradient training, exact advantage is one member of an equivalence class. Any credit of the form

\[
\hat A_t(s,a) = A_t^\pi(s,a) + b_t(s)
\]

with \(b_t(s)\) independent of the sampled action produces the same expected policy gradient. Raw return is the extreme example: terrible as a value estimate, unbiased as a gradient weight.

The metrics therefore come in two families, and every estimator is scored on both:

1. **Identification metrics** — does the credit match exact advantage as a value? (RMSE, sign accuracy, leakage.) These matter when credit is consumed directly: advantage-based filtering, data selection, step-level reward shaping.
2. **Gradient metrics** — does the credit induce the correct policy-gradient direction, and with how much variance? (centered RMSE, gradient alignment; see §10.6–10.7.) These matter for training.

---

## 4. `v0.1` release criteria

A `v0.1.0` release is complete when the repository contains:

- [ ] an exact finite-horizon value and advantage solver;
- [ ] a trajectory sampler;
- [ ] three diagnostic environments;
- [ ] four baseline credit estimators;
- [ ] a reusable benchmark runner;
- [ ] core estimator metrics, covering both identification (value) and gradient-validity families, including baseline-shift-invariant error and policy-gradient alignment;
- [ ] at least three reproducible diagnostic figures;
- [ ] unit tests for the oracle, environments, estimators, and metrics;
- [ ] a README with installation, motivation, examples, and limitations;
- [ ] continuous integration running tests and linting;
- [ ] no GPU or external model dependency.

---

## 5. Explicit non-goals

The following are out of scope for `v0.1`:

- LLM inference;
- agent frameworks;
- PPO or GRPO training;
- transformer implementations;
- learned reward models;
- learned critics;
- token-level credit assignment;
- partially observable environments;
- Gymnasium compatibility;
- distributed execution;
- experiment tracking platforms;
- configuration frameworks;
- integration with `verl`, `vLLM`, TRL, or other training systems;
- attempts to prove that one estimator is universally best;
- a research paper.

No LLM integration should be added before `v0.1.0` exists.

---

## 6. Core abstractions

### 6.1 MDP interface

Every environment should expose a small finite-horizon MDP.

```python
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

State: TypeAlias = Hashable
Action: TypeAlias = Hashable


@dataclass(frozen=True)
class Transition:
    next_state: State
    reward: float
    probability: float
    terminated: bool


class FiniteHorizonMDP(Protocol):
    horizon: int
    initial_state: State

    def states_at(self, timestep: int) -> Sequence[State]:
        """Return every state reachable at this timestep."""

    def actions(
        self,
        timestep: int,
        state: State,
    ) -> Sequence[Action]:
        """Return available actions."""

    def transitions(
        self,
        timestep: int,
        state: State,
        action: Action,
    ) -> Sequence[Transition]:
        """Return the transition distribution."""
```

### 6.2 Policy interface

```python
class Policy(Protocol):
    def action_probabilities(
        self,
        timestep: int,
        state: State,
        actions: Sequence[Action],
    ) -> Mapping[Action, float]:
        """Return a valid probability distribution over actions."""
```

Policies are fixed during evaluation. The benchmark does not train them.

### 6.3 Trajectory representation

```python
@dataclass(frozen=True)
class Step:
    timestep: int
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[Step, ...]

    @property
    def total_return(self) -> float:
        return sum(step.reward for step in self.steps)
```

### 6.4 Oracle output

```python
@dataclass(frozen=True)
class ExactValues:
    state_values: dict[tuple[int, State], float]
    action_values: dict[tuple[int, State, Action], float]
    advantages: dict[tuple[int, State, Action], float]
```

### 6.5 Estimator interface

An estimator may need access to a batch of related trajectories. Some estimators may also use the environment or fixed policy.

```python
@dataclass(frozen=True)
class EstimatorContext:
    mdp: FiniteHorizonMDP
    policy: Policy
    trajectories: tuple[Trajectory, ...]


class CreditEstimator(Protocol):
    name: str

    def estimate(
        self,
        context: EstimatorContext,
    ) -> tuple[tuple[float, ...], ...]:
        """
        Return one credit value for every action in every trajectory.

        The output shape must match the trajectory batch:
        output[i][t] is the estimated credit for trajectories[i].steps[t].
        """
```

---

## 7. Repository structure

```text
agent-credit-bench/
├── LICENSE
├── PLAN.md
├── README.md
├── pyproject.toml
│
├── src/
│   └── agent_credit_bench/
│       ├── __init__.py
│       ├── types.py
│       ├── mdp.py
│       ├── policy.py
│       ├── oracle.py
│       ├── sampling.py
│       ├── benchmark.py
│       ├── metrics.py
│       │
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── delayed_effect.py
│       │   ├── recovery.py
│       │   └── variable_horizon.py
│       │
│       └── estimators/
│           ├── __init__.py
│           ├── oracle.py
│           ├── outcome_broadcast.py
│           ├── batch_centered.py
│           └── turn_loo.py
│
├── experiments/
│   ├── delayed_horizon_sweep.py
│   ├── recovery_diagnostic.py
│   └── variable_horizon_sweep.py
│
├── tests/
│   ├── test_oracle.py
│   ├── test_sampling.py
│   ├── test_delayed_effect.py
│   ├── test_recovery.py
│   ├── test_variable_horizon.py
│   ├── test_estimators.py
│   └── test_metrics.py
│
└── results/
    └── README.md
```

Do not create this entire tree before it is needed. Files should be added milestone by milestone.

---

## 8. Diagnostic environments

## 8.1 `DelayedEffectEnv`

### Purpose

Test whether an estimator leaks delayed terminal reward onto actions that have no effect on the outcome.

### Structure

At `t=0`, the agent selects an action that changes the probability of eventual success.

At later timesteps, it selects among distractor actions with identical transitions and reward distributions.

```text
t=0: GOOD or BAD
          |
          | changes latent success probability
          v
t=1: distractor A or distractor B
t=2: distractor A or distractor B
...
t=H: terminal reward
```

Because the distractor actions are behaviorally identical:

\[
A_t^\pi(s,a)=0
\]

for every distractor action.

### Parameters

```python
DelayedEffectEnv(
    horizon=8,
    good_success_probability=0.8,
    bad_success_probability=0.2,
    num_distractor_actions=2,
)
```

### Primary diagnostics

- error as horizon increases;
- credit assigned to the initial action;
- credit assigned to zero-advantage distractors;
- null-action leakage.

### Required horizons

```text
2, 4, 8, 16, 32
```

---

## 8.2 `RecoveryEnv`

### Purpose

Test whether an estimator distinguishes an early mistake from the later action that repairs it.

### Structure

```text
t=0:
    GOOD ──────────────────────→ success path

    BAD ─→ t=1:
                RECOVER ───────→ success path
                GIVE_UP ───────→ failure path
```

The important sampled trajectory is:

```text
BAD → RECOVER → success
```

A terminal-reward broadcast estimator may assign positive credit to both actions.

The oracle should generally assign:

```text
BAD       negative advantage
RECOVER   positive advantage
```

The exact values depend on the fixed policy.

### Primary diagnostics

- sign assigned to `BAD`;
- sign assigned to `RECOVER`;
- frequency with which the estimator praises both actions;
- estimator behavior on successful recovered trajectories.

---

## 8.3 `VariableHorizonEnv`

### Purpose

Test estimators when actions affect whether future turns exist.

### Structure

```text
t=0:
    STOP      → terminal reward
    CONTINUE  → t=1

t=1:
    STOP      → terminal reward
    CONTINUE  → t=2

...
```

The environment must be configured so that continuing sometimes has positive value and sometimes has negative value, depending on the current state or timestep.

### Primary diagnostics

- credit bias by trajectory length;
- credit bias by timestep;
- behavior as the policy’s stop probability changes;
- difference between trajectory-level and turn-conditioned baselines.

### Implementation order

This is the third environment. Do not implement it until the delayed-effect benchmark and recovery benchmark work.

---

## 9. Baseline estimators

## 9.1 `OracleAdvantage`

Returns the exact advantage of each sampled action.

```python
credit = exact_values.advantages[
    step.timestep,
    step.state,
    step.action,
]
```

### Purpose

- validate the benchmark runner;
- establish the perfect-score baseline;
- catch bugs in metrics;
- provide expected outputs in examples.

The oracle must score perfectly, up to floating-point tolerance.

---

## 9.2 `OutcomeBroadcast`

Assign the trajectory’s total return to every action.

```python
credit_t = trajectory.total_return
```

### Purpose

Provide a deliberately naive baseline that exposes reward smearing.

This output is not necessarily on an advantage scale, so raw RMSE comparisons must be interpreted carefully.

---

## 9.3 `BatchCenteredBroadcast`

For each trajectory, subtract the mean return of peer trajectories and broadcast the centered result to every action.

```python
credit_i_t = return_i - mean(peer_returns)
```

Prefer leave-one-out centering:

```python
credit_i_t = return_i - mean(return_j for j != i)
```

### Purpose

Represent a simple group-relative trajectory baseline without labeling it as a complete GRPO implementation.

---

## 9.4 `TurnLOO`

At each timestep, compare a trajectory’s return against other trajectories that are still active at that timestep.

```python
credit_i_t = return_i - mean(
    return_j
    for j != i
    if trajectory_j contains timestep t
)
```

### Purpose

Study turn-conditioned baselines and variable-horizon behavior.

The precise estimator contract must be documented before implementation. Edge cases such as a single active trajectory must have explicit behavior.

---

## 9.5 Deferred estimator: `MonteCarloAdvantage`

This estimator is useful but not required for the first working benchmark.

It estimates \(Q^\pi(s,a)\) and \(V^\pi(s)\) using sampled continuations from visited states.

It should demonstrate convergence toward the exact oracle as the number of continuation samples increases.

Implement only after the three core environments and four required estimators work.

---

## 10. Metrics

## 10.1 Root mean squared error

\[
\operatorname{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_i
(\hat A_i-A_i)^2
}
\]

Use when estimator output is intended to be on an advantage-compatible scale.

Do not treat RMSE as the only metric.

---

## 10.2 Rank correlation

Measure whether larger estimated credit corresponds to larger exact advantage.

Report Spearman rank correlation over sampled steps.

The implementation must define behavior for constant-valued inputs.

Pooled Spearman is affected by state-dependent shifts (see §3, baseline equivalence class). Where the batch contains \((t,s)\) pairs with at least two distinct sampled actions — common in these tiny MDPs — also report within-state rank agreement, which is invariant to the equivalence class.

---

## 10.3 Sign accuracy

For actions with non-negligible exact advantage:

\[
\operatorname{sign}(\hat A_i)
=
\operatorname{sign}(A_i)
\]

Exclude oracle advantages satisfying:

\[
|A_i| < \epsilon
\]

Suggested default:

```python
epsilon = 1e-8
```

Report excluded zero-advantage actions separately.

---

## 10.4 Zero-credit leakage

Measure how much absolute estimated credit is assigned to actions whose exact advantage is zero.

One possible normalized definition:

\[
\text{LeakageRatio}
=
\frac{
\sum_{i:|A_i|<\epsilon}
|\hat A_i|
}{
\sum_i|\hat A_i|+\epsilon
}
\]

Also report the unnormalized mean:

\[
\operatorname{mean}_{i:|A_i|<\epsilon}
|\hat A_i|
\]

The normalized and raw values answer different questions, so retain both if practical.

---

## 10.5 Per-turn bias and variance

For each timestep:

\[
\operatorname{Bias}_t
=
\mathbb{E}[\hat A_t-A_t]
\]

\[
\operatorname{Variance}_t
=
\operatorname{Var}(\hat A_t)
\]

These should be estimated over repeated trajectory batches and random seeds.

---

## 10.6 Centered RMSE (baseline-shift-invariant error)

Distance from the estimate to the nearest member of the baseline equivalence class:

\[
\operatorname{cRMSE}
=
\min_b
\sqrt{
\frac{1}{N}
\sum_i
\left(\hat A_i - A_i - b(t_i, s_i)\right)^2
}
\]

The minimizing shift has a closed form: for each visited \((t,s)\), \(b(t,s)\) is the mean residual \(\hat A - A\) over sampled steps at that \((t,s)\).

Caveats that must be implemented and reported:

- a \((t,s)\) visited by only one sampled step has its residual centered to exactly zero, which flatters the estimator; report the fraction of steps at multi-visit states as coverage;
- \(\operatorname{cRMSE} \le \operatorname{RMSE}\) always. A large gap means the error is mostly a state-dependent offset (harmless for gradients); a small gap means the error discriminates between actions (harmful for gradients).

---

## 10.7 Policy-gradient alignment

The metric closest to what estimators are actually for: does the credit produce the right training signal?

Use a tabular softmax parameterization with one logit per \((t, s, a)\):

\[
\pi_\theta(a \mid s, t) = \operatorname{softmax}(\theta_{t,s,\cdot})_a
\]

so that

\[
\nabla_{\theta_{t,s,a'}} \log \pi(a \mid s, t)
=
\mathbb{1}[a' = a] - \pi(a' \mid s, t)
\]

No autodiff dependency: all gradients are closed-form under this parameterization, keeping the suite CPU-only and framework-free.

The exact policy gradient follows from the oracle plus the exact state-visitation distribution \(d_t^\pi\) (computed by forward induction, accounting for terminated probability mass):

\[
g^{*}
=
\sum_t \sum_s
d_t^\pi(s)
\sum_a
\pi(a \mid s)\,
A_t^\pi(s,a)\,
\nabla \log \pi(a \mid s, t)
\]

The estimated gradient for one batch of \(B\) trajectories:

\[
\hat g
=
\frac{1}{B}
\sum_i \sum_t
\hat A_{i,t}\,
\nabla \log \pi(a_{i,t} \mid s_{i,t}, t)
\]

Report over repeated seeded batches:

- **direction bias**: cosine similarity between the mean of \(\hat g\) and \(g^{*}\);
- **magnitude bias**: \(\lVert \operatorname{mean}(\hat g) - g^{*} \rVert / \lVert g^{*} \rVert\);
- **variance**: mean squared deviation of \(\hat g\) around its batch mean, plus the per-batch cosine to \(g^{*}\) as a distribution.

Expected calibration behavior (do not hard-code into tests): `OracleAdvantage` should show cosine near one with the lowest variance; `OutcomeBroadcast` should be near-unbiased in direction but with far higher variance — the clearest demonstration that value error and gradient error are different axes.

---

## 10.8 Which metric answers which question

- "Is the output literally an advantage estimate?" → RMSE, sign accuracy.
- "Does it separate relevant from irrelevant actions?" → leakage, rank correlation, centered RMSE.
- "Is it valid credit for policy-gradient training, and at what sample-efficiency cost?" → gradient direction bias and variance.

A benchmark row is complete only when all three columns are present. No single scalar summarizes an estimator, and reporting RMSE alone would misgrade valid-but-uncentered estimators as broken.

---

## 11. Core experiments

## 11.1 Delayed-effect horizon sweep

Run:

```text
horizon ∈ {2, 4, 8, 16, 32}
```

Compare:

- `OracleAdvantage`;
- `OutcomeBroadcast`;
- `BatchCenteredBroadcast`.

Produce:

1. estimator error versus horizon (RMSE and centered RMSE side by side);
2. zero-credit leakage versus horizon;
3. average predicted credit by timestep;
4. gradient direction bias and variance versus horizon.

### Expected qualitative result

Naive trajectory-level estimators should increasingly smear credit over irrelevant actions as the horizon grows. At the same time, `OutcomeBroadcast`'s mean-gradient cosine should stay near one while its gradient variance grows — value smearing and gradient invalidity are different failures, and this figure should show both axes at once.

Do not hard-code this expected result into tests. Tests should validate implementation invariants, not force the experiment to agree with the hypothesis.

---

## 11.2 Recovery diagnostic

Condition analysis on successful trajectories of the form:

```text
BAD → RECOVER → success
```

Produce a plot or table containing average credit assigned to:

- `BAD`;
- `RECOVER`.

### Expected qualitative result

A useful estimator should distinguish the negative early action from the positive recovery action.

---

## 11.3 Variable-horizon sweep (flagship)

This is the flagship experiment — the one place `v0.1` can produce a genuinely non-obvious quantitative result rather than confirming textbook behavior.

The open question it targets: turn-conditioned baselines compare a trajectory against peers *still active at that timestep*, which conditions the baseline on survival. Does that conditioning introduce gradient direction bias, or does it only change credit values and variance? Either answer is informative, the exact-oracle setting can settle it precisely, and the result connects directly to the ongoing length-bias debate around group-relative methods (length normalization in GRPO variants). Do not presuppose the answer; measure it.

Sweep over:

- maximum horizon;
- policy stop probability;
- batch size.

Compare:

- trajectory-centered credit;
- turn-conditioned leave-one-out credit;
- exact advantage.

Produce:

1. credit-value bias versus stop probability;
2. credit-value bias versus trajectory length;
3. credit estimates by timestep;
4. gradient bias–variance decomposition for trajectory-centered versus turn-conditioned baselines;
5. the crossover frontier: the region of (stop probability, horizon, batch size) where turn-conditioning wins on gradient error, if such a region exists.

This experiment should not begin until the first two diagnostics are complete.

---

## 12. Testing strategy

## 12.1 Oracle tests

- [ ] one-step bandit values match a hand calculation;
- [ ] two-step deterministic MDP values match a hand calculation;
- [ ] stochastic transition values match a hand calculation;
- [ ] Bellman identities hold;
- [ ] advantage is equal to \(Q-V\);
- [ ] policy-weighted advantage is zero at every reachable state:

\[
\sum_a \pi(a\mid s)A^\pi(s,a)=0
\]

This final invariant is especially important.

---

## 12.2 Environment tests

For every environment:

- [ ] transition probabilities sum to one;
- [ ] policy probabilities sum to one;
- [ ] all returned actions are valid;
- [ ] all non-terminal next states are reachable at the following timestep;
- [ ] trajectories do not exceed the environment horizon;
- [ ] terminal trajectories stop sampling;
- [ ] seeded sampling is reproducible.

For `DelayedEffectEnv`:

- [ ] distractor actions have equal Q-values;
- [ ] distractor actions have zero advantage;
- [ ] the initial good action has higher Q-value than the bad action.

For `RecoveryEnv`:

- [ ] `RECOVER` has higher Q-value than `GIVE_UP`;
- [ ] under the selected policy, `BAD` has negative advantage;
- [ ] under the selected policy, `RECOVER` has positive advantage.

---

## 12.3 Sampler tests

- [ ] sampled action frequencies approach policy probabilities;
- [ ] sampled transition frequencies approach transition probabilities;
- [ ] empirical mean return approaches exact initial-state value;
- [ ] returned trajectory steps preserve state, action, reward, and timestep alignment.

Use generous statistical tolerances to avoid flaky tests.

---

## 12.4 Estimator tests

- [ ] output batch length matches input batch length;
- [ ] output trajectory lengths match sampled trajectory lengths;
- [ ] all returned credit values are finite;
- [ ] oracle estimator matches exact advantages;
- [ ] broadcast estimator assigns one value to every action in a trajectory;
- [ ] leave-one-out estimators do not include the target trajectory in their baseline;
- [ ] single-member baseline edge cases are handled explicitly.

---

## 12.5 Gradient-metric tests

- [ ] the exact state-visitation distribution \(d_t^\pi\) sums to the alive probability mass at each timestep;
- [ ] oracle credit reproduces the exact gradient: mean estimated gradient converges to \(g^{*}\) (generous statistical tolerance);
- [ ] adding an arbitrary \(b(t,s)\) to oracle credit leaves centered RMSE at zero and the expected gradient unchanged;
- [ ] `OutcomeBroadcast`'s mean gradient converges toward \(g^{*}\) as the number of batches grows (generous statistical tolerance);
- [ ] centered RMSE never exceeds RMSE;
- [ ] single-visit states are counted correctly in the centered-RMSE coverage report.

---

## 13. Milestones

# M0 — Repository skeleton

### Tasks

- [ ] create `pyproject.toml`;
- [ ] create `src/agent_credit_bench/__init__.py`;
- [ ] create `src/agent_credit_bench/types.py`;
- [ ] create `src/agent_credit_bench/mdp.py`;
- [ ] create `src/agent_credit_bench/oracle.py`;
- [ ] create `tests/test_oracle.py`;
- [ ] configure `pytest`;
- [ ] configure `ruff`.

### Exit condition

The package imports successfully and the test suite runs, even if most functionality is not yet implemented.

---

# M1 — Exact oracle

### Tasks

- [ ] implement finite-horizon backward induction;
- [ ] calculate exact \(Q_t^\pi(s,a)\);
- [ ] calculate exact \(V_t^\pi(s)\);
- [ ] calculate exact \(A_t^\pi(s,a)\);
- [ ] validate probability distributions;
- [ ] add one-step bandit test;
- [ ] add two-step deterministic test;
- [ ] add stochastic transition test;
- [ ] test the policy-weighted-zero-advantage invariant.

### First hand-calculated test

```text
State: s0

Action A → reward 1
Action B → reward 0

π(A | s0) = 0.5
π(B | s0) = 0.5
```

Expected values:

\[
Q(s_0,A)=1
\]

\[
Q(s_0,B)=0
\]

\[
V(s_0)=0.5
\]

\[
A(s_0,A)=+0.5
\]

\[
A(s_0,B)=-0.5
\]

### Exit condition

All hand-calculated oracle tests pass.

### Commit

```text
feat: add exact finite-horizon advantage oracle
```

---

# M2 — Sampling and delayed effect

### Tasks

- [ ] implement trajectory sampling;
- [ ] implement a simple tabular policy;
- [ ] implement `DelayedEffectEnv`;
- [ ] verify distractor advantages are exactly zero;
- [ ] implement `OracleAdvantage`;
- [ ] implement `OutcomeBroadcast`;
- [ ] implement `BatchCenteredBroadcast`;
- [ ] run horizon sweep;
- [ ] produce the first diagnostic figure.

### Exit condition

The repository produces a reproducible plot showing estimated credit by timestep or leakage versus horizon.

This is the first point at which the project has demonstrated its core idea.

### Commit sequence

```text
feat: add policy and trajectory sampling
feat: add delayed-effect environment
feat: add broadcast credit baselines
exp: add delayed-effect horizon sweep
```

---

# M3 — Benchmark runner and metrics

### Target API

```python
from agent_credit_bench import run_benchmark
from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import BatchCenteredBroadcast

results = run_benchmark(
    mdp=DelayedEffectEnv(horizon=16),
    policy=policy,
    estimator=BatchCenteredBroadcast(),
    batch_size=1_000,
    seeds=range(10),
)
```

### Tasks

- [ ] implement estimator context;
- [ ] implement reusable benchmark runner;
- [ ] implement RMSE;
- [ ] implement rank correlation;
- [ ] implement sign accuracy;
- [ ] implement zero-credit leakage;
- [ ] implement centered RMSE with coverage reporting;
- [ ] implement the tabular softmax parameterization and exact policy gradient \(g^{*}\);
- [ ] implement the exact state-visitation distribution by forward induction;
- [ ] implement gradient direction-bias and variance metrics;
- [ ] implement per-turn aggregation;
- [ ] save experiment results in a stable tabular format;
- [ ] add tests for every metric.

### Exit condition

A user can run one function and receive structured metrics without editing experiment internals.

### Commit

```text
feat: add reusable benchmark runner and credit metrics
```

---

# M4 — Recovery diagnostic

### Tasks

- [ ] implement `RecoveryEnv`;
- [ ] choose and document the fixed evaluation policy;
- [ ] verify the oracle sign pattern;
- [ ] run all existing estimators;
- [ ] analyze successful recovered trajectories;
- [ ] produce the recovery diagnostic figure.

### Exit condition

The benchmark clearly shows whether an estimator distinguishes `BAD` from `RECOVER`.

### Commit

```text
feat: add recovery credit diagnostic
```

---

# M5 — Variable-horizon diagnostic

### Tasks

- [ ] write a precise environment specification before coding;
- [ ] implement `VariableHorizonEnv`;
- [ ] implement `TurnLOO`;
- [ ] define single-peer and no-peer behavior;
- [ ] sweep policy stop probabilities;
- [ ] measure bias by trajectory length;
- [ ] measure bias by timestep;
- [ ] produce the variable-horizon figures.

### Exit condition

The benchmark exposes at least one measurable difference between trajectory-level and turn-conditioned baselines under variable termination.

### Commit

```text
feat: add variable-horizon credit diagnostic
```

---

# M6 — Public release

### Tasks

- [ ] write README motivation;
- [ ] add installation instructions;
- [ ] add a minimal usage example;
- [ ] add benchmark methodology;
- [ ] add the three primary figures;
- [ ] document limitations, stating the external-validity limits explicitly: no function approximation, no token-level policies, no KL-regularized objectives, no off-policy updates, no stochastic verifiers, no partial observability — a good score here is necessary-ish, not sufficient, for real agentic RL;
- [ ] add prior-work references, positioning the suite against bsuite-style agent diagnostics (which score agents, not estimators) and against approximate Monte-Carlo ground truth (which this replaces with exact values);
- [ ] add a license;
- [ ] configure GitHub Actions;
- [ ] run tests from a clean environment;
- [ ] tag `v0.1.0`.

### README structure

```text
1. One-sentence pitch
2. Why credit estimators need unit tests
3. What this measures that end-task evals and bsuite do not
4. A simple failure example
5. Installation
6. Quick start
7. Environments
8. Estimators
9. Metrics (identification vs gradient families)
10. Results
11. Writing a custom estimator
12. Limitations
13. References
```

### Exit condition

A new user can clone the repository, install it, reproduce one figure, and plug in a custom estimator without reading the source code.

---

## 14. Suggested commit order

```text
1. chore: initialize Python package and test configuration
2. feat: add exact finite-horizon advantage oracle
3. feat: add policy and trajectory sampling
4. feat: add delayed-effect environment
5. feat: add broadcast credit estimators
6. exp: add delayed-effect horizon sweep
7. feat: add benchmark runner and metrics
8. feat: add recovery diagnostic
9. feat: add variable-horizon diagnostic
10. docs: add public benchmark documentation
11. release: agent-credit-bench v0.1.0
```

Every meaningful milestone should end with a working commit.

---

## 15. Scope-control rules

1. **No LLM integration before `v0.1.0`.**
2. **No new environment until the current environment produces a tested diagnostic.**
3. **No abstraction until at least two components require it.**
4. **No framework dependency merely to avoid writing a small amount of code.**
5. **No estimator implementation without a written mathematical definition.**
6. **No claim that an estimator is biased without specifying the target and sampling procedure.**
7. **No polishing the README before the first real figure exists.**
8. **No paper-writing until the benchmark reveals a genuinely non-obvious result.**
9. **Every experiment must be reproducible from a command checked into the repository.**
10. **The benchmark remains CPU-runnable.**
11. **No metric ships without stating which consumer question it answers (§10.8).**
12. **Gradient metrics stay closed-form — no autodiff framework dependency.**

---

## 16. Post-v0.1 roadmap (adoption bridge)

The `v0.1` scope rules stay in force until `v0.1.0` is tagged. A benchmark nobody plugs a real implementation into is a demo, so immediately after the tag, in priority order:

1. **Estimator adapters** — thin wrappers that run the *actual* advantage computations from verl, TRL, and GiGPO-style reference code through the `CreditEstimator` protocol. The protocol was designed so that an adapter should be under ~50 lines; if it is not, that is protocol feedback for `v0.2`.
2. **`MonteCarloAdvantage`** (§9.5) — demonstrates convergence to the oracle and calibrates how much rollout sampling exact metrics are worth.
3. **Flagship write-up** — if the variable-horizon diagnostic (§11.3) shows a measurable gradient bias–variance crossover between trajectory-centered and turn-conditioned baselines, that result — not the suite itself — is the publishable unit (consistent with scope rule 8).

None of these change the `v0.1` definition of done.

---

## 17. Immediate next action

Do only the following:

- [ ] create `pyproject.toml`;
- [ ] create `src/agent_credit_bench/__init__.py`;
- [ ] create `src/agent_credit_bench/mdp.py`;
- [ ] create `src/agent_credit_bench/oracle.py`;
- [ ] create `tests/test_oracle.py`;
- [ ] implement the one-state, two-action hand calculation;
- [ ] make that test pass;
- [ ] commit it.

### Stopping condition

Stop the first work session when this assertion passes:

```python
assert values.action_values[(0, "s0", "A")] == pytest.approx(1.0)
assert values.action_values[(0, "s0", "B")] == pytest.approx(0.0)
assert values.state_values[(0, "s0")] == pytest.approx(0.5)
assert values.advantages[(0, "s0", "A")] == pytest.approx(0.5)
assert values.advantages[(0, "s0", "B")] == pytest.approx(-0.5)
```

That passing test—not the full folder tree—is the first project milestone.
