# agent-credit-bench

AgentCreditBench is a lightweight conformance-test suite for turn-level credit
estimators, using tiny finite-horizon MDPs where the exact policy advantage of
every sampled action can be computed by backward induction.

## Why credit estimators need unit tests

Turn-level credit assignment methods for agentic RL are usually validated
indirectly — by end-task success on agent benchmarks, or against approximate
Monte-Carlo ground truth. Both are noisy and confounded. Here the MDPs are
small enough that `A^pi(s, a)` is exact, so an estimator's output can be
scored directly, on CPU, in seconds.

This differs from bsuite-style diagnostics (which score *agents* via learning
curves — no training loop exists here) and from method papers (which propose
estimators; this scores them).

## A simple failure example

On successful `BAD -> RECOVER` trajectories in the recovery environment:

| estimator                | credit(BAD) | credit(RECOVER) | praises both |
| ------------------------ | ----------- | --------------- | ------------ |
| oracle_advantage         | −0.25       | +0.50           | 0%           |
| outcome_broadcast        | +1.00       | +1.00           | 100%         |
| batch_centered_broadcast | +0.26       | +0.26           | 100%         |
| grpo_style_normalized    | +0.59       | +0.59           | 100%         |
| gigpo_style              | +1.17       | +1.60           | 100%         |

Trajectory-level estimators reward the mistake because the trajectory
eventually succeeded. Even GiGPO-style anchor-state grouping praises BAD: a
BAD-then-recovered trajectory has the same return-to-go from the start state
as a GOOD one, so outcome-grouped credit cannot separate them (it does rank
RECOVER above BAD, unlike the flat broadcasts). The exact oracle separates
the signs.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # + pytest, ruff
pip install -e ".[experiments]"   # + matplotlib, for the figures
```

Zero runtime dependencies; Python >= 3.10.

## Quick start

```python
from agent_credit_bench import run_benchmark, UniformPolicy
from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import BatchCenteredBroadcast

result = run_benchmark(
    mdp=DelayedEffectEnv(horizon=16),
    policy=UniformPolicy(),
    estimator=BatchCenteredBroadcast(),
    batch_size=1_000,
    seeds=range(10),
)
print(result.gradient_direction_bias, result.seed_metrics[0].leakage_ratio)
```

## Environments

- **DelayedEffectEnv** — one consequential first action, then behaviorally
  identical distractors with exactly zero advantage; delayed terminal reward.
- **RecoveryEnv** — an early mistake repaired by a later action; tests
  whether the estimator praises both.
- **VariableHorizonEnv** — STOP/CONTINUE at every turn; continuing has
  positive advantage at some timesteps and negative at others.

## Estimators

- **OracleAdvantage** — exact advantage; the perfect-score anchor.
- **OutcomeBroadcast** — trajectory return broadcast to every step (naive).
- **BatchCenteredBroadcast** — leave-one-out group-centered return, broadcast
  (a minimal group-relative baseline; not a full GRPO implementation).
- **TurnLOO** — leave-one-out baseline over trajectories still active at
  each timestep.
- **GRPOStyleNormalized** — the published GRPO group formula:
  (return − mean) / (std + eps), broadcast.
- **GiGPOStyle** — hierarchical episode + anchor-state step grouping, after
  GiGPO's mechanism.
- **MonteCarloAdvantage** — Q − V from sampled continuations; converges to
  the oracle as rollouts grow.
- **TrajectoryReturnAdapter** — wraps any `f(returns) -> credits` function
  (e.g. the actual TRL/verl group-advantage computation) so external library
  code can be scored without adding dependencies here.

## Scoring real framework implementations

With optional extras, the suite runs directly against the advantage code of
three frameworks — [verl](https://github.com/volcengine/verl)'s importable
`core_algos` functions, [TRL](https://github.com/huggingface/trl)'s
trainer math (transcribed and pinned to the installed source by fingerprint
tests), and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)'s full
`compute_advantages_and_returns` pipeline — not the "-style"
reimplementations above:

```bash
pip install "agent-credit-bench[verl]"   # or [trl]; torch CPU is sufficient
python experiments/cross_framework_conformance.py
```

Selected rows (recovery diagnostic; full table in
`results/cross_framework_recovery.csv`):

| estimator                        | credit(BAD) | credit(RECOVER) | praises both |
| -------------------------------- | ----------- | --------------- | ------------ |
| oracle_advantage                 | −0.25       | +0.50           | 0%           |
| verl_grpo                        | +0.59       | +0.59           | 100%         |
| trl_grpo                         | +0.59       | +0.59           | 100%         |
| openrlhf_group_norm              | +0.59       | +0.59           | 100%         |
| verl_rloo = trl_rloo = openrlhf_rloo | +0.26   | +0.26           | 100%         |
| verl_gae_exact_lam1              | +0.56       | +1.10           | 100%         |
| verl_gae_exact_lam0              | −0.69       | +1.42           | 0%           |

Three headlines. Every outcome-based estimator in all three frameworks
praises the repaired mistake. GAE with a *perfect* critic still does at
λ = 1 — the default in both verl and OpenRLHF — because at λ = 1 the critic
only sets the baseline; only λ < 1 bootstraps on it and separates turns.
And the three RLOO implementations are bit-identical (all match this
suite's `BatchCenteredBroadcast` to 1e-9), while the three GRPOs share the
formula but not the epsilon (verl 1e-6, TRL 1e-4, OpenRLHF 1e-9). Details
and more findings per framework:
[docs/verl_integration.md](docs/verl_integration.md),
[docs/trl_integration.md](docs/trl_integration.md),
[docs/openrlhf_integration.md](docs/openrlhf_integration.md). The
integrations stay out of the core: zero runtime dependencies without the
extras, and each framework's tests skip when it is absent (CI runs each in
a dedicated job; OpenRLHF's is Linux-only).

Going further, [recipes/verl_bridge](recipes/verl_bridge/) runs suite
environments *inside* a real verl training run as multi-turn chat games:
a registered agent loop plays the MDP with a live model, logs episodes,
and `analyze_checkpoint.py` scores every estimator against exact
advantages under the empirical policy the model actually played — exact
ground truth on real training data.
[recipes/verifiers_bridge](recipes/verifiers_bridge/) does the same for
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) (the
environment library behind prime-rl and the Environments Hub) as a
`MultiTurnEnv` with a `load_environment` entry point. The framework-free
half of both bridges (`agent_credit_bench.integrations.bridge`) also works
with any text-in/text-out model, no framework required.

## Metrics: two families

Exact advantage plus any state-dependent shift `b(t, s)` yields the same
expected policy gradient, so a single scalar misgrades estimators:

- **Identification** — is the output literally an advantage estimate?
  RMSE, Spearman rank correlation, sign accuracy, zero-credit leakage.
- **Gradient validity** — does it induce the right training signal?
  Centered (shift-invariant) RMSE, and closed-form gradient direction bias,
  magnitude error, and variance under a tabular softmax parameterization.

## Results

![leakage](results/delayed_leakage.png)

Broadcast estimators put ~0.5 |credit| on every zero-advantage distractor
step at every horizon: per-step smearing is flat, so total leaked credit
grows linearly with horizon and the leakage *ratio* climbs toward 1. The
oracle sits at exactly zero.

![recovery](results/recovery_credit.png)

The failure-example table above, as a figure.

![variable horizon](results/variable_horizon_gradient.png)
![per-turn bias](results/variable_horizon_turn_bias.png)

The flagship finding is a precise null plus a real separation: turn-condition
ed LOO introduces **no measurable gradient direction bias** (cosine ≥ 0.9998
for every estimator, every stop probability), but it removes the large
timestep-dependent credit-value bias that trajectory-centered broadcast
carries under variable termination (±0.8 at the extreme timesteps), at
slightly lower gradient variance. Turn-conditioning buys value calibration,
not direction correction — full write-up with tables in
[docs/turn_conditioning_note.md](docs/turn_conditioning_note.md).

![mc convergence](results/monte_carlo_convergence.png)

Monte Carlo "approximate ground truth" needs ~256 continuation rollouts per
(t, s, a) to get within 0.03 RMSE of the exact oracle on an 8-step
delayed-effect environment — the price the exact solver makes unnecessary.

Reproduce with:

```bash
python experiments/delayed_horizon_sweep.py
python experiments/recovery_diagnostic.py
python experiments/variable_horizon_sweep.py
python experiments/monte_carlo_convergence.py
```

## Writing a custom estimator

Implement one method; no registration needed:

```python
from agent_credit_bench.estimators import EstimatorContext

class MyEstimator:
    name = "my_estimator"

    def estimate(self, context: EstimatorContext):
        return tuple(
            tuple(0.0 for _ in trajectory.steps)
            for trajectory in context.trajectories
        )
```

`run_benchmark` accepts it directly.

## Limitations

A good score here is necessary-ish, not sufficient, for real agentic RL.
Deliberately out of scope: function approximation, token-level policies,
KL-regularized objectives, off-policy updates, learned reward models or
stochastic verifiers, partial observability. The MDPs are tabular and tiny —
that is what makes the ground truth exact.

## References

- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* (GAE), 2015.
- Kool et al., *Buy 4 REINFORCE Samples, Get a Baseline for Free!* (leave-
  one-out baselines), 2019.
- Shao et al., *DeepSeekMath* (GRPO), 2024.
- Osband et al., *Behaviour Suite for Reinforcement Learning* (bsuite), 2019
  — structurally similar diagnostics that score agents, not estimators.
- Surveys and methods for turn-level credit assignment in agentic LLM RL
  (GiGPO, Turn-PPO, TRACE) — the estimators this suite is built to test.
