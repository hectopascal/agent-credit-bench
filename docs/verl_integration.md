# Scoring verl's actual advantage implementations

`agent_credit_bench.integrations.verl` runs the conformance suite against
[verl](https://github.com/volcengine/verl)'s own advantage functions
(`verl.trainer.ppo.core_algos`) — the code that computes advantages in real
verl training runs, not this repo's "-style" reimplementations.

```bash
# From the AgentCreditBench source checkout:
python -m pip install -e ".[verl]"   # torch CPU is sufficient
python experiments/verl_conformance.py
pytest tests/test_verl_integration.py
```

```python
from agent_credit_bench import UniformPolicy, run_benchmark
from agent_credit_bench.envs import RecoveryEnv
from agent_credit_bench.integrations.verl import VerlGRPO

result = run_benchmark(
    mdp=RecoveryEnv(),
    policy=UniformPolicy(),
    estimator=VerlGRPO(),      # calls verl's compute_grpo_outcome_advantage
    batch_size=1_000,
    seeds=range(10),
)
```

Wrapped estimators: `VerlGRPO` (and its `norm_adv_by_std=False` Dr. GRPO
variant), `VerlRLOO`, `VerlReinforcePlusPlus`, and `VerlGAE` with a
controlled critic (`critic="exact"` feeds the oracle's V; `critic="zero"`
feeds zeros). Tested against verl 0.9.0; CI runs the integration on
main-branch pushes and pull requests.

## How trajectories are packed

verl computes advantages on padded `(batch, response_length)` token tensors.
The suite packs **one tensor cell per turn**: cell `(i, t)` holds the reward
of trajectory `i`'s turn `t`, the response mask marks live turns, padding is
masked out, and the whole batch is one group (a suite batch is sampled from
one start state, which is the group-relative estimators' whole-batch
semantics).

This is faithful because of how verl consumes the tensors:

- **Outcome estimators** (GRPO, Dr. GRPO, RLOO) reduce each row to its
  reward *sum* and broadcast over masked cells — identical at any
  granularity.
- **Recursive estimators** (GAE, REINFORCE++) recurse cell by cell, skipping
  masked cells — with one cell per turn the recursion runs over exactly the
  turn-level MDP the suite defines.

Packing uses float64 so conformance comparisons against the suite's pure
Python estimators are exact to rounding, not float32 noise.

## Findings

Recovery diagnostic (`experiments/verl_conformance.py`, batch 2000, seed 0,
recovery-success probability 1.0, mean credit on successful
`BAD -> RECOVER` trajectories, exact advantages −0.25 and +0.50):

| estimator                 | credit(BAD) | credit(RECOVER) | both positive on selected path |
| ------------------------- | ----------- | --------------- | ------------ |
| oracle_advantage          | −0.25       | +0.50           | 0%           |
| verl_grpo                 | +0.59       | +0.59           | 100%         |
| verl_dr_grpo              | +0.26       | +0.26           | 100%         |
| verl_rloo                 | +0.26       | +0.26           | 100%         |
| verl_reinforce_plus_plus  | +0.72       | +0.72           | 100%         |
| verl_gae_exact_lam1       | +0.56       | +1.10           | 100%         |
| verl_gae_exact_lam0       | −0.69       | +1.42           | 0%           |
| verl_gae_zero_lam1        | +0.72       | +0.72           | 100%         |

**1. Every tested outcome-based estimator in verl is positive on the selected
repaired path.** GRPO, Dr. GRPO, RLOO, and REINFORCE++ all assign positive
credit to BAD on 100% of successful `BAD -> RECOVER` trajectories. This
demonstrates a conditional identification limitation on verl's real code; it
does not show that the estimator's expected policy gradient favors BAD.

**2. A perfect critic does not fix it at verl's default λ = 1.**
`compute_gae_advantage_return` with the oracle's exact V still assigns BAD
+0.56 on selected successful repairs: at λ = 1, GAE telescopes to
return-to-go minus baseline, so the critic only sets the baseline and does not
separate the selected turns. At λ = 0 the same critic gives the selected
turns opposite signs (−0.69 / +1.42). These whitened numbers are not literal
oracle advantages, and the selected-path comparison is not an expected-gradient
test. verl defaults to `lam: 1.0`.

The suite's exact value oracle is undiscounted, so `VerlGAE(critic="exact")`
requires `gamma=1.0`. Other gamma values remain available with
`critic="zero"`; they are rejected with the exact critic rather than pairing
discounted GAE with an inconsistent undiscounted value function.

**3. For groups of at least two, verl's RLOO is exactly leave-one-out centering** —
`r·n/(n−1) − mean·n/(n−1) ≡ r − mean(others)` — and matches the suite's
`BatchCenteredBroadcast` to 1e-9 (tested on variable-length batches, which
also exercises the padding). Dr. GRPO (plain mean-centering) differs from it
by exactly the factor n/(n−1). verl 0.9.0 explicitly broadcasts the sequence
score for singleton GRPO/RLOO groups (normalized GRPO divides it by
`1 + epsilon`); the leave-one-out identity is undefined there.

**4. verl's GRPO divides by the Bessel-corrected sample std plus 1e-6**
(`torch.std`), not the population std that DeepSeekMath's formula is usually
transcribed with (and that this suite's `GRPOStyleNormalized` uses, with
epsilon 1e-4). Relative to population-standard-deviation normalization, the
sample-standard-deviation convention reduces the normalized output scale by
29.3% at n=2, 1.58% at n=32, 0.100% at n=500, and 0.025% at n=2000 (before
the epsilon delta). The suite pins this difference; downstream training
impact is not tested here.

**5. GAE and REINFORCE++ whiten advantages across the batch** before
returning them. With an exact critic and GAE lambda 0, delayed-effect
intermediate turns have zero one-step TD residuals, so whitening maps them
to a shared constant. At positive lambda, later TD residuals propagate
backward; zero immediate TD error does not imply zero multistep advantage.
A fixed action-independent baseline preserves the expected policy gradient,
but can change finite-batch gradients and their variance. Batch whitening also
introduces data-dependent centering and scaling. Centered RMSE removes group
shifts; gradient-validity metrics measure their finite-sample consequences.

Likewise, reducing GAE lambda below 1 does not guarantee negative BAD credit
on a selected successful recovery. Before whitening, that credit is
`-0.25 + 0.5 * lambda` in the default recovery environment. The sign separation
shown in the table is the lambda-0 endpoint, with the stated batch and critic.

## What this does and does not capture

It scores verl's advantage computation — the function that turns rewards
into per-token credit. Everything downstream of it in a real run (PPO
clipping, KL penalties, within-turn token structure, optimizer dynamics) is
out of scope here, as is critic *learning* (`VerlGAE` deliberately replaces
the learned critic with a controlled one). Validating those requires
training-time experiments — e.g. running a verbalized suite environment
inside a real verl training loop, where the underlying MDP stays known and
exact advantages remain computable per checkpoint.
