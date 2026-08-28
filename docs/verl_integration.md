# Scoring verl's actual advantage implementations

`agent_credit_bench.integrations.verl` runs the conformance suite against
[verl](https://github.com/volcengine/verl)'s own advantage functions
(`verl.trainer.ppo.core_algos`) — the code that computes advantages in real
verl training runs, not this repo's "-style" reimplementations.

```bash
pip install "agent-credit-bench[verl]"   # torch CPU is sufficient
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
feeds zeros). Tested against verl 0.9.0; CI runs the integration on every
push.

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

Recovery diagnostic (`experiments/verl_conformance.py`, batch 2000, mean
credit on successful `BAD -> RECOVER` trajectories, exact advantages −0.25
and +0.50):

| estimator                 | credit(BAD) | credit(RECOVER) | praises both |
| ------------------------- | ----------- | --------------- | ------------ |
| oracle_advantage          | −0.25       | +0.50           | 0%           |
| verl_grpo                 | +0.59       | +0.59           | 100%         |
| verl_dr_grpo              | +0.26       | +0.26           | 100%         |
| verl_rloo                 | +0.26       | +0.26           | 100%         |
| verl_reinforce_plus_plus  | +0.72       | +0.72           | 100%         |
| verl_gae_exact_lam1       | +0.56       | +1.10           | 100%         |
| verl_gae_exact_lam0       | −0.69       | +1.42           | 0%           |
| verl_gae_zero_lam1        | +0.72       | +0.72           | 100%         |

**1. Every outcome-based estimator in verl praises repaired mistakes.**
GRPO, Dr. GRPO, RLOO, and REINFORCE++ all assign positive credit to the BAD
action on 100% of trajectories that recovered — the suite's headline failure,
reproduced on verl's real code.

**2. A perfect critic does not fix it at verl's default λ = 1.**
`compute_gae_advantage_return` with the oracle's exact V still praises BAD
(+0.56): at λ = 1, GAE telescopes to return-to-go minus baseline, so the
critic only sets the baseline and never separates turns. At λ = 0 the same
critic recovers the exact sign structure (−0.69 / +1.42, 0% praised). The
critic is consulted in proportion to 1 − λ; verl defaults to `lam: 1.0`.

**3. verl's RLOO is exactly leave-one-out centering** —
`r·n/(n−1) − mean·n/(n−1) ≡ r − mean(others)` — and matches the suite's
`BatchCenteredBroadcast` to 1e-9 (tested on variable-length batches, which
also exercises the padding). Dr. GRPO (plain mean-centering) differs from it
by exactly the factor n/(n−1).

**4. verl's GRPO divides by the Bessel-corrected sample std plus 1e-6**
(`torch.std`), not the population std that DeepSeekMath's formula is usually
transcribed with (and that this suite's `GRPOStyleNormalized` uses, with
epsilon 1e-4). A ~0.1–1% scale delta at typical group sizes — irrelevant for
training, but exactly the kind of implementation drift a conformance suite
should pin down rather than average over.

**5. GAE and REINFORCE++ whiten advantages across the batch** before
returning them. On delayed-effect distractor turns whose TD errors are
exactly zero, whitening leaves one shared nonzero constant — a pure
credit-*value* shift that the gradient-validity metrics correctly ignore and
the identification metrics correctly flag (see the two-families discussion
in the README).

## What this does and does not capture

It scores verl's advantage computation — the function that turns rewards
into per-token credit. Everything downstream of it in a real run (PPO
clipping, KL penalties, within-turn token structure, optimizer dynamics) is
out of scope here, as is critic *learning* (`VerlGAE` deliberately replaces
the learned critic with a controlled one). Validating those requires
training-time experiments — e.g. running a verbalized suite environment
inside a real verl training loop, where the underlying MDP stays known and
exact advantages remain computable per checkpoint.
