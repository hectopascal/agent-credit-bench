# Re-validation report

Audit date: 2026-08-30.

This report records what was rerun, which claims changed, and what the
remaining scope limits are. Numerical claims in the README and framework
notes were reconciled against the committed CSVs after regeneration.

## Validation status

| Area | Validation performed | Result |
| --- | --- | --- |
| Core MDPs, oracle, estimators, metrics, bridges | Core-only pytest run | 118 passed; 37 optional-framework tests excluded from this count |
| verl adapter | Targeted run against real verl 0.9.0 | 12 passed |
| TRL adapter | Targeted run against real TRL 1.12.0 plus source fingerprints | 10 passed |
| verifiers bridge | Targeted run against real verifiers 0.1.14 | 6 passed |
| OpenRLHF adapter | Linux/amd64 container, real OpenRLHF 0.11.0 pipeline | 9 passed |
| Static checks | Ruff | Passed |
| Core experiment artifacts | All five core/plot-producing entry points rerun from their defaults | CSVs and plots regenerated |

The optional dependency versions are exact pins, and the verl, TRL, and
OpenRLHF adapters also fail closed at runtime if another release is installed.
The OpenRLHF CPU recipe is in `requirements/openrlhf-cpu.txt`; it omits
GPU-only execution paths and is for scoring, not training.

## Claims that were corrected

### Recovery is conditional identification, not expected-gradient bias

The successful `BAD -> RECOVER` slice does show outcome-broadcast methods
assigning positive credit to both actions. It does not by itself show that
their expected policy gradients favor BAD: unsuccessful BAD trajectories
supply offsetting credit. The regenerated 30-seed diagnostic therefore
reports both the selected slice and full-batch aggregates. For example,
`BatchCenteredBroadcast` averages about +0.249 on selected recovered BAD
actions but about -0.249 over all BAD actions, with mean batch-gradient cosine
about 0.9999.

The corrected evidence is in `results/recovery_diagnostic.csv`. The bridge
analysis is likewise scoped to the fitted finite-sample `(t, state)` Markov
projection of logged action frequencies, not the original
history-conditioned LLM policy.

### GiGPO mechanics and related invariants are now directly pinned

The GiGPO recovery results were correct but under-tested. Recomputing the
README's equal-weighted 30-seed aggregate gives +1.1511 credit for BAD and
+1.5714 for RECOVER. Disabling the episode-level term gives +0.5756 and
+0.9958. The often-quoted +0.59/+1.01 counterfactual is seed 0 alone, not the
README protocol; only the BAD value is halved, while RECOVER falls by about
36.6%. A hand calculation now pins both the episode contribution and GiGPO's
population-standard-deviation normalization. The two-level combination,
state-keyed anchor groups, and suffix returns agree with the published GiGPO
construction; this repository's `-style` variant deliberately fixes gamma to
1 and a population-standard-deviation normalizer rather than claiming the
paper's exact training constants.

No fourth environment was needed to distinguish the other GiGPO mechanics.
`VariableHorizonEnv` already revisits the `"alive"` state at multiple
timesteps, and its configurable `continue_reward` supports genuinely dense
rewards. Exact fixtures now distinguish state-only anchors from `(t, state)`
anchors and suffix return-to-go from full episode return. The featured
variable-horizon sweep still omits GiGPO, so this is implementation-conformance
coverage rather than a new empirical comparison in the published figures.

Monte Carlo continuation tests now include `RecoveryEnv`, where GOOD ends
before the horizon, and therefore pin rollout termination. Spearman tie tests
pin average ranks to an exact value rather than merely checking that the
result lies between -1 and 1. The explicit empty-distribution validation
branch remains for its clearer error message; deleting it is behaviorally
equivalent because the subsequent sum-to-one check still rejects an empty
distribution.

### TurnLOO's old no-peer fallback was biased

The original implementation emitted zero credit when a surviving trajectory
had no peer. That removed its REINFORCE term. The production estimator now
uses raw return, corresponding to a zero, action-independent fallback
baseline. Exact enumeration over all batches of sizes 1, 2, and 3 in the
five-step environment matches the oracle expected gradient to absolute
tolerance 1e-12.

The 5,000-seed reproduction still records the old failure. At batches 2, 4,
and 8 its mean-gradient cosine was 0.974320, 0.988629, and 0.995717; the
corrected estimator gives 0.999876, 0.999947, and 0.999966. The correction is
not a universal variance win: raw-return fallback is noisy at tiny groups.
In the 200-seed featured slice, TurnLOO's normalized gradient MSE becomes
lower than trajectory-centering at group size 32 and remains lower at 128
and 500. See `results/turn_loo_fallback_audit.csv`,
`results/variable_horizon_sweep.csv`, and
`docs/turn_conditioning_note.md`.

### Monte Carlo convergence used independent RNG streams

The earlier experiment initialized primary-batch sampling and continuation
sampling from the same seed, coupling the two streams. The corrected protocol
uses trajectory seeds 0-29 and records a disjoint continuation seed
`1,000,003 + 7,919 * trajectory_seed`. With a uniform policy, batch 200,
horizon 8, success probabilities 0.8/0.2, and separate K-sample estimates for
each cached Q and V value (with two distractor actions), mean RMSE is
0.0359 +/- 0.0055 at K=256 and 0.0178 +/- 0.0024 at K=1,024, where the
uncertainty is the population standard deviation across 30 trajectory seeds.
The decreasing curve remains consistent with
the expected inverse-square-root sampling rate. See
`results/monte_carlo_convergence.csv`.

### Leakage now distinguishes three quantities

Under the default delayed-effect protocol, unnormalized broadcast estimators
put about 0.5 mean absolute credit on each zero-advantage distractor. That
per-step value stays flat; total absolute distractor credit per episode grows
with horizon, and the fraction of all absolute credit on distractors rises
from 0.5 at horizon 2 to 0.96875 at horizon 32. The result is structural
smearing, not growth in per-distractor credit. The CSV records all three
measures and the full protocol.

### Cross-framework results are version- and convention-specific

The committed comparison is a protocol-checked merge from separately pinned
environments. For group sizes at least two, the three tested RLOO paths agree
with the suite's leave-one-out centering within 1e-9 on tested batches; verl
0.9.0 has a separate raw-score singleton fallback while TRL and OpenRLHF
adapters reject singleton RLOO groups.

The GRPO variants share a broad formula but differ in epsilon. Their
normalization conventions also need precise labels: group normalization uses
Bessel-corrected sample standard deviation, while OpenRLHF's later batch
whitening uses population standard deviation and verl's `masked_whiten` uses
the sample convention. That population/sample choice, not just float32
rounding, explains the roughly 1e-4 OpenRLHF/verl gaps in whitened recovery
rows. Downstream training effects were not tested.

### Bridge and API edge cases now fail closed

- Malformed model replies use a deterministic minimum-return action by
  default and may instead be configured to raise; they are never silently
  interpreted as the first legal action.
- Mixed environment parameters in a checkpoint log are rejected.
- The verl chat bridge does not execute or reward an action whose generated
  token span is truncated.
- Recursive OpenRLHF estimators reject trajectories with nonterminal rewards,
  because the framework input accepts one scalar environment/reward-model
  reward per sequence.
- Invalid probabilities, empty batches, unsupported gamma/critic pairings,
  and unsupported framework versions are rejected explicitly.
- Zero-gradient cosine and relative-error metrics are `None`, not a numeric
  direction sentinel; finite cosines are clamped to their mathematical
  range.
- Accurate metric names are `mean_gradient_cosine` and
  `relative_mean_gradient_error`. Deprecated object and row aliases remain
  available for 0.x consumers.

## Remaining boundaries

The oracle is exact only for the supplied finite, Markov, undiscounted
tabular model and policy. The experiments do not validate learned critics,
function approximation, token-level credit within a turn, PPO clipping,
optimizer dynamics, KL-regularized objectives, stochastic reward models, or
end-to-end training gains. Experimental comparisons are specific to the
recorded seeds, batch sizes, policies, reward layouts, and environment
parameters. CSVs are the evidence; plots and rounded prose are summaries.
