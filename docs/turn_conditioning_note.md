# Turn-conditioned baselines: corrected variable-horizon result

*This note supersedes the original batch-500-only interpretation. The current
artifacts come from the full default sweep in
`experiments/variable_horizon_sweep.py`: horizons {3, 5, 8}, group sizes
{2, 4, 8, 32, 128, 500}, stop probabilities {0.2, 0.35, 0.5, 0.65, 0.8},
seeds 0–199, `continue_reward=0`, and stop rewards made by repeating
`(0, 2, 1, 3)` to each horizon. Summary and per-turn data are in
`results/variable_horizon_sweep.csv` and
`results/variable_horizon_turn_stats.csv`.*

## Question

TurnLOO compares each trajectory with *other* trajectories still active at a
timestep. Does this survival-conditioned baseline preserve the expected
policy gradient? Separately, when does it improve credit-value calibration or
finite-batch gradient error relative to one trajectory-level baseline?

Those are three different questions: expected-gradient validity,
identification of per-turn advantage values, and finite-batch mean-squared
gradient error. The original write-up blurred them.

## Audit finding: the old no-peer rule was wrong

The initial TurnLOO implementation emitted zero whenever a trajectory had no
active peer. That is not a neutral baseline: it deletes the trajectory's
policy-gradient contribution at that timestep. A batch of 500 made the event
rare enough in the featured setting to produce a misleading empirical null.

On the five-step environment with stop rewards `(0, 2, 1, 3, 0)`,
`continue_reward=0`, and stop probability 0.5, a reproduction over seeds
0–4,999 of the old rule found:

| batch | cosine(mean gradient, exact) | relative mean-gradient error |
| ----: | ---------------------------: | -----------------------: |
| 2     | 0.974320                     | 0.225378                 |
| 4     | 0.988629                     | 0.151427                 |
| 8     | 0.995717                     | 0.092861                 |

The earlier statement that TurnLOO introduced "no measurable gradient
direction bias" across the tested problem was therefore too broad. It was a
result for one large group size, not validation of the estimator's edge-case
contract.

The legacy rule is retained only inside
`experiments/turn_loo_fallback_audit.py`; its 5,000-seed output is committed
as `results/turn_loo_fallback_audit.csv` so these before/after numbers remain
reproducible after the production estimator was fixed.

## Correction and expected-gradient result

When no peer is active, the corrected estimator uses a zero baseline and
therefore emits the trajectory's raw return. This is ordinary REINFORCE for
that sample; it preserves the contribution instead of replacing it with zero.

For the peer-present case, the baseline uses only other independently sampled
trajectories. Whether those peers survived to timestep `t` does not depend on
the current trajectory's action, so it remains an action-independent control
variate for that trajectory. Exact enumeration over every possible trajectory
batch in the five-step environment verifies
`E[estimated gradient] = exact gradient` for batch sizes 1, 2, and 3 to an
absolute tolerance of 1e-12.

The corrected 5,000-seed reproduction at horizon 5 and stop probability 0.5
is consistent with that exact result:

| batch | mean-gradient cosine | relative mean-gradient error | gradient variance |
| ----: | -------------------: | -----------------------: | ----------------: |
| 2     | 0.999876             | 0.019731                 | 1.125613          |
| 4     | 0.999947             | 0.010308                 | 0.367836          |
| 8     | 0.999966             | 0.009006                 | 0.142420          |

The residual mean-gradient errors are Monte Carlo estimation error, not an
exact bias calculation; the enumeration is the stronger validity check.

## Value calibration still improves at large groups

The useful identification result survives. In the committed 200-seed sweep at
horizon 5, batch 500, and stop probability 0.5, the whole-batch trajectory
baseline has per-turn mean credit error as high as +0.813 at timestep 1 and
−0.809 at timestep 4. TurnLOO's largest absolute per-turn error in that
configuration is 0.0134.

The reason is selection by survival. Trajectories active late in an episode
are not representative of the whole batch, so a whole-batch mean return
mis-centers their credit. TurnLOO estimates the mean return within the active
peer population. This matters when downstream logic consumes credit values
directly, such as advantage-thresholded filtering, step-level data selection,
or diagnostic interpretation.

## Finite-batch gradient error has a crossover

Unbiasedness does not imply lowest variance. The raw-return fallback is noisy
when tiny groups frequently have no peer, and corrected TurnLOO has slightly
higher normalized gradient MSE than trajectory-centering at small groups. In
the 5,000-seed audit, TurnLOO versus trajectory-centered normalized MSE is
3.0659 versus 2.9637 at batch 2, 1.0019 versus 0.8859 at batch 4, and 0.3879
versus 0.3597 at batch 8.

In the committed 200-seed sweep at horizon 5 and stop probability 0.5, the
ordering reverses as the active peer pool grows:

| batch | trajectory-centered MSE | TurnLOO MSE |
| ----: | ----------------------: | ----------: |
| 32    | 0.082830                | 0.073787    |
| 128   | 0.018318                | 0.014509    |
| 500   | 0.004497                | 0.003270    |

The crossover is not universal: it varies with horizon, stop probability, and
group size. `results/variable_horizon_frontier.png` plots the complete sweep;
the CSV, rather than one featured slice, is the evidence for a particular
configuration.

## Corrected interpretation

Turn-conditioning can buy substantially better per-turn value calibration,
and with enough active peers it can also reduce gradient MSE. It is not a
blanket variance improvement. The raw-return fallback (a zero baseline) is a
sufficient action-independent choice for expected-gradient validity, but it
is not unique; any fallback baseline must not depend on the sampled action.
The practical bias-variance tradeoff depends on the survival distribution and
group size.

This remains a result on small tabular MDPs. It validates the mechanism and
provides regression tests for estimator implementations; it does not establish
that the same crossover materially affects an LLM training run.
