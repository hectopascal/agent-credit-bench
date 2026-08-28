# Turn-conditioned baselines buy value calibration, not direction correction

*The flagship variable-horizon result (plan.md §11.3, §16 item 3). Numbers
from `results/variable_horizon_sweep.csv`; regenerate with
`python experiments/variable_horizon_sweep.py --batch-size 500 --num-seeds 10`.*

## Question

Turn-conditioned leave-one-out baselines (TurnLOO) compare a trajectory
against peers *still active at that timestep*. Survival correlates with what
a trajectory did, so a natural worry is that conditioning the baseline on
survival biases the policy gradient. Does it — or does turn-conditioning only
change credit values and variance?

## Setup

`VariableHorizonEnv` with stop rewards (0, 2, 1, 3, 0): STOP/CONTINUE at every
turn, forced STOP at the end, continuing has positive exact advantage at t=0
and t=2 and negative at t=1 and t=3 under a 0.5 stop probability. Policies
sweep stop probability over {0.2, 0.35, 0.5, 0.65, 0.8}; batches of 500
trajectories, 10 seeds. Estimators: exact advantage (anchor),
trajectory-centered LOO broadcast, TurnLOO. Gradient metrics use the exact
closed-form g* under a tabular softmax parameterization (plan.md §10.7).

## Results

**1. No measurable gradient direction bias from turn-conditioning.**
cosine(mean gradient, g*) at every stop probability:

| stop prob | oracle   | trajectory-centered | TurnLOO  |
| --------- | -------- | ------------------- | -------- |
| 0.2       | 0.999869 | 0.999856            | 0.999889 |
| 0.35      | 0.999988 | 0.999959            | 0.999985 |
| 0.5       | 0.999966 | 0.999803            | 0.999915 |
| 0.65      | 0.999997 | 0.999933            | 0.999986 |
| 0.8       | 0.999966 | 0.999890            | 0.999962 |

All three sit within sampling noise of each other; TurnLOO is if anything
*closer* to g* than trajectory-centering at every point. The worried-about
bias does not exist. This matches theory: the peer set active at t is
independent of trajectory i's own actions, so the baseline stays a valid
control variate — E[b · grad log pi] = 0 survives the conditioning.

**2. Turn-conditioning removes large per-turn credit-value bias.**
At stop probability 0.5, trajectory-centered credit is biased by up to
±0.8 per timestep (+0.80 at t=1, −0.80 at t=4; see
`results/variable_horizon_turn_bias.png`): trajectories alive at late
timesteps are a return-selected subpopulation, and a baseline computed over
the *whole* batch mis-centers them. TurnLOO's per-turn bias is below 0.05
everywhere — its baseline is the conditional mean of exactly that
subpopulation.

**3. A consistent but modest variance reduction.**
TurnLOO's gradient variance is 3–20% below trajectory-centering at every stop
probability (e.g. 0.00153 vs 0.00178 at p=0.5), while the oracle sits 3–8x
lower still.

## Interpretation

Turn-conditioning is *value* calibration, not *direction* correction. If all
you consume is the policy-gradient direction, trajectory-centering was never
biased and turn-conditioning buys only a modest variance improvement. But any
consumer that reads credit values directly — advantage-thresholded filtering,
step-level data selection, per-turn reward shaping, or analysis that
interprets per-turn credit — inherits the ±0.8 per-turn distortion under
variable termination, and turn-conditioning removes it. This reframes the
length-bias discussion around group-relative methods: under variable horizon
the pathology of a whole-batch baseline shows up in credit values (and
therefore in anything value-consuming), not in the expected gradient.

## Honest scope

This is a clean, exactly-measured result on a five-step tabular MDP — a
workshop-note-sized claim, not a paper (scope rule 8). What it settles is the
mechanism; whether the value-bias term matters at LLM scale depends on how
much of the training pipeline consumes credit values rather than gradients.
