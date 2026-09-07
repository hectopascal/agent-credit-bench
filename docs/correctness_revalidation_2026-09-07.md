# Correctness fixes and complete study rerun

Date: 2026-09-07. Baseline artifacts: commit `4f44b9b`, captured before this
rerun. Evaluation used the current working tree, including the fixes from
preceding audit rounds and the seven fixes described below.

**The current, qualified study findings remain supported.** All seven experiment
entry points were rerun with their default protocols, including the real pinned
framework advantage implementations. Across all eight existing CSVs (2,275
rows), six files reproduce byte for byte. The other two differ only in 29
numeric cells, with maximum absolute change `1.0000680839006293e-15`. No reported
estimator ranking, sign conclusion, crossover, or rounded numerical claim
changes. All six plots were regenerated and visually checked; two long labels
were wrapped to avoid clipping.

This validates the recorded finite-MDP experiments. It does not restore claims
withdrawn in earlier reviews or establish end-to-end LLM training gains.

## Fixes and exposure of the studies

| Confirmed issue | Correction | Exposure of recorded experiments |
| --- | --- | --- |
| Action-order-dependent oracle values and upstream gradients | Compute V directly with accurate summation; center advantages around an on-policy Q near V, with a value-based tie break. Apply the same reference selection to gradient centering. | The failure required widely separated reward scales and an off-policy reference; the default studies use small bounded rewards. |
| Python-version-dependent rejection of valid episode totals | Serialize totals using `math.fsum`; accept the explicitly reconstructed legacy left-to-right sum or a result within two ulps of the stable sum. Individual transition rewards still match MDP support exactly. | The studies sample trajectories directly; they do not read bridge JSONL logs. |
| Oracle/sampler disagreement for approximately normalized probabilities | Normalize accepted policy and transition weights, and consume the normalized transitions in the oracle, visitation, sampler, Monte Carlo continuations, and bridge. | Default study distributions do not exhibit the audited probability-mass discrepancy. |
| Zero truncation reward beats valid negative returns | Use the lowest supported complete return over actions and positive-probability outcomes as the truncation penalty. Record `credit_bench_truncated`; do not execute discarded tokens or log an incomplete episode as complete. | Chat truncation is not part of the tabular or framework advantage experiments. |
| Recursive malformed-reply fallback overflows the stack | Replace recursion with iterative backward induction. | Experiment sampling does not invoke the text-parser fallback. |
| Overlapping or punctuation-bearing legal action names parse incorrectly | Prefer exact complete replies; use alphanumeric boundaries and longest overlapping mentions before selecting the last mention. | Built-in study action names do not trigger the failure. |
| Finite extreme gradient scales break norm/cosine | Use `math.hypot` and a dot product of normalized components; retain rejection of nonfinite inputs. | Recorded gradient magnitudes are far from the underflow/overflow cases. |

The earlier Monte Carlo benchmark reseeding fix also remains in place. The
published convergence experiment already supplies a distinct continuation seed
for every trajectory seed through standalone estimator contexts, so its
protocol does not depend on `run_benchmark`'s former fixed estimator randomness.

The framework rerun exposed a provenance issue in the comparison script: an
unrelated installed AgentCreditBench distribution could label a source checkout
with its old version. The script now checks the imported package location and
reports `source-tree` for a checkout. Framework version guards remain active.

## Before/after artifact comparison

Comparisons require identical schemas and row counts, then compare every
serialized cell in row order. “Identical” below means byte-for-byte identical,
not just matching rounded prose. Some experiment CSVs intentionally serialize
statistics to 12 significant digits; equality therefore applies at that
artifact precision.

| Artifact | Rows | Changed cells | Maximum absolute numeric change |
| --- | ---: | ---: | ---: |
| `delayed_horizon_sweep.csv` | 250 | 0 | 0 |
| `recovery_diagnostic.csv` | 150 | 0 | 0 |
| `turn_loo_fallback_audit.csv` | 9 | 0 | 0 |
| `variable_horizon_sweep.csv` | 270 | 0 | 0 |
| `variable_horizon_turn_stats.csv` | 1,395 | 13 | 1.0001e-15 |
| `monte_carlo_convergence.csv` | 180 | 16 | 6.9389e-17 |
| `verl_recovery.csv` | 8 | 0 | 0 |
| `cross_framework_recovery.csv` | 13 | 0 | 0 |

The [machine-readable comparison](../results/revalidation_2026-09-07/comparison.json)
contains every changed cell and before/after SHA-256 hashes. The saved baseline
can also be recovered with `git show 4f44b9b:results/<filename>`.

## Which findings remain defensible

- **Delayed-effect leakage:** about 0.5 absolute credit per irrelevant turn for
  unnormalized broadcast estimators. Total distractor credit increases with
  horizon; the leakage fraction reaches 0.96875 at horizon 32. Oracle leakage
  remains zero. This concerns credit identification, not proof of a biased
  expected policy gradient.
- **Recovery:** on selected successful `BAD -> RECOVER` paths, the oracle gives
  BAD -0.25 and RECOVER +0.50; the broadcast and featured normalized estimators
  give BAD positive credit. Full-batch aggregates still distinguish this
  conditional diagnostic from expected-gradient bias. The GiGPO aggregate
  remains approximately +1.1511 / +1.5714 on the selected slice.
- **TurnLOO:** the old zero fallback remains demonstrably biased. The corrected
  raw-return fallback passes exact expected-gradient checks. At horizon 5 and
  stop probability 0.5, its normalized gradient MSE remains higher than
  trajectory centering at batches 2, 4, and 8, then lower at 32, 128, and 500.
  For example, batch 8 gives 0.396917 versus 0.377312; batch 32 gives 0.073787
  versus 0.082830. The empirical crossover is protocol-specific, not a universal
  variance advantage or a confidence interval on the crossover location.
- **Monte Carlo convergence:** mean RMSE remains 0.0359392 with population
  standard deviation 0.00553993 at K=256, and 0.0178315 +/- 0.00238385 at
  K=1,024. The curve remains consistent with inverse-square-root convergence.
- **Framework comparisons:** all 21 rows across the standalone verl table and
  merged framework table reproduce exactly. RLOO agreement and the documented
  normalization/epsilon differences remain supported for the tested pins and
  tensor conventions. TRL coverage includes source fingerprints and its real
  `nanstd` helper; it is still a checked transcription of inline trainer math.

Claims of universal TurnLOO superiority, recovery-slice evidence of expected
gradient bias, exact ground truth for a history-conditioned LLM, or downstream
training improvements remain unsupported. The narrower conclusions above are
the ones justified by the experiments.

## Verification

| Check | Result |
| --- | --- |
| Core suite, including the new regression and differential tests | 226 passed; 37 optional tests skipped in the core environment |
| Real verl 0.9.0 | 12 passed |
| Real TRL 1.12.0, including both source fingerprints | 10 passed |
| Real OpenRLHF 0.11.0 with the repository's CPU dependency pins | 9 passed |
| Real verifiers 0.1.14, scripted client driving its rollout loop | 6 passed |
| Ruff and `git diff --check` | Passed |
| Python 3.12 log generation followed by Python 3.10 analysis | Accepted, exit 0 |

That is 263 distinct passing tests across compatible environments, not a claim
that all optional dependencies coexist in one installation. New targeted tests
cover every named failure, including horizon 512, punctuation and overlapping
action names, negative/zero/positive truncation rewards, and extreme finite
gradient scales.

`tests/test_differential.py` additionally checks 3,000 seeded random MDPs against
an independent Fraction-based oracle and gradient calculation, including action
order permutations. Cases include stochastic transitions, deterministic and
nonuniform policies, dense rewards, and early termination. The observed maximum
absolute error was 3.552713678800501e-15. Exhaustive batches of sizes 1-3 for 40
additional MDPs check OutcomeBroadcast, TurnLOO, and BatchCenteredBroadcast
(where batch size is at least two) against exact expected gradients; maximum
absolute error was 1.5265566588595902e-16. These are deterministic regression
checks at ordinary reward scales; the specific large-scale failures have
separate targeted fixtures.

## Protocol and provenance

All default protocols were retained:

- Delayed sweep: horizons 2, 4, 8, 16, 32; batch 1,000; seeds 0-9.
- Recovery: batch 2,000; seeds 0-29; recovery success probability 1.
- Old-versus-corrected TurnLOO audit: 5,000 seeds; batches 2, 4, 8.
- Variable horizon: horizons 3, 5, 8; batches 2, 4, 8, 32, 128, 500; stop
  probabilities 0.2, 0.35, 0.5, 0.65, 0.8; 200 seeds per setting.
- Monte Carlo: horizon 8; batch 200; seeds 0-29; K=1, 4, 16, 64, 256, 1,024;
  continuation seed `1,000,003 + 7,919 * trajectory_seed`.
- Framework comparisons: RecoveryEnv, batch 2,000, seed 0, recovery success
  probability 1; 498 selected successful recovered trajectories.

Core execution used Python 3.10.11 on macOS. Framework execution used Linux
x86_64, Python 3.11.16, and Torch 2.13.0+cpu in separate temporary environments.
verl/TRL used Transformers 5.10.4; OpenRLHF used Transformers 5.15.0, DeepSpeed
0.19.5, and Ray 2.55.0. OpenRLHF used the repository's import-only CPU stubs for
flash-attn and vLLM, whose callables raise if exercised. Those GPU stubs were
excluded from the verl/TRL environment. CPU dependencies were reused from the
existing validation container, with separate compatible package overlays and
isolated namespace paths. No framework advantage function was replaced or
patched. No GPU training was run.

Evidence is retained under [results/revalidation_2026-09-07](../results/revalidation_2026-09-07/):
core run logs, core and framework dependency versions, partial framework CSVs,
the comparison, and hashes of the evaluated source files. The dependency
inventory includes ambient packages; only the named framework was tested in
each respective environment.

From an environment with the core package and experiment dependencies available:

```sh
PYTHONPATH=src python -m pytest -q
ruff check .
PYTHONPATH=src MPLBACKEND=Agg python experiments/delayed_horizon_sweep.py
PYTHONPATH=src MPLBACKEND=Agg python experiments/recovery_diagnostic.py
PYTHONPATH=src python experiments/turn_loo_fallback_audit.py
PYTHONPATH=src MPLBACKEND=Agg python experiments/variable_horizon_sweep.py
PYTHONPATH=src MPLBACKEND=Agg python experiments/monte_carlo_convergence.py
```

In compatible pinned framework environments, run their corresponding
`tests/test_*_integration.py` modules and:

```sh
# In the verl/TRL environment:
PYTHONPATH=src python experiments/verl_conformance.py
PYTHONPATH=src python experiments/cross_framework_conformance.py --out /tmp/cross_verl_trl.csv
# In the OpenRLHF CPU environment:
PYTHONPATH=src python experiments/cross_framework_conformance.py --out /tmp/cross_openrlhf.csv
# Merge only after each environment's checks pass:
PYTHONPATH=src python experiments/cross_framework_conformance.py --merge-input /tmp/cross_verl_trl.csv /tmp/cross_openrlhf.csv
```
