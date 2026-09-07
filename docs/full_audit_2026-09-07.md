# Full repository audit — 7 September 2026

Audited revision: `982bd15646b81315b066453ecd38e817e9e4258c` (version 0.5.0).

**Follow-up:** the findings below describe that revision. Fixes and their
validation are recorded in [the remediation report](audit_fixes_2026-09-07.md).

The published default experiments reproduce, and all 263 existing tests pass across compatible environments. The repository nevertheless has reproducible correctness failures at rollout-budget boundaries, under extreme but accepted finite reward scales, and in one small-batch experiment path. Some explanatory claims also need narrower wording. These findings are consolidated by failure family; the number of findings was not a stopping criterion.

This audit changed no implementation files. New audit documents and `results/claims_audit_2026-09-07/` appeared concurrently; they are outside the fixed source snapshot. Their candidate observations were independently checked where included below. This report and its evidence archive are the deliverables from this full audit.

## Findings requiring changes

### F1 — P2: Rollout limits can reward incomplete episodes in both bridges

**Verifiers:** [episode_return](../recipes/verifiers_bridge/credit_bench_env.py) at lines 70–72 returns the session's partial return. Actions are applied later in `env_response`, at lines 105–109. In the actual pinned verifiers 0.1.14 rollout loop, reaching the turn cap can stop processing after a model response is added but before this callback processes it.

Reproduction: `VariableHorizonEnv(stop_rewards=(-1, -1), continue_reward=0)`, scripted replies `CONTINUE`, then `STOP`, and `max_turns=2`. Only `CONTINUE` is applied; the session is incomplete, no complete episode is logged, and the rubric gives **0**. Every legal complete episode returns **−1**. With `max_turns=3`, both replies are applied and the rubric gives −1. At `max_turns=1`, even a first-response `STOP` is omitted.

This makes truncation better than every legal completion and drops a valid final action. The default horizon-plus-two cap avoids this particular boundary during ordinary successful play; explicitly lowered caps expose it.

Recommended fix: finalize a pending assistant action exactly once when the framework stops, then give genuinely incomplete episodes the supported minimum complete return. Cover actual framework stop ordering, complete-at-cap episodes, and incomplete episodes in regression tests.

**verl:** [CreditBenchBridgeLoop.run](../recipes/verl_bridge/bridge_agent_loop.py), lines 106–126, can append an observation that fills the response budget and return a sequence ending in mask-zero observation tokens. The loop supplies the intended scalar truncation penalty, but the pinned worker places that reward on the last response cell. The recursive GAE and REINFORCE++ implementations skip masked observation cells, dropping this reward.

Reproduction with the actual bridge class and actual verl 0.9.0 advantage functions, using a scripted generator/tokenizer:

| Rollout | Returned IDs | Response mask | Scalar reward | Generated-token return |
| --- | --- | --- | --- | --- |
| `CONTINUE`, observation exhausts budget | `[1, 201, 202]` | `[1, 0, 0]` | −1 | **0** |
| Complete `STOP` | `[1]` | `[1]` | −1 | −1 |

With these two rows in one batch, both GAE and REINFORCE++ give the truncated row approximately **+0.7071** advantage and the completed row **−0.7071**. Thus the penalty is present in the bridge output but lost downstream. This specific failure concerns recursive estimators; default GRPO sums the row's rewards and does not lose the scalar this way.

Recommended fix: ensure budget-truncated output cannot end with observations after the last generated token, or use a supported reward-placement path targeting the last live token. Add an observation-boundary regression through the real advantage functions. The existing overlong-generated-reply test covers a different boundary.

Evidence: `test_full_regressions.py`, `verifiers-regression.txt`, `framework_probes.py`, and `framework-probe-observations.txt` in the archive.

### F2 — P3: Numerical robustness is inconsistent across accepted finite inputs

The earlier common-offset oracle fix is effective for the tested one-step bandit, but it does not establish consistent arithmetic across the pipeline. These are one numerical failure family with several affected components. The default bounded-reward studies do not exhibit these failures.

| Component and location | Valid finite input | Expected result | Observed result |
| --- | --- | --- | --- |
| [Oracle](../src/agent_credit_bench/oracle.py), lines 55–63 | Uniform root choice: `CHAIN` with deterministic rewards `(1e16, 1, -1e16)`, or `QUIT` returning 0 | Stable CHAIN return 1; root softmax gradient `(+0.25, −0.25)` | Root advantages and gradient are zero |
| [Monte Carlo rollout](../src/agent_credit_bench/estimators/monte_carlo.py), line 59 | Same structure, CHAIN rewards `(1, 1e16, -1e16)` | Root credits approach `(+0.5, −0.5)` | Both remain zero, including at 1,024 rollouts |
| [BatchCenteredBroadcast](../src/agent_credit_bench/estimators/batch_centered.py), lines 31–36, and [TurnLOO](../src/agent_credit_bench/estimators/turn_loo.py), lines 33–49 | Two one-step returns `1e16` and `1e16 + 2` | Leave-one-out credits `(−2, +2)` | `(0, +4)` |
| [GRPO-style](../src/agent_credit_bench/estimators/grpo_style.py) and [GiGPO-style](../src/agent_credit_bench/estimators/gigpo_style.py) group arithmetic | Same two returns | Symmetric centered credits under their respective formulas | `(0, 1.99980002)` and `(0, 3.99960004)` |
| [RMSE](../src/agent_credit_bench/metrics.py), lines 26–32 | Errors `(+s, −s)`, with `s=1e-170` or `1e160` | RMSE equals the representable value `s` | Zero or `OverflowError`, respectively |
| [Minimum-return dynamic program](../src/agent_credit_bench/integrations/bridge.py), lines 164–170 | CHAIN rewards `(-1e16, -1, 1e16)`, or QUIT returning 0 | Minimum supported complete return −1 | Returns 0 |

The oracle and minimum-return failures round away a residual in an intermediate `reward + future` operation. Accurate summation of already-rounded transition contributions cannot recover it. Monte Carlo uses ordinary sequential addition, whereas `Trajectory.total_return` uses stable summation. Increasing sampling effort therefore cannot cure the demonstrated discrepancy. Group means similarly lose a representable action difference when formed around a large common offset. RMSE squares before scaling, losing or overflowing a representable final magnitude.

Recommended fix: decide and enforce a numerical contract. Within that contract, use centered group statistics, stable continuation sums, scaled norms, and an approach that preserves required residuals through Bellman recursion. Alternatively, explicitly reject unsupported dynamic ranges. Compare components against rational/high-precision references on the same representable inputs, including reward-order permutations. A local replacement of one `sum` with `math.fsum` is insufficient for the whole family.

Evidence: `test_full_regressions.py`, `core-regressions.txt`, `boundary_probes.py`, and `boundary-results.txt`. The oracle and Monte Carlo fixtures deliberately use different reward orders to isolate each fault.

### F3 — P3: The recovery experiment crashes when the all-GOOD slice is empty

[recovery_diagnostic.py](../experiments/recovery_diagnostic.py), line 131, unconditionally calls `fmean` on the all-GOOD subset.

Run:

```sh
PYTHONPATH=src .venv/bin/python experiments/recovery_diagnostic.py \
  --batch-size 2 --num-seeds 1 --seed 0 \
  --out /tmp/recovery-audit.png --csv-out /tmp/recovery-audit.csv
```

Both sampled episodes start with BAD, and the batch contains a successful recovery, so the primary selected-recovery diagnostic is available. Nevertheless, the unrelated empty all-GOOD slice raises `StatisticsError` and aborts CSV production. This is distinct from intentionally rejecting a batch with no successful recovery. The default batch size does not fail.

Recommended fix: emit an unavailable value for this empty aggregate, retaining its sample count, and allow the available diagnostics to be exported. Evidence: the CLI regression in `test_full_regressions.py` and its traceback in `core-regressions.txt`.

### F4 — P3: GAE and baseline-shift explanations overstate what is guaranteed

The [VerlGAE docstring](../src/agent_credit_bench/integrations/verl.py), lines 214–216, suggests that choosing `lam < 1` separates the selected BAD and RECOVER turns. The useful tested endpoint is `lam=0`; simply reducing lambda below 1 does not guarantee negative BAD credit. On the default recovery batch, actual pinned verl produces:

| Lambda | Selected BAD credit | Selected RECOVER credit |
| --- | --- | --- |
| 0.49 | +0.000330232 | +1.321303492 |
| 0.75 | +0.313855828 | +1.214147483 |

Before whitening, selected BAD credit is `−0.25 + 0.5 * lambda`; batch whitening also shifts its sign threshold. State the endpoint result and conditions rather than a blanket sign-separation claim.

[docs/verl_integration.md](verl_integration.md), lines 110–115, also conflates zero immediate TD residuals with zero multistep GAE. On delayed-effect horizon 4, batch size 200, seed 0, intermediate-turn credits are one constant at lambda 0, but range from approximately **−1.8909 to +1.8992** at lambda 1. Later TD residuals propagate backward.

The same paragraph says gradient-validity metrics ignore a shared credit shift. A fixed action-independent baseline preserves the expected score-function gradient, but can change finite-batch gradients and their variance. In an independent horizon-8 delayed-effect probe over 30 seeds and batches of 200, adding 1 to oracle credits preserves zero centered RMSE but changes gradient variance from **0 to 0.01949258**, and relative empirical mean-gradient error from approximately zero to **0.0934820**. Batch-dependent whitening additionally changes scale and centering.

Recommended fix: distinguish expectation identities, finite-batch metrics, and batch-dependent normalization. The reproduced lambda-0/lambda-1 endpoint tables remain valid. Evidence: `gae_claims.py`, `claims_probes.py`, and `claims-results.json`.

## Research-reporting qualifications

The frontier is an empirical comparison, and a sampled winner need not minimize expected error. I independently enumerated all **330 multinomial batch compositions** for horizon 8, stop probability 0.65, batch size 4, accounting for all ordered-batch probability mass:

| Estimator | Published 200-seed normalized MSE | Expected normalized MSE by exhaustive enumeration |
| --- | --- | --- |
| BatchCenteredBroadcast | 0.723659508499 | **0.752148293603** |
| TurnLOO | **0.715446837666** | 0.784549437128 |

The ordering reverses. This is sampling uncertainty, not a CSV arithmetic defect. Label the frontier's winners as empirical and provide paired uncertainty or exact expectations where feasible. This audit independently established the reversal for this cell; it did not enumerate every frontier cell.

Two additional API/reporting choices deserve explicit treatment, without presenting them as hidden arithmetic bugs: `run_benchmark` accepts duplicate seeds despite describing independently seeded batches, and `spearman` deliberately returns zero for undefined constant-input correlation. Unique-seed validation or reporting would prevent repeated observations masquerading as replication; an explicit unavailable correlation value would distinguish undefined from defined zero. Default studies use unique seeds. These behaviors are visible in the current source, and the latter is expressly documented.

## Coverage and verification

| Surface | Completed work | Outcome |
| --- | --- | --- |
| Source | Reviewed all 41 production, integration, recipe, experiment, and script Python files; recorded SHA-256 inventory | Fixed revision identified above; all 62 entries in the committed revalidation source manifest match |
| Core suite, Python 3.10 | Full suite | 226 passed, 37 optional skips |
| Python 3.12 plus verifiers 0.1.14 | Full suite | 232 passed, 31 optional skips |
| Clean Linux archive, Python 3.11, verl 0.9.0 and TRL 1.12.0 | Full suite | 248 passed, 15 optional skips |
| OpenRLHF 0.11.0, compatible Linux environment | Dedicated integration suite | 9 passed |
| Combined existing-test coverage | Reconciled framework-specific tests across environments | **All 263 distinct existing tests passed; no optional test remained unexecuted** |
| Independent mathematical checks | Existing 3,000 random-MDP rational-oracle cases and exhaustive small-batch tests; additional 200 random-MDP trajectory enumerations | Additional enumeration worst absolute gradient difference approximately `6.66e-16` |
| Framework probes | 100 randomized dense-reward variable-length verl/TRL batches; 100 randomized terminal-reward OpenRLHF batches | Formula/masking checks passed at specified tolerances; bridge budget defect separately reproduced |
| Dependency provenance | Compared nine relevant installed framework source files against wheel RECORD hashes | All matched; advantage routines were not patched |
| Input boundaries | Estimator output alignment/nonfinite probes and ten corrupted-log mutations | Rejected as intended |
| Experiments | Reran all seven default entry points and merged framework outputs | Eight CSVs, **2,275 rows, byte-identical**, zero changed cells |
| Packaging and lint | Ruff; wheel and sdist build; isolated wheel smoke with site-packages disabled | Passed; core smoke required no runtime dependency |
| Site and operations | Static assembled-site local links/anchors, JSON-LD, workflow/configuration and shell-script review | Static checks passed; no deployment performed |

The default CSV counts are 250 delayed-horizon rows, 150 recovery rows, 9 TurnLOO fallback rows, 270 variable-horizon summary rows, 1,395 per-turn rows, 180 Monte Carlo rows, 8 verl rows, and 13 cross-framework rows. Detailed comparison is in `artifact-comparison.json`.

Previously reported failures involving fixed Monte Carlo continuation randomness, acceptance of an impossible logged recovery reward, and one-step oracle loss of a large-offset action gap were retested and are fixed. Those fixes do not invalidate F2's distinct multistep and estimator arithmetic reproductions.

OpenRLHF ran with the repository's documented import-only CPU stubs for unavailable GPU packages. Its real advantage functions executed. Small constant-group normalization residuals originate in the upstream floating-point computation; the observed maximum was approximately `2.22e-7`, and they were not counted as a separate repository defect. Linux core/verl/TRL tests used a clean `git archive` snapshot to avoid stale bytecode or editable-install contamination.

## Evidence and reproduction

[Download the evidence archive](../results/full_audit_2026-09-07/evidence.zip). It contains the adversarial probes, failing expectations, test logs, dependency checks, source inventory, regenerated CSVs, and artifact comparison. The seven new pytest cases are **expected to fail on the audited revision**: six core cases and one verifiers case. They are kept outside the regular test suite until fixes are made.

From the repository root, extract into a fresh temporary directory and run the core regressions:

```sh
unzip results/full_audit_2026-09-07/evidence.zip -d /tmp/acb-audit-evidence
PYTHONPATH=src:tests .venv/bin/python -m pytest -q \
  /tmp/acb-audit-evidence/test_full_regressions.py -k 'not verifiers'
```

Use an environment with verifiers 0.1.14 for its regression:

```sh
PYTHONPATH=src:tests:recipes/verifiers_bridge python -m pytest -q \
  /tmp/acb-audit-evidence/test_full_regressions.py -k verifiers
```

Run `framework_probes.py` and `gae_claims.py` from the repository root with `PYTHONPATH=src` in the pinned verl/TRL environment; run `openrlhf_probes.py` in the documented OpenRLHF CPU environment. The framework probe prints the erroneous bridge returns as well as checking independent randomized formulas. `claims_probes.py` writes its result to `/tmp/agent-credit-bench-full-audit`, which must exist before running it.

## Limits and disposition

This is a completed repository audit with broad source and executable coverage, not a proof that no undiscovered defects exist. It did not execute GPU/distributed training, a real model server, a full optimizer run, deployment, or high-contention multi-process logging. The bridge reproduction uses scripted generation with the actual framework algorithms. Site checks cover structure and artifact availability, not a browser visual review. Dependency provenance checks cover relevant installed files, not a vulnerability-database audit of every transitive package. No new external research claims are inferred from CPU conformance alone.

Prioritize F1 before relying on capped rollouts in training; repair the F2 numerical contract and F3 CLI failure; narrow F4 and qualify sampled frontier rankings before extending research conclusions. The default benchmark tables are reproducible at the audited revision, while these boundary and interpretation issues remain open.
