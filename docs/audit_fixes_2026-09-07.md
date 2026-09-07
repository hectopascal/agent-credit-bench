# Full-audit fixes — 7 September 2026

These working-tree changes address F1–F4 in the [full audit](full_audit_2026-09-07.md), based on commit `982bd15646b81315b066453ecd38e817e9e4258c`. The original report and its failing-test archive remain historical evidence.

## Changes

- **Verifiers rollout limits:** complete replies are applied in `add_trajectory_step`, before the framework checks turn and cumulative token caps. Token-truncated replies are not executed. Incomplete episodes receive the minimum supported complete return, and only completed episodes are logged. Actual-framework regressions cover completion at the cap, unfinished episodes, cumulative completion-token caps, truncated replies, and exactly one log entry.
- **verl observation limits:** trailing observation tokens are removed from returned responses, keeping the final reward on a generated token. Regressions cover observations that exactly fill and exceed the response budget, then run actual GAE and REINFORCE++ reward recursion.
- **Numerical consistency:** oracle and minimum-return backward induction preserve intermediate residuals with rational arithmetic on validated float inputs. Public oracle outputs are rounded independently and must fit in finite floats. Monte Carlo uses stable continuation sums and preserves sampled mean differences until credit is returned. Group baselines center around a reference before averaging; GiGPO preserves suffix-return residuals; RMSE scales errors before squaring. Regressions cover cancellation in all six reward orders, common offsets, group permutations, very small/large representable RMSE, and unrepresentable oracle output.
- **Recovery CLI:** an empty all-GOOD slice produces an unavailable CSV value instead of aborting. The CSV now includes `num_all_good` and `num_all_bad`. Plot generation uses the headless Agg backend.
- **Interpretation:** documentation distinguishes lambda-0 sign separation from intermediate-lambda behavior, one-step TD residuals from multistep GAE, and expected-gradient identities from finite-batch effects. The frontier is labeled empirical, with the independently enumerated ranking reversal disclosed in the README.
- **Replication:** duplicate seeds are rejected by `run_benchmark` to prevent repeated observations being presented as independent batches.

The documented Spearman zero sentinel for constant inputs is retained as an existing API convention. Changing undefined correlation to `None` is a separate API/reporting change, rather than part of the reproduced arithmetic failures. It remains explicitly identified in the audit.

## Validation

**All 291 distinct tests passed across compatible environments**, including 28 new regression cases:

| Environment/check | Result |
| --- | --- |
| Python 3.10 core suite before the last seven added cases | 238 passed; 46 optional skips |
| Final numerical/CLI regression module, Python 3.10 | 19 passed, including those seven additional cases |
| Final Python 3.12 full suite with verifiers 0.1.14 | 256 passed; 35 skips covered in other environments |
| Linux CPU, verl 0.9.0 and TRL 1.12.0 | 25 passed |
| Linux CPU, OpenRLHF 0.11.0 | 9 passed |
| Independent trajectory enumeration on 200 random MDPs | Largest gradient discrepancy approximately `6.11e-16` |
| Ruff over the complete workspace and `git diff --check` | Passed |
| Wheel/sdist build and built-wheel smoke with site-packages disabled | Passed; no core runtime dependencies |

OpenRLHF used the documented import-only CPU stubs; its advantage functions were real. Framework tests and experiment reruns used a separate source copy in the audit container. No GPU, distributed training, or actual model-server run was performed. Rational arithmetic increases computation relative to float-only recursion, particularly for larger custom MDPs.

## Published artifacts

All seven default experiment entry points were rerun, covering eight CSVs and 2,275 rows. [The complete comparison](../results/audit_fixes_2026-09-07/artifact-comparison.json) records every changed cell.

- Five CSVs remain byte-identical: delayed-horizon, Monte Carlo convergence, TurnLOO fallback, verl conformance, and cross-framework conformance.
- Recovery keeps every previous metric value and adds the two slice-count columns.
- Variable-horizon summaries change 31 Spearman means, with maximum absolute change `0.001850480288`, due to floating-point changes affecting rank ties. One relative mean-gradient error changes by approximately `1e-14`.
- Seven per-turn credit-bias values change by at most approximately `1e-16`.
- Every published normalized gradient-MSE value is unchanged at CSV precision, so the empirical frontier ordering is unchanged. Its plot title now explicitly says “Empirical”; the updated image was visually checked.

The changed CSVs and frontier image have been refreshed in `results/`. [Validation artifacts](../results/audit_fixes_2026-09-07/verification-summary.json) include test summaries, logs, source hashes, and regenerated tables. No commit or deployment was performed.
