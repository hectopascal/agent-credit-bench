# Scoring TRL's advantage computations

`agent_credit_bench.integrations.trl` runs the conformance suite against the
advantage math of [TRL](https://github.com/huggingface/trl)'s `GRPOTrainer`
and `RLOOTrainer`.

```bash
pip install "agent-credit-bench[trl]"   # torch CPU is sufficient
python experiments/cross_framework_conformance.py
pytest tests/test_trl_integration.py
```

## Methodology: transcription + fingerprint

Unlike verl, TRL has no importable advantage function — the computation is
inline in each trainer's `_generate_and_score_completions`. The integration
therefore *transcribes* those lines (same ops, shapes, and epsilons), calls
TRL's real `nanstd` helper for every std, and pins the transcription with
**fingerprint tests** that assert the transcribed expressions still appear
verbatim in the installed TRL source. A TRL release that changes the math
fails the fingerprint instead of silently invalidating the numbers.
Transcribed from TRL 1.12.0.

Wrapped estimators: `TrlGRPO` (`scale_rewards` = `"group"` — TRL's default —
`"batch"`, or `"none"`, the Dr. GRPO recommendation) and `TrlRLOO`
(`normalize_advantages=False` by default, per TRL). The suite batch is one
group, matching the verl integration's semantics; there `"batch"` and
`"group"` scaling coincide.

## Findings

1. **TRL's RLOO is bit-exactly verl's RLOO.** The default `RLOOTrainer`
   advantage — leave-one-out baseline, no normalization — matches the
   suite's `BatchCenteredBroadcast` (and therefore verl's
   `compute_rloo_outcome_advantage`) to 1e-9. Two frameworks, one estimator.

2. **TRL's GRPO agrees with verl on the std but not the epsilon.** Both
   divide by the Bessel-corrected sample std; TRL adds 1e-4 where verl adds
   1e-6 (and OpenRLHF 1e-9). On the recovery diagnostic that is a 0.5864 vs
   0.5863 difference — irrelevant for training, but exactly the kind of
   implementation drift a conformance suite should pin down.

3. **`nanstd` computes its Bessel correction in float32.** The
   `count / (count - 1)` factor divides two integer tensors, which torch
   promotes to float32, so even float64 rewards carry ~1e-8 error. Harmless,
   but it bounds how tightly TRL numbers can be compared across frameworks.

4. **Every TRL outcome estimator praises repaired mistakes.** GRPO, Dr.
   GRPO (`scale_rewards="none"`), and RLOO all assign positive credit to the
   BAD action on 100% of recovered trajectories — the suite's headline
   failure, reproduced on a third implementation.

## What this does and does not capture

It scores the advantage assignment — the map from rewards to per-completion
credit. Token-level loss shaping, importance ratios, TRL's NaN handling for
unscorable completions (which never triggers here), and everything else
downstream are out of scope. See `docs/verl_integration.md` for the shared
packing semantics and `experiments/cross_framework_conformance.py` for the
combined table.
