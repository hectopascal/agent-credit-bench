# Scoring OpenRLHF's advantage pipeline

`agent_credit_bench.integrations.openrlhf` runs the conformance suite
against [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)'s advantage
computation — and unlike the verl and TRL integrations, it calls the *whole
pipeline*: `RemoteExperienceMaker.compute_advantages_and_returns`, unbound
on a stubbed trainer, so the group baselines, reward placement, GAE /
cumulative-return recursion, and batch whitening are all OpenRLHF's released
code end to end.

Wrapped estimators: `OpenRLHFOutcome(estimator)` for `rloo`, `group_norm`
(their GRPO), `dr_grpo`, `reinforce_baseline` (their REINFORCE++-baseline),
and `reinforce`; `OpenRLHFGAE(gamma, lam, critic)` with the same controlled
critic as `VerlGAE`. Tested against openrlhf 0.11.0.

## Installing (Linux x86_64 only)

OpenRLHF ships Linux-x86_64 wheels only, and a full install drags in vllm
and flash-attn, which need CUDA. For CPU conformance scoring:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
pip install openrlhf --no-deps
pip install agent-credit-bench ray transformers peft deepspeed pylatexenc torchdata
python scripts/openrlhf_cpu_stubs.py   # import-time stubs for flash_attn/vllm
pytest tests/test_openrlhf_integration.py
```

The stubs satisfy module-level imports the pipeline never calls (flash-attn
backs ring attention, vllm the rollout engines); every stubbed callable
raises if actually invoked. Scoring only — never train in a stubbed
environment. On macOS, run the same recipe in a `--platform linux/amd64`
container; CI runs it natively (`.github/workflows/ci.yml`,
`openrlhf-integration`).

## Packing semantics

OpenRLHF's estimator interface consumes **one scalar reward per sequence**
(`Experience.rewards` is `(B,)`; `compute_reward` scatters it onto the last
unmasked cell — intermediate rewards only ever come from the KL term). Each
trajectory therefore contributes its total return, placed on its final
turn, with one tensor cell per turn. For every suite environment this
coincides with the per-turn reward structure, because suite rewards are
only emitted on the episode-ending turn. KL is zero, length penalties are
disabled, and the whole batch is one group.

## Findings

Recovery diagnostic (`experiments/cross_framework_conformance.py`,
batch 2000, exact advantages −0.25 / +0.50):

| estimator                   | credit(BAD) | credit(RECOVER) | praises both |
| --------------------------- | ----------- | --------------- | ------------ |
| oracle_advantage            | −0.25       | +0.50           | 0%           |
| openrlhf_group_norm         | +0.59       | +0.59           | 100%         |
| openrlhf_rloo               | +0.26       | +0.26           | 100%         |
| openrlhf_reinforce_baseline | +0.72       | +0.72           | 100%         |
| openrlhf_gae_exact_lam1     | +0.56       | +1.10           | 100%         |
| openrlhf_gae_exact_lam0     | −0.69       | +1.42           | 0%           |

1. **The verl findings replicate on OpenRLHF almost digit for digit.**
   `openrlhf_rloo` matches `BatchCenteredBroadcast` (and hence verl's and
   TRL's RLOO) to 1e-9; `group_norm` reproduces `verl_grpo`'s 0.5864;
   `reinforce_baseline` lands on verl's REINFORCE++ 0.718; the two GAE rows
   match verl's to three decimals. Three frameworks, one failure pattern.

2. **A perfect critic still praises repaired mistakes at OpenRLHF's
   default λ.** `--algo.advantage.lambd` defaults to 1 — same blind spot,
   same default, as verl. At λ=0 the same critic separates the signs.

3. **The epsilon lottery, third draw.** OpenRLHF's group_norm divides by
   `torch.std(-1) + 1e-9` (Bessel-corrected). verl: 1e-6. TRL: 1e-4. Same
   published formula, three epsilons.

4. **Whitening is selective and float32.** The pipeline batch-whitens
   `gae`, `reinforce`, and `reinforce_baseline` but *not* `rloo`,
   `group_norm`, or `dr_grpo` — and casts to float32 to compute the
   mean/rstd, which bounds cross-framework comparisons of whitened
   estimators at ~1e-5.

5. **Turn-level rewards do not exist in this interface.** The pipeline's
   reward input is one scalar per sequence; anything turn-level must
   already be folded into it. The suite's environments happen to be
   representable exactly (rewards arrive on terminal turns), but the
   interface itself is a bandit-shaped contract — worth knowing before
   reading any OpenRLHF estimator as "turn-level credit assignment."

## What this does and does not capture

Everything downstream of the advantage tensor (PPO clipping, KL penalties,
optimizer dynamics) and critic *learning* are out of scope, as in the other
integrations. The stubbed-self call covers the advantage pipeline itself:
`compute_advantages_and_returns` plus its two helpers, with length
penalties and KL shaping disabled through their own configuration paths.
