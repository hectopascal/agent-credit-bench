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
python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install openrlhf==0.11.0 --no-deps
python -m pip install -r requirements/openrlhf-cpu.txt
python -m pip install -e .   # from the AgentCreditBench source checkout
python scripts/openrlhf_cpu_stubs.py   # import-time stubs for flash_attn/vllm
pytest tests/test_openrlhf_integration.py
```

The requirements file is the version-aligned CPU smoke-test recipe used by
CI. It pins the framework-specific constraints that OpenRLHF declares exactly;
transitive dependencies can still move. A short unpinned `pip install ray
transformers peft deepspeed ...` command is already known to violate
OpenRLHF 0.11.0's constraints.

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
turn, with one tensor cell per turn. This exactly represents the default
benchmark configurations, whose rewards are only emitted on the
episode-ending turn. KL is zero, length penalties are disabled, and the
whole batch is one group.

For recursive estimators (`reinforce` and GAE), the integration rejects a
trajectory with a nonzero reward before its final turn: moving such a
reward to the last cell would change its return-to-go and would not be a
faithful conformance test. The group-relative outcome estimators remain valid
because their contract intentionally reduces the sequence to its total
return. As with the verl adapter, `OpenRLHFGAE(critic="exact")` requires
`gamma=1.0`, matching the suite's undiscounted exact-value oracle;
`critic="zero"` supports other gamma values.

## Findings

Recovery diagnostic (`experiments/cross_framework_conformance.py`, batch
2000, seed 0, recovery-success probability 1.0, exact advantages −0.25 /
+0.50):

| estimator                   | credit(BAD) | credit(RECOVER) | both positive on selected path |
| --------------------------- | ----------- | --------------- | ------------ |
| oracle_advantage            | −0.25       | +0.50           | 0%           |
| openrlhf_group_norm         | +0.59       | +0.59           | 100%         |
| openrlhf_rloo               | +0.26       | +0.26           | 100%         |
| openrlhf_reinforce_baseline | +0.72       | +0.72           | 100%         |
| openrlhf_gae_exact_lam1     | +0.56       | +1.10           | 100%         |
| openrlhf_gae_exact_lam0     | −0.69       | +1.42           | 0%           |

1. **The selected-path pattern replicates on OpenRLHF, with a documented
   normalization delta.**
   For groups of at least two, `openrlhf_rloo` matches
   `BatchCenteredBroadcast` (and hence verl's and TRL's RLOO) to 1e-9;
   `group_norm` reproduces `verl_grpo`'s 0.5864;
   `reinforce_baseline` lands on verl's REINFORCE++ 0.718; the two GAE rows
   match verl's to three decimals. Whitened rows are not numerically identical
   because OpenRLHF uses population-standard-deviation normalization while
   verl uses the sample convention.
   These values are conditioned on successful repairs and show the same
   identification pattern; they are not estimates of each method's expected
   gradient.

2. **An exact critic still gives selected BAD positive credit at OpenRLHF's
   default λ.** `--algo.advantage.lambd` defaults to 1. At λ=0 the same
   critic separates the two selected turns' signs. This is again a
   conditional credit-value result, not an expected-gradient claim.

3. **The epsilon lottery, third draw.** OpenRLHF's group_norm divides by
   `torch.std(-1) + 1e-9` (Bessel-corrected). verl: 1e-6. TRL: 1e-4. Same
   published formula, three epsilons.

4. **Whitening is selective, population-normalized, and float32.** The
   pipeline batch-whitens
   `gae`, `reinforce`, and `reinforce_baseline` but *not* `rloo`,
   `group_norm`, or `dr_grpo`. Its whitening helper divides by the population
   standard deviation, while verl's `masked_whiten` applies Bessel's
   correction. This convention creates the systematic roughly 1e-4 gaps in the committed
   recovery CSV (for example 0.718059 vs 0.717940 for the baseline); the
   float32 mean/rstd computation adds smaller rounding differences.

5. **There is no turn-level environment-reward input.** The pipeline's
   environment/reward-model input is one scalar per sequence; anything turn-level must
   already be folded into it. The default benchmark configurations are
   representable exactly because their rewards arrive on terminal turns, but
   parameterizations such as `VariableHorizonEnv(continue_reward != 0)` are
   rejected for recursive estimators. The interface itself is a bandit-shaped
   contract — worth knowing before
   reading any OpenRLHF estimator as "turn-level credit assignment."

## What this does and does not capture

Everything downstream of the advantage tensor (PPO clipping, KL penalties,
optimizer dynamics) and critic *learning* are out of scope, as in the other
integrations. The stubbed-self call covers the advantage pipeline itself:
`compute_advantages_and_returns` plus its two helpers, with length
penalties and KL shaping disabled through their own configuration paths.
