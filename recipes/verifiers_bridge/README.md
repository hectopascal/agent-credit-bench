# verifiers bridge: suite environments as a verifiers MultiTurnEnv

This recipe plays the suite's verbalized MDPs inside
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) (0.1.x) — the
environment library used by prime-rl and the Environments Hub. Same idea as
`recipes/verl_bridge`: the underlying MDP stays known, so real rollouts from
a real model can be fitted to a finite-sample Markov projection, and the
episode log feeds directly into `recipes/verl_bridge/analyze_checkpoint.py`.
Dynamic programming is exact for that fitted projection, not for a potentially
history-conditioned LLM policy.

## Files

- `credit_bench_env.py` — `CreditBenchEnv(vf.MultiTurnEnv)` plus the
  Environments-Hub `load_environment(**kwargs)` entry point. Each rollout
  opens a `BridgeSession`; `env_response` feeds the assistant's reply
  through `session.act` and returns the next observation; episode end uses
  verifiers' `final_env_response` idiom and appends the finished episode to
  a JSONL log (`episodes_path` argument or `CREDIT_BENCH_EPISODES_PATH`
  env var). The rubric reward is the episode's total return.

All environment mechanics (rendering, parsing, parse-failure handling,
transition sampling) live in the installed package at
`agent_credit_bench.integrations.bridge`, shared with the verl recipe. The
default `parse_failure_policy="minimum_return"` uses backward induction to
choose a deterministic lowest-return action; set it to `"raise"` for strict
evaluation. The executed action, parse status, and policy are logged.

## Quick eval

```bash
# From the AgentCreditBench source checkout:
python -m pip install -e ".[verifiers]"
```

```python
import sys; sys.path.insert(0, "recipes/verifiers_bridge")
from credit_bench_env import load_environment

env = load_environment(env="recovery", episodes_path="episodes/eval.jsonl")
results = env.evaluate_sync(client, "your-model", num_examples=100)
```

Any OpenAI-compatible client works (`vf-eval` CLI included). Afterwards:

```bash
python recipes/verl_bridge/analyze_checkpoint.py episodes/eval.jsonl
```

reports the empirical policy, parsed-rate, and every estimator's credit
quality against exact advantages for the fitted finite-sample Markov
projection. It rejects inputs that mix environments or `env_params`.
New logs include all environment parameters automatically. Legacy logs must
have an explicit `env_params` object; recover missing configurations from the
original run before analysis.
The analyzer also rejects episodes whose steps, rewards, completion, or recorded
total return disagree with the declared MDP.

## Training

Point a verifiers-compatible trainer (e.g. prime-rl) at the environment
module the usual way — package this directory as an environment (the
`load_environment` entry point is the whole contract) or install it on the
Environments Hub. Rollout seeds default to fresh entropy per episode;
`seed=...` exists for debugging only and warns, since a fixed seed replays
identical transition outcomes in every rollout of a group.

## Validation status

Tested end-to-end against verifiers 0.1.14's real `MultiTurnEnv` rollout
loop with a scripted client (`tests/test_verifiers_integration.py`): full
episodes, env-side termination, minimum-return parse fallback, episode
logging, and rubric scoring, with the logged JSONL round-tripping into suite
trajectories. Not yet exercised: a live training run through prime-rl.
