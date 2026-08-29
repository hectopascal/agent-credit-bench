# verifiers bridge: suite environments as a verifiers MultiTurnEnv

This recipe plays the suite's verbalized MDPs inside
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) (0.1.x) — the
environment library used by prime-rl and the Environments Hub. Same idea as
`recipes/verl_bridge`: the underlying MDP stays known, so real rollouts from
a real model remain exactly scoreable, and the episode log feeds directly
into `recipes/verl_bridge/analyze_checkpoint.py`.

## Files

- `credit_bench_env.py` — `CreditBenchEnv(vf.MultiTurnEnv)` plus the
  Environments-Hub `load_environment(**kwargs)` entry point. Each rollout
  opens a `BridgeSession`; `env_response` feeds the assistant's reply
  through `session.act` and returns the next observation; episode end uses
  verifiers' `final_env_response` idiom and appends the finished episode to
  a JSONL log (`episodes_path` argument or `CREDIT_BENCH_EPISODES_PATH`
  env var). The rubric reward is the episode's total return.

All environment mechanics (rendering, parsing, the deterministic fallback
for unparseable replies, transition sampling) live in the installed package
at `agent_credit_bench.integrations.bridge`, shared with the verl recipe —
including the exactness-under-an-imperfect-model argument documented there.

## Quick eval

```bash
pip install "agent-credit-bench[verifiers]"
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
quality against exact advantages under the policy the model actually played.

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
episodes, env-side termination, fallback parsing, episode logging, and
rubric scoring, with the logged JSONL round-tripping into suite
trajectories. Not yet exercised: a live training run through prime-rl.
