# verl bridge: exact credit ground truth inside a real training run

This recipe runs the suite's environments *inside* real verl training, as
multi-turn chat games. The underlying MDP stays known, so real rollouts from
a real model remain exactly scoreable: measure the empirical policy the
model actually played, solve exact values under it, and every suite metric
applies to genuine training data — no Monte-Carlo approximation, no
simulated policies.

## Files

- `bridge_agent_loop.py` — a verl (0.9.0, experimental agent-loop API)
  `AgentLoopBase` subclass that plays any suite environment: renders states
  to chat, parses replies to actions, feeds the episode return back through
  `reward_score`, and logs finished episodes as JSONL.
- `agent_loops.yaml` — registry file verl loads via
  `actor_rollout_ref.rollout.agent.agent_loop_config_path`.
- `make_dataset.py` — builds the train/val parquet rows that select this
  agent loop (needs pandas + pyarrow, which a verl install provides).
- `analyze_checkpoint.py` — reads the episode log, reports the empirical
  policy, parsed-rate, and every estimator's credit quality (RMSE, centered
  RMSE, sign accuracy, leakage, Spearman) against exact advantages under
  the measured policy. Core-suite only; verl estimator rows appear when
  verl is installed.

The framework-free machinery (verbalization, parsing, `BridgeSession`,
`EmpiricalTabularPolicy`) lives in the installed package at
`agent_credit_bench.integrations.bridge` — `play_episode` drives one episode
with any `messages -> reply` callable, so the same analysis works on
episodes from an API model or a scripted bot with no verl at all.

## Why exactness survives an imperfect model

The text-to-action mapping (including the deterministic fallback for
unparseable replies) is part of the environment: whatever gets *executed*
defines the effective policy, and the oracle is exact for that policy.
`analyze_checkpoint.py` reports the parsed-rate; if it is low, the measured
policy is mostly the fallback and the prompt needs work, but the numbers
are still exact for what was actually played.

## Smoke test without a GPU

```python
import json, random
from agent_credit_bench.envs import RecoveryEnv
from agent_credit_bench.integrations.bridge import episode_record, play_episode

rng = random.Random(0)
bot = lambda messages: rng.choice(["I pick GOOD.", "BAD, then I will fix it.", "RECOVER"])
with open("episodes.jsonl", "w") as f:
    for _ in range(500):
        s = play_episode(RecoveryEnv(), bot)
        f.write(json.dumps(episode_record(s, env="recovery", extra={"env_params": {}})) + "\n")
```

```bash
python analyze_checkpoint.py episodes.jsonl
```

## Training run (template)

Requires a working verl 0.9.0 install with an inference backend
(`pip install "verl[vllm]"` or `[sglang]` — the backend also provides the
fastapi/uvicorn/cachetools stack the agent-loop path imports), plus
`pip install agent-credit-bench` in the same environment.

```bash
cd recipes/verl_bridge
python make_dataset.py --env recovery --n-train 4096 --n-val 512 --out-dir data/

export PYTHONPATH=$PWD:$PYTHONPATH
export CREDIT_BENCH_EPISODES_PATH=$PWD/episodes/run1.jsonl
mkdir -p episodes

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/recovery_train.parquet \
    data.val_files=data/recovery_val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$PWD/agent_loops.yaml \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.total_epochs=1
```

Treat the flag block as a template against verl 0.9.0's `ppo_trainer`
config — validate names against your verl install and hardware. A 1.5B
model on these 2–5-turn episodes is a small single-GPU run.

During or after training:

```bash
python analyze_checkpoint.py episodes/run1.jsonl --last 2000
```

`--last N` windows the newest episodes, approximating "the current
checkpoint's policy" as the log grows. For per-checkpoint precision, point
`CREDIT_BENCH_EPISODES_PATH` at a fresh file per eval, or slice the log by
line ranges.

## What to look for

- **Recovery env + GRPO:** the conformance suite predicts positive credit
  on BAD for 100% of repaired episodes. Watch the empirical
  `pi(BAD | s0)` trajectory across windows: broadcast credit predicts a
  slower fall (or transient rise) than an oracle-informed signal would give.
- **Parsed-rate** should sit near 1.0 after the first few steps; models
  learn the format fast, and the fallback keeps early noise well-defined.
- **Estimator table on real data:** the tabular findings (RLOO ≡
  batch-centering, GRPO ≈ scaled version of it, GAE-with-critic sign
  behavior) should reproduce on real LLM trajectories — that
  reproduction is the external-validity evidence this recipe exists for.

## Validation status

Tested locally (no GPU): all bridge machinery (session, parsing, fallback,
serialization, empirical policy — `tests/test_bridge.py`), the analyzer in
both a bare and a verl environment, and that the agent loop imports and
registers under verl 0.9.0 exactly as `agent_loops.yaml` declares
(`tests/test_verl_integration.py`). The token bookkeeping mirrors verl's own
`tool_agent_loop` idiom (running sequence, mask 1 on generated tokens,
`remove_system_prompt` + turn-separator observation deltas, mask 0).

Not yet exercised: an end-to-end GPU training run (trainer-side reward
plumbing from `reward_score`, chat-template edge cases per model family,
response-length sizing). Expect the usual first-run friction there.
