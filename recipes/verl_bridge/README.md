# verl bridge: projected credit targets inside a real training run

This recipe runs the suite's environments *inside* real verl training, as
multi-turn chat games. The underlying MDP stays known, so real rollouts from
a real model can be reduced to a finite-sample tabular Markov projection,
`pi_hat(action | timestep, state)`. Dynamic programming then gives exact
values for that fitted projection, and every suite metric applies to the
logged sample without Monte Carlo value estimation. This is not exact ground
truth for a history-conditioned LLM policy: the projection collapses prompt
and conversation history and its action frequencies have sampling error.

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
  Markov projection, parsed-rate, and every estimator's credit quality (RMSE,
  centered RMSE, sign accuracy, leakage, Spearman) against exact advantages
  for the fitted projection. It rejects logs that mix environments or
  `env_params`. Core-suite only; verl estimator rows appear when verl is
  installed.

The framework-free machinery (verbalization, parsing, `BridgeSession`,
`EmpiricalTabularPolicy`) lives in the installed package at
`agent_credit_bench.integrations.bridge` — `play_episode` drives one episode
with any `messages -> reply` callable, so the same analysis works on
episodes from an API model or a scripted bot with no verl at all.

## What is exact, and what is projected

The text-to-action mapping is part of the environment, so every executed
action—including a fallback action—is known exactly in the log. The analyzer
counts those actions at each `(timestep, state)` and solves the resulting
tabular policy exactly. An LLM can nevertheless choose differently after two
histories that lead to the same MDP state. Those histories are merged by the
projection, so the resulting advantages are not exact values for the original
history-conditioned policy.

Unparseable replies use `parse_failure_policy="minimum_return"` by default.
Backward induction chooses the action with the lowest expected episode return
when later fallback choices also minimize return; declared action order breaks
exact ties. Thus malformed output in `RecoveryEnv` selects `BAD`, then
`GIVE_UP`, rather than receiving reward through the formerly favorable first
action. For strict evaluation, pass `parse_failure_policy="raise"` to
`play_episode` or `BridgeSession`; use
`make_dataset.py --parse-failure-policy raise` for the verl recipe. The chosen
action, parse status, and failure policy are logged, so the behavior is
reproducible.

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
        f.write(json.dumps(episode_record(s, env="recovery")) + "\n")
```

```bash
python analyze_checkpoint.py episodes.jsonl
```

Do not combine files produced with different `env_params`; the analyzer now
fails closed instead of solving all trajectories against the last record's MDP.

`episode_record` automatically records all constructor parameters for built-in
environments, including defaults. Custom environments require explicit
`extra={"env_params": {...}}` metadata. Older records with an explicit parameter
object (including `{}` for defaults) remain supported. Records with missing or
null `env_params` are rejected: restore the configuration from the original run
before analyzing them. Only use `{}` when that run used the environment defaults.

Before fitting the policy, the analyzer validates every episode against the
declared MDP: initial state, consecutive turns and states, legal actions,
positive-probability transition support, rewards, termination, and total return.
Invalid logs are rejected with an episode and, where applicable, turn index.
An episode is complete when it terminates or exhausts the environment horizon.
For the same checks in Python, call `trajectory_from_record(record, mdp=mdp)`.
Omitting `mdp` retains the parsing-only behavior for existing callers.

## Training run (template)

Requires a working verl 0.9.0 install with an inference backend
(`pip install "verl[vllm]"` or `[sglang]` — the backend also provides the
fastapi/uvicorn/cachetools stack the agent-loop path imports), plus
`python -m pip install -e .` from the AgentCreditBench source checkout in the
same environment.

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
checkpoint's policy" as the log grows. At least two episodes must remain after
windowing, so `N` must be at least 2. Smaller batches are rejected before
analysis because the group-relative estimators require peers. For
per-checkpoint precision, point
`CREDIT_BENCH_EPISODES_PATH` at a fresh file per eval, or slice the log by
line ranges.

## What to look for

- **Recovery env + GRPO:** verify the conditional diagnostic directly: BAD
  should receive positive credit within successful repaired episodes, while
  all-BAD aggregates and gradient metrics must be examined separately. The
  selected-path sign alone does not predict the direction of policy change.
- **Parsed-rate** should sit near 1.0 after the first few steps; a low rate
  means the projection contains substantial environment-selected fallback
  behavior rather than model-selected actions.
- **Estimator table on real data:** the tabular findings (RLOO ≡
  batch-centering, GRPO ≈ scaled version of it, GAE-with-critic sign
  behavior) can be checked on real LLM trajectories. Agreement supports those
  findings for the fitted Markov projection; it does not establish an exact
  oracle for history-conditioned behavior.

## Validation status

Truncated rollouts are omitted from episode logs and receive the environment's
lowest supported complete return as their training reward. This includes dense
rewards, early termination, and the worst positive-probability stochastic
outcome. It prevents a zero truncation reward from beating valid negative-return
episodes. `credit_bench_truncated` records the event, and discarded tokens never
execute an action. This is an explicit failure penalty, not a completed episode
return. Increase `response_length` if it occurs frequently.

Episode totals use `math.fsum`. Analysis also accepts legacy left-to-right sums
and differences within two ulps of the stable sum for cross-version log
compatibility; individual rewards still require exact MDP support.

Tested locally (no GPU): all bridge machinery (session, parsing,
minimum-return fallback, serialization, Markov projection —
`tests/test_bridge.py`), analyzer environment-spec validation, the analyzer in
both a bare and a verl environment, and that the agent loop imports and
registers under verl 0.9.0 exactly as `agent_loops.yaml` declares
(`tests/test_verl_integration.py`). The token bookkeeping mirrors verl's own
`tool_agent_loop` idiom (running sequence, mask 1 on generated tokens,
`remove_system_prompt` + turn-separator observation deltas, mask 0).

Not yet exercised: an end-to-end GPU training run (trainer-side reward
plumbing from `reward_score`, chat-template edge cases per model family,
response-length sizing). Expect the usual first-run friction there.
