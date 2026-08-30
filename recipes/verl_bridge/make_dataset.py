"""Build train/val parquet files for the bridge agent loop.

The prompt column is a placeholder — CreditBenchBridgeLoop builds its own
messages from the environment — but verl's dataset pipeline requires one.
Each row selects the agent loop by name and carries the env in extra_info.

Usage:
    python make_dataset.py --env recovery --n-train 4096 --n-val 512 \
        --out-dir data/
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def rows(
    n: int,
    env: str,
    env_params: dict,
    parse_failure_policy: str = "minimum_return",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": f"Play {env}."}],
                "agent_name": "credit_bench_bridge",
                "data_source": "agent_credit_bench",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "env": env,
                    "env_params": env_params,
                    "parse_failure_policy": parse_failure_policy,
                },
            }
            for _ in range(n)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="recovery")
    parser.add_argument("--env-params", default="{}", help="JSON dict")
    parser.add_argument(
        "--parse-failure-policy",
        choices=("minimum_return", "raise"),
        default="minimum_return",
    )
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    env_params = json.loads(args.env_params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        path = args.out_dir / f"{args.env}_{split}.parquet"
        rows(n, args.env, env_params, args.parse_failure_policy).to_parquet(path)
        print(f"wrote {path} ({n} rows)")


if __name__ == "__main__":
    main()
