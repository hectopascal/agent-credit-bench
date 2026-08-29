"""Write import-time stubs for flash_attn and vllm into this environment.

OpenRLHF's import chain (``experience_maker`` -> ``ray.launcher`` ->
``models`` / ``vllm_engine``) imports flash-attn and vllm at module level,
and neither installs on a CPU-only machine. The advantage pipeline this
suite scores never *calls* either — flash-attn only backs ring attention,
vllm only the rollout engines — so minimal stubs that satisfy the imports
are enough for conformance runs. Every stubbed callable raises if actually
invoked.

For scoring only. Never use a stubbed environment for training.

Usage (after ``pip install openrlhf --no-deps`` on Linux x86_64)::

    python scripts/openrlhf_cpu_stubs.py
"""

import sysconfig
from pathlib import Path

STUBS = {
    "flash_attn/__init__.py": (
        "# stub written by agent-credit-bench scripts/openrlhf_cpu_stubs.py\n"
        "__version__ = '0.0.0+stub'\n"
    ),
    "flash_attn/bert_padding.py": (
        "def _unavailable(*args, **kwargs):\n"
        "    raise RuntimeError('flash_attn stub: not available on CPU')\n"
        "index_first_axis = pad_input = rearrange = unpad_input = _unavailable\n"
    ),
    "flash_attn/utils/__init__.py": "",
    "flash_attn/utils/distributed.py": (
        "def all_gather(*args, **kwargs):\n"
        "    raise RuntimeError('flash_attn stub: not available on CPU')\n"
    ),
    "vllm/__init__.py": (
        "# stub written by agent-credit-bench scripts/openrlhf_cpu_stubs.py\n"
        "__version__ = '0.0.0+stub'\n"
        "class SamplingParams:\n"
        "    pass\n"
        "class LLM:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise RuntimeError('vllm stub: not available on CPU')\n"
    ),
    "vllm/inputs.py": "class TokensPrompt(dict):\n    pass\n",
    "vllm/utils.py": (
        "import uuid\n\ndef random_uuid():\n    return str(uuid.uuid4())\n"
    ),
}


def main() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    for relative, content in STUBS.items():
        target = site_packages / relative
        if "stub" not in content and target.exists():
            continue  # never clobber a real install's submodule
        if (
            relative.endswith("__init__.py")
            and target.exists()
            and "stub" not in target.read_text()[:200]
        ):
            raise SystemExit(
                f"refusing to overwrite a real install at {target.parent}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    print(f"stubs written under {site_packages}")


if __name__ == "__main__":
    main()
