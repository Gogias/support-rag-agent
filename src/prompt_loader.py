from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"
SHARED = yaml.safe_load((PROMPTS_DIR / "shared.yaml").read_text(encoding="utf-8"))


def load_prompt(name: str) -> dict:
    p = yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    p["system"] = "\n\n".join([*(SHARED[k] for k in p.get("uses", [])), p["system"]])
    return p
