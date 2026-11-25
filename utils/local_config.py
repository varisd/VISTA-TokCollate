"""
Load machine-local path configuration from local.env at the repo root.

local.env is not committed to git and must be created on each machine.
It must define DATA_ROOT, the base directory for training data, tokenized
data, and tokenizers on that machine.

Example local.env:
    DATA_ROOT=/path/to/your/data
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def _load_local_env() -> dict:
    env_file = _REPO_ROOT / "local.env"
    if env_file.exists():
        config = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
        return config

    return {}


def get_data_root() -> str:
    """Return DATA_ROOT from local.env, the environment, or the repository data directory."""
    cfg = _load_local_env()
    data_root = os.environ.get("DATA_ROOT") or cfg.get("DATA_ROOT")
    if data_root:
        return os.path.expandvars(os.path.expanduser(data_root))

    return str(_REPO_ROOT / "data")


def resolve_path(path: str) -> str:
    """Replace ${DATA_ROOT} placeholders and resolve environment variables."""
    resolved = os.path.expandvars(path.replace("${DATA_ROOT}", get_data_root()))
    return str(Path(resolved).expanduser())
