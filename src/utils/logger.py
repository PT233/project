"""
logger.py — W&B experiment tracking wrapper.

Usage:
    from src.utils.logger import WandbLogger

    logger = WandbLogger(config={'lr': 1e-4, 'epochs': 30}, project='cancer-histo')
    logger.log({'loss': 0.5, 'epoch': 1})
    logger.finish()

Environment variables required:
    WANDB_API_KEY  — personal W&B API key (mandatory)
    WANDB_ENTITY   — team entity name (optional, falls back to W&B default)
"""

import logging
import os
import sys
from pathlib import Path

import wandb

# Module-level logger — error format: [ERROR][logger] <message>
_log = logging.getLogger("logger")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.DEBUG)
_log.propagate = False  # prevent double-printing via root logger


def _load_dotenv_if_present() -> None:
    """Load simple KEY=VALUE entries from .env without requiring python-dotenv."""
    project_root = Path(__file__).resolve().parents[2]
    for env_path in (Path.cwd() / ".env", project_root / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


class WandbLogger:
    """Thin wrapper around wandb.init / wandb.log / wandb.finish.

    Parameters
    ----------
    config : dict
        Hyper-parameters / experiment config to log to W&B.
    project : str
        W&B project name.
    """

    def __init__(self, config: dict, project: str) -> None:
        _load_dotenv_if_present()

        project_root = Path(__file__).resolve().parents[2]
        wandb_dir = Path(
            os.environ.get("WANDB_DIR", str(project_root / "artifacts" / "wandb"))
        )
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)

        # spec ④: WANDB_API_KEY must be set; exit(1) otherwise.
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            _log.error("WANDB_API_KEY not set")
            sys.exit(1)

        entity = os.environ.get("WANDB_ENTITY", None)

        self._run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            reinit=True,
        )

    def log(self, metrics: dict) -> None:
        """Log a dictionary of metrics to the current W&B run.

        Parameters
        ----------
        metrics : dict
            Key-value pairs to log (e.g. {'loss': 0.5, 'acc': 0.92}).
        """
        if self._run is None:
            _log.error("wandb run is not initialised; cannot log metrics")
            return
        wandb.log(metrics)

    def finish(self) -> None:
        """Mark the W&B run as finished and flush all pending data."""
        if self._run is not None:
            wandb.finish()
            self._run = None


# __main__ validation block — verifies the class can be instantiated and
# called without a live W&B connection (uses wandb offline mode).
