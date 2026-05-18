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

import wandb

# ---------------------------------------------------------------------------
# Module-level logger — error format: [ERROR][logger] <message>
# ---------------------------------------------------------------------------
_log = logging.getLogger("logger")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.DEBUG)
_log.propagate = False  # prevent double-printing via root logger


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
        # DoD ④: WANDB_API_KEY must be set; exit(1) otherwise.
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


# ---------------------------------------------------------------------------
# __main__ validation block — verifies the class can be instantiated and
# called without a live W&B connection (uses wandb offline mode).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # Check for WANDB_API_KEY before attempting anything
    if not os.environ.get("WANDB_API_KEY"):
        # Simulate missing key scenario to show correct error format, then exit
        _log.error("WANDB_API_KEY not set")
        sys.exit(1)

    # Run in offline mode so no real network call is made
    os.environ.setdefault("WANDB_MODE", "offline")

    print("Initialising WandbLogger in offline mode …")
    logger = WandbLogger(config={"lr": 1e-4, "epochs": 1}, project="test")
    print("WandbLogger initialised successfully.")

    logger.log({"loss": 0.5})
    print("log({'loss': 0.5}) — OK")

    logger.finish()
    print("finish() — OK")

    print("All DoD checks passed.")
