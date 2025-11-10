"""Gene regulatory network regression pipeline."""

try:  # pragma: no cover - support running without package install
    from .config import PipelineConfig, PathsConfig, TrainingConfig, ModelConfig
except ImportError:  # pragma: no cover
    from config import PipelineConfig, PathsConfig, TrainingConfig, ModelConfig  # type: ignore[no-redef]

def main(*args, **kwargs):  # pragma: no cover - thin wrapper for CLI entrypoint
    try:
        from .cli import main as _cli_main
    except ImportError:  # pragma: no cover - fallback when executed as flat scripts
        from cli import main as _cli_main  # type: ignore[attr-defined]

    return _cli_main(*args, **kwargs)

__all__ = [
    "PipelineConfig",
    "PathsConfig",
    "TrainingConfig",
    "ModelConfig",
    "main",
]
