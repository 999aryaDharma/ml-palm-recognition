"""
Path resolution utilities for PalmNet-Lite ML module.

Guarantees consistent path resolution regardless of Current Working Directory (CWD),
whether scripts are run from repository root, app/ml/, or subdirectories.
"""
from pathlib import Path

# __file__ = app/ml/palm_recognition/paths.py
ML_ROOT = Path(__file__).resolve().parent.parent              # app/ml
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent # repo root


def resolve_ml_path(path_str: str | Path) -> Path:
    """Resolve a path relative to app/ml or repo root.

    Behavior:
    - Absolute paths are returned as-is.
    - Relative paths starting with 'app/ml/' resolve relative to REPO_ROOT.
    - Relative paths starting with 'app/' resolve relative to REPO_ROOT.
    - Other relative paths resolve relative to ML_ROOT (app/ml).

    Args:
        path_str: Path string or Path object

    Returns:
        Resolved absolute Path object
    """
    if not path_str:
        return ML_ROOT

    p = Path(path_str)
    if p.is_absolute():
        return p

    p_str = str(path_str).replace("\\", "/")

    if p_str.startswith("app/ml/"):
        return (REPO_ROOT / p_str).resolve()
    elif p_str.startswith("app/"):
        return (REPO_ROOT / p_str).resolve()
    elif p_str.startswith("../"):
        return (ML_ROOT / p_str).resolve()
    else:
        return (ML_ROOT / p_str).resolve()
