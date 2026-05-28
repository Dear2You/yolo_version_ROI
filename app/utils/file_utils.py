import os
import shutil
from pathlib import Path
from typing import Iterable, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def short_path(full_path: str, keep_parts: int = 5) -> str:
    if not full_path:
        return ""
    parts = Path(full_path).parts
    if len(parts) <= keep_parts:
        return full_path
    return str(Path("...").joinpath(*parts[-keep_parts:]))


def list_images(folder_path: str, supported_exts: Iterable[str]) -> List[str]:
    exts = {ext.lower() for ext in supported_exts}
    results = []
    for name in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, name)
        if os.path.isfile(full) and Path(name).suffix.lower() in exts:
            results.append(full)
    return results


def safe_copy(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
