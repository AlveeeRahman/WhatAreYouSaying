"""Provenance manifest, following the three-layer dependency view of Vangala, Adibifar, Gehani and
Malik (2026): claimed dependencies (what requirements.txt declares), runtime dependencies (the
distributions actually imported while the run executed) and the gap between them. Also records
what Sciunit-style capture (Ton That, Fils, Yuan and Malik, 2017) records for re-execution:
inputs by hash, code by commit, seeds, model revisions and hardware.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(repo: Path | None = None) -> str | None:
    repo = repo or Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def claimed_dependencies(req: Path | None = None) -> dict[str, str]:
    req = req or Path(__file__).resolve().parents[1] / "requirements.txt"
    out = {}
    if req.exists():
        for line in req.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-"):
                name = line.split("==")[0].split(">=")[0].split(" @")[0].strip()
                out[name.lower()] = line
    return out


def runtime_dependencies() -> dict[str, str]:
    """Distributions behind every top-level module currently imported (D_r in Vangala et al.)."""
    from importlib import metadata

    pkgs = metadata.packages_distributions()
    seen = {}
    for mod in list(sys.modules):
        top = mod.split(".")[0]
        for dist in pkgs.get(top, []):
            if dist not in seen:
                try:
                    seen[dist] = metadata.version(dist)
                except metadata.PackageNotFoundError:
                    seen[dist] = "?"
    return dict(sorted(seen.items(), key=lambda kv: kv[0].lower()))


def hf_snapshot(model_id: str) -> str | None:
    """Commit hash of the locally cached snapshot of a Hugging Face model (its revision)."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                revs = sorted(repo.revisions, key=lambda r: r.last_modified)
                return revs[-1].commit_hash
    except Exception:
        return None
    return None


def manifest(inputs: dict[str, str] | None = None, seeds: dict | None = None,
             models: list[str] | dict[str, str | None] | None = None, extra: dict | None = None) -> dict:
    """`models` is a list of Hub ids (the newest cached snapshot is recorded) or {id: pinned revision}."""
    import torch

    models = models if isinstance(models, dict) else {mid: None for mid in (models or [])}
    m = dict(
        code_commit=git_commit(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seeds=seeds or {},
        inputs={k: sha256(v) for k, v in (inputs or {}).items()},
        input_paths=dict(inputs or {}),
        models={mid: rev or hf_snapshot(mid) for mid, rev in models.items()},
        claimed_dependencies=claimed_dependencies(),
        runtime_dependencies=runtime_dependencies(),
    )
    claimed = set(m["claimed_dependencies"])
    runtime = {k.lower() for k in m["runtime_dependencies"]} - {"ways"}
    m["runtime_not_claimed"] = sorted(runtime - claimed)
    m["runtime_multiplier"] = round(len(runtime) / max(len(claimed), 1), 2)
    if extra:
        m.update(extra)
    return m


def write(path, **kw):
    m = manifest(**kw)
    Path(path).write_text(json.dumps(m, indent=1, default=str))
    return m
