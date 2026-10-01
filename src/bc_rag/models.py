"""FastEmbed model cache. Called from Python, not a FastEmbed CLI."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def cache_directory() -> Path:
    override = os.environ.get("FASTEMBED_CACHE_PATH") or os.environ.get("BC_RAG_MODELS")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "bc-rag" / "fastembed").resolve()


def cache_folder_name(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def _supported_rows() -> list[dict]:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    rows: list[dict] = []
    rows.extend(TextEmbedding.list_supported_models())
    rows.extend(SparseTextEmbedding.list_supported_models())
    rows.extend(TextCrossEncoder.list_supported_models())
    return rows


def huggingface_repo_ids(model_name: str) -> list[str]:
    """FastEmbed often caches the ONNX repo (xenova/...) not the Hub model id."""
    ids = [model_name]
    for row in _supported_rows():
        if str(row.get("model")) != model_name:
            continue
        sources = row.get("sources") or {}
        hf = sources.get("hf") if isinstance(sources, dict) else None
        if isinstance(hf, str) and hf and hf not in ids:
            ids.append(hf)
    return ids


def official_name_for_cache_folder(folder_name: str) -> str:
    slug = folder_name.removeprefix("models--").replace("--", "/", 1)
    for row in _supported_rows():
        name = str(row.get("model"))
        if cache_folder_name(name) == folder_name:
            return name
        sources = row.get("sources") or {}
        hf = sources.get("hf") if isinstance(sources, dict) else None
        if isinstance(hf, str) and cache_folder_name(hf) == folder_name:
            return name
    return slug


@dataclass(slots=True)
class ModelRecord:
    name: str
    kind: str
    cached: bool
    path: Path | None
    size_gb: float | None
    configured: bool


def list_known_models() -> dict[str, str]:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    known: dict[str, str] = {}
    for row in TextEmbedding.list_supported_models():
        known[str(row["model"])] = "dense"
    for row in SparseTextEmbedding.list_supported_models():
        known[str(row["model"])] = "sparse"
    for row in TextCrossEncoder.list_supported_models():
        known[str(row["model"])] = "rerank"
    return known


def configured_model_names(root: Path, *, local_only: bool = False) -> list[str]:
    """Model ids the project's spaces name. `local_only` keeps only FastEmbed models."""
    from bc_rag.config import load_config

    configuration, _ = load_config(root, groups=False)
    names: list[str] = []
    for spec in configuration.spaces.values():
        for model in (spec.dense, spec.sparse, spec.rerank):
            if model is None or (local_only and model.provider != "local"):
                continue
            if model.model not in names:
                names.append(model.model)
    return names


def is_api_model(model_name: str) -> bool:
    """True for Voyage and Jina API ids, which FastEmbed cannot download."""
    from bc_rag.voyage_api import is_voyage_model

    return is_voyage_model(model_name) or model_name.startswith(("rerank-", "jina-"))


def find_cached_path(model_name: str) -> Path | None:
    cache = cache_directory()
    if not cache.is_dir():
        return None
    folder_names = {cache_folder_name(repo) for repo in huggingface_repo_ids(model_name)}
    for folder_name in folder_names:
        exact = cache / folder_name
        if exact.is_dir():
            return exact
    needles = {repo.replace("/", "--") for repo in huggingface_repo_ids(model_name)}
    for child in cache.iterdir():
        if not child.is_dir():
            continue
        if any(needle in child.name for needle in needles):
            return child
    return None


def list_models(root: Path | None) -> list[ModelRecord]:
    """Cached models, plus the project's local models when `root` is given."""
    known = list_known_models()
    configured = set(configured_model_names(root, local_only=True)) if root is not None else set()
    names = list(configured)
    cache = cache_directory()
    if cache.is_dir():
        for child in sorted(cache.iterdir()):
            if not child.is_dir() or not child.name.startswith("models--"):
                continue
            pretty = official_name_for_cache_folder(child.name)
            if pretty not in names:
                names.append(pretty)
    records: list[ModelRecord] = []
    for name in names:
        cached_path = find_cached_path(name)
        records.append(
            ModelRecord(
                name=name,
                kind=known.get(name, "unknown"),
                cached=cached_path is not None,
                path=cached_path,
                size_gb=_dir_size_gb(cached_path) if cached_path else None,
                configured=name in configured,
            )
        )
    return records


def download_model(model_name: str) -> Path:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if is_api_model(model_name):
        raise ValueError(f"{model_name} is an API model. It runs remotely and is never downloaded.")
    cache = str(cache_directory())
    kind = list_known_models().get(model_name)
    if kind == "sparse":
        SparseTextEmbedding(model_name=model_name, cache_dir=cache)
    elif kind == "rerank":
        TextCrossEncoder(model_name=model_name, cache_dir=cache)
    else:
        TextEmbedding(model_name=model_name, cache_dir=cache)
    found = find_cached_path(model_name)
    if found is None:
        raise FileNotFoundError(
            f"downloaded {model_name} but cache folder was not found under {cache}"
        )
    return found


def remove_model(model_name: str) -> Path:
    found = find_cached_path(model_name)
    if found is None:
        raise FileNotFoundError(f"{model_name} is not in {cache_directory()}")
    shutil.rmtree(found)
    return found


def _dir_size_gb(path: Path) -> float:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024**3)
