from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import ollama
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize


DEFAULT_INPUT = Path("data/sample-data.json")
DEFAULT_CACHE = Path("data/message_body_embeddings_qwen3.npz")
DEFAULT_OUTPUT = Path("reports/message_body_hierarchy.md")

REACTION_WORDS = {
    "lol",
    "lmao",
    "lmfao",
    "damn",
    "damnn",
    "rip",
    "oof",
    "bruh",
    "real",
    "true",
    "fr",
}
THANKS_WORDS = {"thanks", "thank", "thx", "ty"}
YES_WORDS = {"yes", "yeah", "yep", "yea", "yup", "sure", "ok", "okay"}
NO_WORDS = {"no", "nah", "nope", "not"}


@dataclass(frozen=True)
class BodyRecord:
    index: int
    body: str


def normalize_short_text(body: str) -> str:
    return re.sub(r"\s+", " ", body.strip().lower())


def is_mostly_punctuation(body: str) -> bool:
    text = body.strip()
    if not text:
        return True
    punctuation_count = sum(not char.isalnum() for char in text)
    return punctuation_count / max(len(text), 1) > 0.7


def short_bucket(body: str) -> str | None:
    text = normalize_short_text(body)
    words = re.findall(r"[\w']+", text)

    if is_mostly_punctuation(text):
        return "punctuation_or_symbols"
    if len(words) <= 2 and any(word in THANKS_WORDS for word in words):
        return "thanks"
    if len(words) <= 2 and any(word in YES_WORDS for word in words):
        return "affirmation"
    if len(words) <= 2 and any(word in NO_WORDS for word in words):
        return "negation"
    if len(words) <= 2 and any(word in REACTION_WORDS for word in words):
        return "short_reaction"
    if len(text) < 25 or len(words) < 4:
        return "short_context_fragment"
    return None


def load_body_records(
    path: Path,
    filter_short_bodies: bool,
) -> tuple[list[BodyRecord], dict[str, list[str]]]:
    export = json.loads(path.read_text())
    records: list[BodyRecord] = []
    buckets: dict[str, list[str]] = {}
    for index, message in enumerate(export.get("messages", [])):
        body = (message.get("content") or "").strip()
        if not body:
            buckets.setdefault("empty_body", []).append(body)
            continue

        bucket = short_bucket(body) if filter_short_bodies else None
        if bucket:
            buckets.setdefault(bucket, []).append(body)
        else:
            records.append(BodyRecord(index=index, body=body))
    return records, buckets


def body_hash(records: list[BodyRecord], model: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode())
    for record in records:
        digest.update(record.body.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_cached_embeddings(
    cache_path: Path, records: list[BodyRecord], model: str
) -> np.ndarray | None:
    if not cache_path.exists():
        return None

    cache = np.load(cache_path, allow_pickle=False)
    if str(cache["model"]) != model:
        return None
    if str(cache["body_hash"]) != body_hash(records, model):
        return None
    return cache["embeddings"].astype(np.float32)


def save_cached_embeddings(
    cache_path: Path, records: list[BodyRecord], model: str, embeddings: np.ndarray
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        model=model,
        body_hash=body_hash(records, model),
        embeddings=embeddings,
    )


def embed_records(
    records: list[BodyRecord],
    model: str,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    cached = load_cached_embeddings(cache_path, records, model)
    if cached is not None:
        print(f"loaded cached embeddings from {cache_path}", flush=True)
        return cached

    embeddings: list[Sequence[float]] = []
    bodies = [record.body for record in records]
    for start in range(0, len(bodies), batch_size):
        batch = bodies[start : start + batch_size]
        response = ollama.embed(model=model, input=batch)
        embeddings.extend(response.embeddings)
        print(f"embedded {min(start + batch_size, len(bodies))}/{len(bodies)}", flush=True)

    array = np.asarray(embeddings, dtype=np.float32)
    save_cached_embeddings(cache_path, records, model, array)
    return array


def fit_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
        copy=True,
    )
    return clusterer.fit_predict(vectors)


def pca_clustering_space(
    embeddings: np.ndarray,
    dimensions: int,
    random_state: int,
) -> tuple[np.ndarray, str]:
    normalized = normalize(embeddings)
    dimensions = min(dimensions, normalized.shape[0] - 1, normalized.shape[1])
    if dimensions < 2:
        return normalized, "normalized embeddings without PCA"

    pca = PCA(n_components=dimensions, random_state=random_state)
    projected = pca.fit_transform(normalized)
    vectors = normalize(projected)
    explained = float(np.sum(pca.explained_variance_ratio_))
    note = (
        f"normalized embeddings -> PCA({dimensions}) -> normalized PCA vectors "
        f"({explained:.1%} variance explained)"
    )
    return vectors, note


def representative_indices(
    vectors: np.ndarray,
    member_indices: np.ndarray,
    sample_count: int,
) -> list[int]:
    cluster_vectors = vectors[member_indices]
    centroid = cluster_vectors.mean(axis=0, keepdims=True)
    nearest, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
    medoid_position = int(nearest[0])
    distances = np.linalg.norm(cluster_vectors - cluster_vectors[medoid_position], axis=1)
    ordered = member_indices[np.argsort(distances)]
    return [int(index) for index in ordered[:sample_count]]


def describe_cluster(
    lines: list[str],
    records: list[BodyRecord],
    vectors: np.ndarray,
    member_indices: np.ndarray,
    heading: str,
    samples: int,
) -> None:
    examples = representative_indices(vectors, member_indices, samples)
    lines.extend([heading, ""])
    for index in examples:
        lines.append(f"- {json.dumps(records[index].body, ensure_ascii=False)}")
    lines.append("")


def write_report(
    output: Path,
    records: list[BodyRecord],
    buckets: dict[str, list[str]],
    vectors: np.ndarray,
    top_labels: np.ndarray,
    min_subcluster_size: int,
    min_samples: int,
    samples: int,
    max_subclusters: int,
    clustering_note: str,
) -> None:
    lines = [
        "# Hierarchical Message Body Clusters",
        "",
        f"Topic-bearing bodies clustered: {len(records)}",
        f"Short/no-content bodies held out: {sum(len(values) for values in buckets.values())}",
        f"Clustering space: {clustering_note}",
        "",
        "Only raw body strings were embedded. Metadata was not embedded or used for clustering.",
        "",
        "## Held-Out Buckets",
        "",
    ]

    for bucket, bodies in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
        lines.extend([f"### {bucket} ({len(bodies)} bodies)", ""])
        for body in bodies[:samples]:
            lines.append(f"- {json.dumps(body, ensure_ascii=False)}")
        lines.append("")

    noise_indices = np.where(top_labels == -1)[0]
    if len(noise_indices):
        describe_cluster(
            lines,
            records,
            vectors,
            noise_indices,
            f"## HDBSCAN Noise / Outliers ({len(noise_indices)} bodies)",
            samples,
        )

    cluster_ids = [label for label in sorted(set(top_labels)) if label != -1]
    cluster_ids.sort(key=lambda label: int(np.sum(top_labels == label)), reverse=True)
    lines.extend(["## Top-Level Clusters", ""])

    for cluster_id in cluster_ids:
        member_indices = np.where(top_labels == cluster_id)[0]
        describe_cluster(
            lines,
            records,
            vectors,
            member_indices,
            f"### Cluster {cluster_id} ({len(member_indices)} bodies)",
            samples,
        )

        if len(member_indices) < min_subcluster_size * 3:
            continue

        sub_vectors = vectors[member_indices]
        sub_labels = fit_hdbscan(
            sub_vectors,
            min_cluster_size=min_subcluster_size,
            min_samples=min_samples,
        )
        subcluster_ids = [label for label in sorted(set(sub_labels)) if label != -1]
        subcluster_ids.sort(key=lambda label: int(np.sum(sub_labels == label)), reverse=True)

        if not subcluster_ids:
            continue

        lines.extend(["Subclusters:", ""])
        for subcluster_id in subcluster_ids[:max_subclusters]:
            local_positions = np.where(sub_labels == subcluster_id)[0]
            global_indices = member_indices[local_positions]
            describe_cluster(
                lines,
                records,
                vectors,
                global_indices,
                f"#### Cluster {cluster_id}.{subcluster_id} ({len(global_indices)} bodies)",
                min(samples, 8),
            )

        local_noise_count = int(np.sum(sub_labels == -1))
        if local_noise_count:
            lines.append(f"Subcluster noise: {local_noise_count} bodies")
            lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
