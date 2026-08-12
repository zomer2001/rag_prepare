from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOTQA_FALLBACK_REPOSITORY = "RAGLAB/data"
HOTPOTQA_FALLBACK_FILENAME = "eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _identifier(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def download_file(url: str, destination: Path) -> Path:
    """Download once and atomically publish the completed file."""
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    if temporary.exists():
        temporary.unlink()
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def download_hotpotqa(download_dir: Path) -> Path:
    destination = download_dir / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    try:
        return download_file(HOTPOTQA_URL, destination)
    except Exception as official_error:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "HotpotQA official download failed and the Hugging Face fallback "
                "requires the evaluation extra"
            ) from official_error
        cached = Path(
            hf_hub_download(
                repo_id=HOTPOTQA_FALLBACK_REPOSITORY,
                repo_type="dataset",
                filename=HOTPOTQA_FALLBACK_FILENAME,
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.part")
        shutil.copy2(cached, temporary)
        temporary.replace(destination)
        return destination


def _sample_indices(size: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size > size:
        raise ValueError(f"sample-size must be between 1 and {size}, got {sample_size}")
    return sorted(random.Random(seed).sample(range(size), sample_size))


def _hotpot_contexts(row: dict[str, Any]) -> list[tuple[str, list[str]]]:
    value = row.get("context") or []
    if isinstance(value, dict):
        titles = value.get("title") or []
        sentences = value.get("sentences") or []
        return [
            (str(title), [str(sentence) for sentence in sentence_list])
            for title, sentence_list in zip(titles, sentences, strict=False)
        ]
    result: list[tuple[str, list[str]]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append((str(item[0]), [str(sentence) for sentence in item[1]]))
    return result


def _hotpot_supporting_facts(row: dict[str, Any]) -> list[tuple[str, int]]:
    value = row.get("supporting_facts") or []
    if isinstance(value, dict):
        titles = value.get("title") or []
        sentence_ids = value.get("sent_id") or []
        return [
            (str(title), int(sentence_id))
            for title, sentence_id in zip(titles, sentence_ids, strict=False)
        ]
    result: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append((str(item[0]), int(item[1])))
    return result


def prepare_hotpotqa(
    raw_path: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {raw_path}")
    indices = _sample_indices(len(rows), sample_size, seed)
    selected = [rows[index] for index in indices]

    corpus: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    for source_index, row in zip(indices, selected, strict=True):
        contexts = _hotpot_contexts(row)
        context_by_title = {title: sentences for title, sentences in contexts}
        for title, sentences in contexts:
            document_id = _identifier("hotpot-doc", title)
            corpus.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "title": title,
                    "text": " ".join(sentences).strip(),
                    "source": "HotpotQA distractor validation",
                },
            )

        evidence_by_title: dict[str, list[str]] = {}
        for title, sentence_id in _hotpot_supporting_facts(row):
            sentences = context_by_title.get(title)
            if sentences is None:
                raise ValueError(f"HotpotQA gold title missing from context: {title}")
            if sentence_id < 0 or sentence_id >= len(sentences):
                raise ValueError(f"Invalid supporting sentence {title}[{sentence_id}]")
            evidence_by_title.setdefault(title, []).append(sentences[sentence_id])
        gold_evidence = [
            {
                "document_id": _identifier("hotpot-doc", title),
                "text": " ".join(sentences),
            }
            for title, sentences in evidence_by_title.items()
        ]
        if not gold_evidence:
            raise ValueError(f"HotpotQA query has no gold evidence: {row.get('_id')}")
        queries.append(
            {
                "query_id": str(row.get("_id") or f"hotpot-{source_index}"),
                "question": str(row["question"]),
                "question_type": str(row.get("type") or row.get("_type") or "multi_hop"),
                "answer": str(row.get("answer") or ""),
                "gold_evidence": gold_evidence,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus.jsonl", corpus.values())
    _write_jsonl(output_dir / "queries.jsonl", queries)
    manifest = {
        "dataset": "hotpotqa-distractor-validation",
        "source_url": HOTPOTQA_URL,
        "source_query_count": len(rows),
        "sample_size": sample_size,
        "seed": seed,
        "source_indices": indices,
        "corpus_size": len(corpus),
        "gold_document_count": len(
            {item["document_id"] for query in queries for item in query["gold_evidence"]}
        ),
        "sampling": "python random.Random(seed).sample; restored to source order",
        "corpus_mode": "all ten distractor-setting contexts for selected queries",
    }
    _write_manifest(output_dir / "split_manifest.json", manifest)
    return manifest


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe zip member: {member.filename}")
        handle.extractall(destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_beir_root(extracted_dir: Path, name: str) -> Path:
    candidates = (extracted_dir / name, extracted_dir)
    for candidate in candidates:
        if (candidate / "corpus.jsonl").is_file() and (candidate / "queries.jsonl").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find extracted BEIR dataset {name} in {extracted_dir}")


def _read_qrels(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing qrels header: {path}")
        for row in reader:
            query_id = row.get("query-id") or row.get("query_id")
            corpus_id = row.get("corpus-id") or row.get("corpus_id")
            score = float(row.get("score") or 0)
            if query_id and corpus_id and score > 0:
                result.setdefault(str(query_id), []).append(str(corpus_id))
    return result


def prepare_beir(
    extracted_dir: Path,
    output_dir: Path,
    *,
    name: str,
    split: str,
    sample_size: int,
    seed: int,
    corpus_size: int,
    distractor_seed: int,
) -> dict[str, Any]:
    source_dir = _find_beir_root(extracted_dir, name)
    corpus_rows = _read_jsonl(source_dir / "corpus.jsonl")
    query_rows = _read_jsonl(source_dir / "queries.jsonl")
    qrels = _read_qrels(source_dir / "qrels" / f"{split}.tsv")
    query_by_id = {str(item["_id"]): item for item in query_rows}
    eligible_ids = sorted(set(query_by_id) & set(qrels))
    selected_indices = _sample_indices(len(eligible_ids), sample_size, seed)
    selected_ids = [eligible_ids[index] for index in selected_indices]

    corpus_by_id = {str(item["_id"]): item for item in corpus_rows}
    gold_ids = {document_id for query_id in selected_ids for document_id in qrels[query_id]}
    missing = sorted(gold_ids - set(corpus_by_id))
    if missing:
        raise ValueError(f"BEIR qrels reference {len(missing)} missing documents")
    distractor_candidates = sorted(set(corpus_by_id) - gold_ids)
    target_size = max(corpus_size, len(gold_ids))
    distractor_count = min(target_size - len(gold_ids), len(distractor_candidates))
    distractor_ids = set(
        random.Random(distractor_seed).sample(distractor_candidates, distractor_count)
    )
    selected_corpus_ids = gold_ids | distractor_ids

    corpus = [
        {
            "document_id": f"beir-{name}:{document_id}",
            "title": str(item.get("title") or document_id),
            "text": "\n\n".join(
                part
                for part in (str(item.get("title") or "").strip(), str(item["text"]).strip())
                if part
            ),
            "source": f"BEIR/{name}",
            "beir_document_id": document_id,
        }
        for document_id, item in corpus_by_id.items()
        if document_id in selected_corpus_ids
    ]
    queries = [
        {
            "query_id": f"beir-{name}:{query_id}",
            "question": str(query_by_id[query_id].get("text") or ""),
            "question_type": f"beir_{name}",
            "gold_evidence": [
                {"document_id": f"beir-{name}:{document_id}"} for document_id in qrels[query_id]
            ],
        }
        for query_id in selected_ids
    ]
    if any(not item["question"].strip() for item in queries):
        raise ValueError("Selected BEIR query has empty text")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus.jsonl", corpus)
    _write_jsonl(output_dir / "queries.jsonl", queries)
    manifest = {
        "dataset": f"beir-{name}",
        "split": split,
        "source_query_count": len(eligible_ids),
        "source_corpus_count": len(corpus_rows),
        "sample_size": sample_size,
        "seed": seed,
        "query_ids": selected_ids,
        "corpus_size_requested": corpus_size,
        "corpus_size": len(corpus),
        "gold_document_count": len(gold_ids),
        "distractor_count": len(distractor_ids),
        "distractor_seed": distractor_seed,
        "sampling": "python random.Random(seed).sample over sorted qrel query IDs",
        "corpus_mode": "all gold documents plus fixed random distractors",
    }
    _write_manifest(output_dir / "split_manifest.json", manifest)
    return manifest


def download_and_prepare_beir(
    download_dir: Path,
    output_dir: Path,
    *,
    name: str,
    split: str,
    sample_size: int,
    seed: int,
    corpus_size: int,
    distractor_seed: int,
) -> dict[str, Any]:
    archive = download_file(BEIR_URL.format(name=name), download_dir / f"{name}.zip")
    extracted_dir = download_dir / name
    if not extracted_dir.exists():
        _safe_extract(archive, extracted_dir)
    return prepare_beir(
        extracted_dir,
        output_dir,
        name=name,
        split=split,
        sample_size=sample_size,
        seed=seed,
        corpus_size=corpus_size,
        distractor_seed=distractor_seed,
    )
