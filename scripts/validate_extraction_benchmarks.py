#!/usr/bin/env python3
"""Fetch DWIE article text and validate DWIE, MAVEN, and MAVEN-ERE files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "data" / "benchmarks" / "extraction"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def _iter_relation_pairs(relations: Any):
    if isinstance(relations, dict):
        for relation_type, pairs in relations.items():
            for pair in pairs:
                yield relation_type, pair
    elif isinstance(relations, list):
        for pair in relations:
            yield "relation", pair


def fetch_dwie_content(
    annotations_dir: Path,
    article_map_path: Path,
    output_dir: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Fetch article bodies with the official DW API mapping, with resume support."""
    article_map = json.loads(article_map_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    id_replacements = {18525950: 19026607}
    content_replacements = {
        "DW_40663341": [
            ("starting with Sunday's", "starting Sunday's"),
            ("$1 million (€840,000)", "one million dollars\u00a0(840,000 euros)"),
            ("who kneel in protest during", "to kneel in protest during"),
        ]
    }
    counters: Counter[str] = Counter()

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for index, item in enumerate(article_map, start=1):
            article_id = item["id"]
            destination = output_dir / f"{article_id}.json"
            if destination.exists():
                try:
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = None
                if existing and existing.get("id") == article_id and existing.get("content"):
                    counters["resumed"] += 1
                    continue

            annotation = json.loads(
                (annotations_dir / f"{article_id}.json").read_text(encoding="utf-8")
            )
            article_number = int(article_id[3:])
            url = item["url"]
            if article_number in id_replacements:
                url = url.replace(str(article_number), str(id_replacements[article_number]))

            response = None
            for attempt in range(1, 4):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    time.sleep(attempt)
            assert response is not None
            article = response.json()
            content = "{}\n{}\n{}".format(
                article.get("name", ""), article.get("teaser", ""), article.get("text", "")
            )
            for old, new in content_replacements.get(article_id, []):
                content = content.replace(old, new)

            for mention in annotation.get("mentions", []):
                expected = mention.get("text", "").replace("\u00a0", " ").replace("\u200b", "")
                actual = content[mention["begin"] : mention["end"]]
                if actual != expected:
                    counters["mention_span_mismatches"] += 1

            payload = {
                "id": annotation["id"],
                "content": content,
                "tags": annotation.get("tags", []),
                "mentions": annotation.get("mentions", []),
                "concepts": annotation.get("concepts", []),
                "relations": annotation.get("relations", []),
                "frames": annotation.get("frames", []),
                "iptc": annotation.get("iptc", []),
            }
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            counters["downloaded"] += 1
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
            if digest != item["hash"]:
                counters["hash_mismatches"] += 1
            if index % 50 == 0 or index == len(article_map):
                print(f"DWIE fetch: {index}/{len(article_map)}")
            time.sleep(0.1)

    counters["expected"] = len(article_map)
    counters["files"] = len(list(output_dir.glob("*.json")))
    return dict(counters)


def validate_dwie(annotations_dir: Path, content_dir: Path) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No DWIE annotations found in {annotations_dir}")

    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        counters["documents"] += 1
        counters["mentions"] += len(document.get("mentions", []))
        counters["entities"] += len(document.get("concepts", []))
        counters["relations"] += len(document.get("relations", []))
        split_counts.update(tag for tag in document.get("tags", []) if tag != "all")
        concept_ids = {concept["concept"] for concept in document.get("concepts", [])}
        for mention in document.get("mentions", []):
            if mention.get("concept") not in concept_ids:
                counters["invalid_mention_entity_refs"] += 1
        for relation in document.get("relations", []):
            if relation.get("s") not in concept_ids or relation.get("o") not in concept_ids:
                counters["invalid_relation_entity_refs"] += 1

        content_path = content_dir / path.name
        if content_path.exists():
            content_document = json.loads(content_path.read_text(encoding="utf-8"))
            content = content_document.get("content", "")
            if content:
                counters["documents_with_content"] += 1
            for mention in content_document.get("mentions", []):
                expected = mention.get("text", "").replace("\u00a0", " ").replace("\u200b", "")
                if content[mention["begin"] : mention["end"]] != expected:
                    counters["content_span_mismatches"] += 1

    error_keys = (
        "invalid_mention_entity_refs",
        "invalid_relation_entity_refs",
        "content_span_mismatches",
    )
    return {
        **dict(counters),
        "splits": dict(split_counts),
        "structural_errors": sum(counters[key] for key in error_keys),
    }


def validate_maven(data_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    total_errors = 0
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        counters: Counter[str] = Counter()
        event_types: set[str] = set()
        for document in read_jsonl(path):
            counters["documents"] += 1
            sentences = document.get("content", [])
            if split == "test":
                mentions = document.get("candidates", [])
                counters["candidate_triggers"] += len(mentions)
            else:
                events = document.get("events", [])
                counters["event_clusters"] += len(events)
                mentions = [mention for event in events for mention in event.get("mention", [])]
                counters["event_mentions"] += len(mentions)
                event_types.update(event.get("type", "") for event in events)
                counters["negative_triggers"] += len(document.get("negative_triggers", []))
                mentions = [*mentions, *document.get("negative_triggers", [])]

            for mention in mentions:
                sentence_index = mention.get("sent_id", -1)
                offset = mention.get("offset", [])
                if not 0 <= sentence_index < len(sentences) or len(offset) != 2:
                    counters["invalid_trigger_offsets"] += 1
                    continue
                tokens = sentences[sentence_index].get("tokens", [])
                begin, end = offset
                if not 0 <= begin < end <= len(tokens):
                    counters["invalid_trigger_offsets"] += 1
                    continue
                actual = " ".join(tokens[begin:end]).lower()
                expected = mention.get("trigger_word", "").lower()
                if actual != expected:
                    counters["trigger_text_mismatches"] += 1
        counters["event_types"] = len(event_types)
        split_errors = counters["invalid_trigger_offsets"]
        total_errors += split_errors
        summary["splits"][split] = dict(counters)
    summary["structural_errors"] = total_errors
    return summary


def validate_maven_ere(data_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    total_errors = 0
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        counters: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        for document in read_jsonl(path):
            counters["documents"] += 1
            sentences = document.get("tokens", [])
            timex = document.get("TIMEX", [])
            counters["timex_mentions"] += len(timex)
            if split == "test":
                event_mentions = document.get("event_mentions", [])
                counters["event_mentions"] += len(event_mentions)
                known_ids = {mention["id"] for mention in event_mentions}
            else:
                events = document.get("events", [])
                counters["event_coreference_clusters"] += len(events)
                event_mentions = [
                    mention for event in events for mention in event.get("mention", [])
                ]
                counters["event_mentions"] += len(event_mentions)
                known_ids = {event["id"] for event in events}
            known_ids.update(item["id"] for item in timex)

            for mention in [*event_mentions, *timex]:
                sentence_index = mention.get("sent_id", -1)
                offset = mention.get("offset", [])
                if not 0 <= sentence_index < len(sentences) or len(offset) != 2:
                    counters["invalid_mention_offsets"] += 1
                    continue
                begin, end = offset
                if not 0 <= begin < end <= len(sentences[sentence_index]):
                    counters["invalid_mention_offsets"] += 1

            if split != "test":
                for field in ("temporal_relations", "causal_relations", "subevent_relations"):
                    for relation_type, pair in _iter_relation_pairs(document.get(field, {})):
                        relation_counts[f"{field}.{relation_type}"] += 1
                        if len(pair) != 2 or pair[0] not in known_ids or pair[1] not in known_ids:
                            counters["invalid_relation_refs"] += 1

        counters["relations"] = sum(relation_counts.values())
        split_errors = counters["invalid_mention_offsets"] + counters["invalid_relation_refs"]
        total_errors += split_errors
        summary["splits"][split] = {**dict(counters), "relation_types": dict(relation_counts)}
    summary["structural_errors"] = total_errors
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--fetch-dwie-content", action="store_true")
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    dwie_root = benchmark_root / "dwie" / "raw"
    dwie_annotations = dwie_root / "annos"
    dwie_content = dwie_root / "annos_with_content"
    if args.fetch_dwie_content:
        fetch_summary = fetch_dwie_content(
            dwie_annotations,
            dwie_root / "article_id_to_url.json",
            dwie_content,
            timeout=args.http_timeout,
        )
        print("DWIE download:", json.dumps(fetch_summary, ensure_ascii=False, sort_keys=True))

    summary = {
        "dwie": validate_dwie(dwie_annotations, dwie_content),
        "maven": validate_maven(benchmark_root / "maven" / "raw"),
        "maven_ere": validate_maven_ere(benchmark_root / "maven_ere" / "raw" / "MAVEN_ERE"),
    }
    summary["ok"] = all(dataset["structural_errors"] == 0 for dataset in summary.values())
    output = args.output or (benchmark_root / "validation_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Validation summary written to {output}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
