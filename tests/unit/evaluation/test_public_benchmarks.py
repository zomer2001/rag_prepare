from __future__ import annotations

import json
from pathlib import Path

from rag_graph_demo.evaluation.datasets import load_bundle
from rag_graph_demo.evaluation.public_benchmarks import prepare_beir, prepare_hotpotqa


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prepare_hotpotqa_preserves_all_supporting_documents(tmp_path: Path) -> None:
    raw_path = tmp_path / "hotpot.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "How are Alpha and Beta connected?",
                    "answer": "answer",
                    "type": "bridge",
                    "context": [
                        ["Alpha", ["Alpha first.", "Alpha supports the answer."]],
                        ["Beta", ["Beta supports the answer."]],
                        ["Noise", ["Unrelated."]],
                    ],
                    "supporting_facts": [["Alpha", 1], ["Beta", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    manifest = prepare_hotpotqa(raw_path, output_dir, sample_size=1, seed=7)
    bundle = load_bundle(output_dir)

    assert manifest["gold_document_count"] == 2
    assert len(bundle.documents) == 3
    assert len(bundle.queries[0].gold_evidence) == 2
    assert {item.text for item in bundle.queries[0].gold_evidence} == {
        "Alpha supports the answer.",
        "Beta supports the answer.",
    }


def test_prepare_beir_keeps_gold_and_adds_fixed_distractors(tmp_path: Path) -> None:
    source = tmp_path / "source" / "toy"
    _write_jsonl(
        source / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Gold", "text": "gold text"},
            {"_id": "d2", "title": "Noise 1", "text": "noise"},
            {"_id": "d3", "title": "Noise 2", "text": "noise"},
        ],
    )
    _write_jsonl(
        source / "queries.jsonl",
        [{"_id": "q1", "text": "query"}],
    )
    qrels = source / "qrels" / "test.tsv"
    qrels.parent.mkdir(parents=True)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    manifest = prepare_beir(
        tmp_path / "source",
        output_dir,
        name="toy",
        split="test",
        sample_size=1,
        seed=7,
        corpus_size=2,
        distractor_seed=11,
    )
    bundle = load_bundle(output_dir)

    assert manifest["gold_document_count"] == 1
    assert manifest["distractor_count"] == 1
    assert len(bundle.documents) == 2
    assert bundle.queries[0].gold_evidence[0].document_id == "beir-toy:d1"
