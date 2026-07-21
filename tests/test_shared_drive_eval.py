import json

import pytest

from scripts.eval_shared_drive import evaluate_cases, load_cases, markdown_summary


def test_eval_loader_rejects_duplicate_case_ids(tmp_path):
    case = {
        "case_id": "duplicate",
        "question": "Question",
        "expected_source": "Guide.pdf",
        "expected_locator": {"type": "page", "value": 1},
        "expected_terms": ["answer"],
        "answerable": True,
    }
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(case) + "\n" + json.dumps(case) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_cases(dataset)


def test_eval_metrics_cover_source_locator_terms_and_abstention():
    cases = [
        {
            "case_id": "answerable",
            "question": "Question",
            "expected_source": "Guide.pdf",
            "expected_locator": {"type": "page", "value": 2},
            "expected_terms": ["grounded"],
            "answerable": True,
        },
        {
            "case_id": "unanswerable",
            "question": "Unknown",
            "expected_source": "",
            "expected_locator": {"type": "none", "value": None},
            "expected_terms": [],
            "answerable": False,
        },
    ]

    def searcher(query, _top_k):
        if query == "Unknown":
            return []
        return [
            {
                "text": "Grounded evidence",
                "metadata": {"source_name": "Guide.pdf", "page_number": 2},
            }
        ]

    report = evaluate_cases(cases, searcher)

    assert report["metrics"]["source_recall_at_k"] == 1
    assert report["metrics"]["locator_recall_at_k"] == 1
    assert report["metrics"]["evidence_term_recall"] == 1
    assert report["metrics"]["mrr"] == 1
    assert report["metrics"]["abstention"] == 1
    assert "Source Recall@5: 1.000" in markdown_summary(report)
