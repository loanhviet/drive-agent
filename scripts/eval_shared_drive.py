"""Evaluate shared Drive retrieval against a reviewed JSONL dataset."""

import argparse
import json
from pathlib import Path
from typing import Callable

from config import DRIVE_CORPUS_ID
from services import embedding
from services.drive_vectorstore import get_drive_document_store


REQUIRED_FIELDS = {
    "case_id",
    "question",
    "expected_source",
    "expected_locator",
    "expected_terms",
    "answerable",
}


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number} is not valid JSON: {error}") from error
        missing = REQUIRED_FIELDS.difference(case)
        if missing:
            raise ValueError(f"Line {line_number} is missing fields: {', '.join(sorted(missing))}")
        case_id = str(case["case_id"]).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Line {line_number} has an empty or duplicate case_id")
        if not isinstance(case["expected_terms"], list):
            raise ValueError(f"Line {line_number} expected_terms must be a list")
        locator = case["expected_locator"]
        if not isinstance(locator, dict) or locator.get("type") not in {"page", "section", "none"}:
            raise ValueError(f"Line {line_number} has an invalid expected_locator")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


def _source_matches(actual: str, expected: str) -> bool:
    return actual.strip().casefold() == expected.strip().casefold()


def _locator_matches(metadata: dict, expected: dict) -> bool:
    if expected["type"] == "none":
        return True
    if expected["type"] == "page":
        return metadata.get("page_number") == expected.get("value")
    actual = str(metadata.get("section") or "").casefold()
    return str(expected.get("value") or "").casefold() in actual


def evaluate_cases(
    cases: list[dict],
    searcher: Callable[[str, int], list[dict]],
    *,
    top_k: int = 5,
) -> dict:
    details = []
    reciprocal_ranks = []
    source_hits = 0
    locator_hits = 0
    term_hits = 0
    abstention_hits = 0
    answerable_count = sum(bool(case["answerable"]) for case in cases)
    unanswerable_count = len(cases) - answerable_count

    for case in cases:
        results = searcher(case["question"], top_k)
        expected_source = str(case["expected_source"])
        matching = [
            (rank, result)
            for rank, result in enumerate(results, start=1)
            if _source_matches(
                str(result.get("metadata", {}).get("source_name", "")),
                expected_source,
            )
        ]
        source_hit = bool(matching)
        locator_hit = any(
            _locator_matches(result.get("metadata", {}), case["expected_locator"])
            for _, result in matching
        )
        combined = " ".join(result.get("text", "") for _, result in matching).casefold()
        terms_hit = all(str(term).casefold() in combined for term in case["expected_terms"])

        if case["answerable"]:
            source_hits += int(source_hit)
            locator_hits += int(locator_hit)
            term_hits += int(terms_hit)
            reciprocal_ranks.append(1 / matching[0][0] if matching else 0.0)
        else:
            abstained = not results
            abstention_hits += int(abstained)

        details.append(
            {
                "case_id": case["case_id"],
                "results_count": len(results),
                "source_hit": source_hit,
                "locator_hit": locator_hit,
                "terms_hit": terms_hit,
                "top_sources": [
                    result.get("metadata", {}).get("source_name", "") for result in results
                ],
            }
        )

    divisor = max(1, answerable_count)
    return {
        "cases": len(cases),
        "answerable_cases": answerable_count,
        "unanswerable_cases": unanswerable_count,
        "metrics": {
            "source_recall_at_k": source_hits / divisor,
            "locator_recall_at_k": locator_hits / divisor,
            "evidence_term_recall": term_hits / divisor,
            "mrr": sum(reciprocal_ranks) / divisor,
            "abstention": abstention_hits / max(1, unanswerable_count),
            "top_k": top_k,
        },
        "details": details,
    }


def markdown_summary(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Shared Drive retrieval evaluation",
        "",
        f"- Cases: {report['cases']}",
        f"- Source Recall@{metrics['top_k']}: {metrics['source_recall_at_k']:.3f}",
        f"- Locator Recall@{metrics['top_k']}: {metrics['locator_recall_at_k']:.3f}",
        f"- Evidence term recall: {metrics['evidence_term_recall']:.3f}",
        f"- MRR: {metrics['mrr']:.3f}",
        f"- Abstention: {metrics['abstention']:.3f}",
        "",
        "## Failures",
        "",
    ]
    failures = [
        item
        for item in report["details"]
        if not (item["source_hit"] and item["locator_hit"] and item["terms_hit"])
    ]
    lines.extend(
        f"- {item['case_id']}: sources={item['top_sources']}" for item in failures
    )
    if not failures:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    store = get_drive_document_store()

    def searcher(query: str, top_k: int) -> list[dict]:
        return store.search(
            embedding.embed_query(query),
            corpus_id=DRIVE_CORPUS_ID,
            top_k=top_k,
        )

    report = evaluate_cases(cases, searcher, top_k=args.top_k)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_summary(report), encoding="utf-8")


if __name__ == "__main__":
    main()
