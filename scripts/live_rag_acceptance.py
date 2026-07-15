"""Opt-in live acceptance for Drive extraction, Gemini embeddings, and Qdrant RAG."""

import os
import tempfile
import time
import uuid

from qdrant_client import QdrantClient

from agent import Agent
from registry.context import execution_context
from registry.registry import ToolRegistry
from services import embedding
from services.audit import AuditStore
from services.vectorstore import VectorStore, set_vector_store_for_testing
from tools import google_drive, memory, read_file


CV_NAME = "Lo_Anh_Viet_CV.pdf"
XAI_NAME = (
    "Nhom_3_NGHIÊN CỨU ỨNG DỤNG TRÍ TUỆ NHÂN TẠO CÓ THỂ GIẢI THÍCH "
    "(XAI) TRONG PHÂN LOẠI ĐỘ KHẨN CẤP CỦA PHIẾU HỖ TRỢ.pdf"
)


class CountingEmbeddingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.name = provider.name
        self.model = provider.model
        self.dimension = provider.dimension
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts, task_type):
        self.calls.append((task_type, len(texts)))
        return self.provider.embed(texts, task_type)


def _actor(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "role": "admin",
        "scopes": ["drive:read", "memory:read", "memory:write"],
    }


def _file_id(files: list[dict], name: str) -> str:
    match = next((item for item in files if item["name"] == name), None)
    if match is None:
        raise AssertionError(f"Drive acceptance file was not found: {name}")
    return match["id"]


def _read_and_save(file_id: str) -> tuple[dict, dict, float]:
    downloaded = google_drive.get_drive_file(file_id)
    extracted = read_file.read_file_tool.handler(downloaded["artifact_id"])
    started = time.perf_counter()
    saved = memory.save_memory(document_ref=extracted["document_ref"], category="document")
    return extracted, saved, time.perf_counter() - started


def _assert_retrieval(source_name: str, query: str, expected: tuple[str, ...]) -> dict:
    result = memory.search_memory(
        query,
        top_k=5,
        memory_type="document",
        source_name=source_name,
    )
    combined = " ".join(item["text"] for item in result["memories"]).casefold()
    missing = [term for term in expected if term.casefold() not in combined]
    if result["status"] != "found" or missing:
        raise AssertionError(
            f"Retrieval failed for {source_name!r}; status={result['status']}, missing={missing}"
        )
    top = result["memories"][0]
    return {
        "query": query,
        "top_score": round(top["score"], 4),
        "citation": top["citation"],
        "results_count": result["results_count"],
    }


def main() -> None:
    qdrant_port = int(os.getenv("LIVE_QDRANT_PORT", "6335"))
    collection = f"drive_agent_live_acceptance_{uuid.uuid4().hex}"
    user_id = f"rehearsal-{uuid.uuid4()}"
    client = QdrantClient(host="127.0.0.1", port=qdrant_port)
    provider = CountingEmbeddingProvider(embedding.get_embedding_provider())
    embedding.set_embedding_provider_for_testing(provider)
    set_vector_store_for_testing(VectorStore(client, collection, provider.dimension))

    try:
        with execution_context(_actor(user_id)):
            files = google_drive.list_drive_files()["files"]
            cv_read, cv_saved, cv_seconds = _read_and_save(_file_id(files, CV_NAME))
            xai_read, xai_saved, xai_seconds = _read_and_save(_file_id(files, XAI_NAME))

            checks = [
                _assert_retrieval(
                    CV_NAME,
                    "Dự án phát hiện ảnh AI dùng bao nhiêu ảnh và đạt ROC-AUC bao nhiêu?",
                    ("86,000", "0.9356"),
                ),
                _assert_retrieval(
                    XAI_NAME,
                    "Random Forest đạt Accuracy và F1-macro bao nhiêu?",
                    ("74,52", "73,50"),
                ),
                _assert_retrieval(
                    XAI_NAME,
                    "LIME và SHAP khác nhau thế nào trong cách giải thích?",
                    ("cục bộ", "toàn cục"),
                ),
                _assert_retrieval(
                    XAI_NAME,
                    "Phiếu System outage được dự đoán mức ưu tiên nào và độ tin cậy bao nhiêu?",
                    ("HIGH", "91,0"),
                ),
            ]

            with tempfile.TemporaryDirectory() as temporary_directory:
                registry = ToolRegistry(
                    authenticator=lambda _token: _actor(user_id),
                    audit_store=AuditStore(os.path.join(temporary_directory, "audit.db")),
                )
                agent = Agent(
                    service_api_key="live-acceptance",
                    session_id="live-acceptance",
                    registry=registry,
                )
                agent_answer = agent.run(
                    "Theo tài liệu XAI đã lưu, Random Forest đạt Accuracy và F1-macro bao nhiêu? "
                    "Chỉ trả lời hai số cùng nguồn và mục tài liệu."
                )
                if "search_memory" not in agent.last_tools_used:
                    raise AssertionError("Live Agent did not call search_memory")
                search_call = next(
                    call
                    for message in agent.conversation_history
                    for call in message.get("tool_calls", [])
                    if call["name"] == "search_memory"
                )
                search_result = next(
                    result["result"]
                    for message in agent.conversation_history
                    for result in message.get("results", [])
                    if result["name"] == "search_memory"
                )
                search_payload = search_result["result"] if search_result["ok"] else None
                if not search_payload or search_payload["status"] != "found":
                    raise AssertionError(
                        "Live Agent search did not find memory; "
                        f"arguments={search_call['arguments']}, result={search_result}"
                    )
                accuracy_found = any(value in agent_answer for value in ("74,52", "0,7452"))
                f1_found = any(value in agent_answer for value in ("73,50", "0,7350"))
                if not accuracy_found or not f1_found:
                    raise AssertionError(
                        "Live Agent answer did not contain the grounded metrics; "
                        f"answer={agent_answer!r}"
                    )
                if not any(marker in agent_answer for marker in ("4.3", "Nhom_3")):
                    raise AssertionError("Live Agent answer did not include a document citation")
                if "](drive/" in agent_answer:
                    raise AssertionError("Live Agent invented a Drive URL from file_id")

        with execution_context(_actor(f"isolated-{uuid.uuid4()}")):
            isolated = memory.search_memory(
                "Random Forest",
                memory_type="document",
                source_name=XAI_NAME,
            )
            if isolated["status"] != "insufficient_data":
                raise AssertionError("Memory search leaked data across users")

        document_batches = [
            size for task_type, size in provider.calls if task_type == "RETRIEVAL_DOCUMENT"
        ]
        print(
            {
                "status": "passed",
                "drive_files": len(files),
                "cv": {
                    "total_chars": cv_read["total_chars"],
                    "preview_chars": len(cv_read["content"]),
                    "is_truncated": cv_read["is_truncated"],
                    "chunks": cv_saved["chunks_saved"],
                    "index_seconds": round(cv_seconds, 2),
                },
                "xai": {
                    "total_chars": xai_read["total_chars"],
                    "preview_chars": len(xai_read["content"]),
                    "is_truncated": xai_read["is_truncated"],
                    "chunks": xai_saved["chunks_saved"],
                    "index_seconds": round(xai_seconds, 2),
                },
                "document_embedding_batches": document_batches,
                "retrieval_checks": checks,
                "live_agent": {
                    "tools_used": agent.last_tools_used,
                    "answer": agent_answer,
                },
                "cross_user_isolation": "passed",
            }
        )
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()
        set_vector_store_for_testing(None)
        embedding.set_embedding_provider_for_testing(None)


if __name__ == "__main__":
    main()
