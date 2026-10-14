from typing import Any

from pymilvus import MilvusClient

from settings import PDF_COLLECTION_NAME
from text_utils import preprocess_wechat_text
from storage.milvus_schema import create_or_load_pdf_collection


def insert_pdf_chunks(
    records: list[dict[str, Any]],
    client: MilvusClient,
    embedder,
) -> bool:
    """Embed and upsert PDF chunk records into the pdf_documents collection."""
    create_or_load_pdf_collection(client)

    candidates = []
    seen_ids = set()

    for record in records:
        cleaned_text = preprocess_wechat_text(record["text"])
        if not cleaned_text:
            continue

        chunk_id = record.get("msg_id")
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        candidates.append({
            "id": chunk_id,
            # Fallback to "chat_id" keeps compatibility with records built by
            # older versions of pdf_ingest (where chat_id held the filename).
            "file_name": record.get("file_name") or record.get("chat_id") or "unknown",
            "file_path": record.get("file_path", ""),
            "text": cleaned_text,
            "page_number": int(record.get("page_number") or 0),
            "chunk_index": int(record.get("chunk_index") or 0),
            "timestamp": int(record.get("timestamp") or 0),
        })

    if not candidates:
        return False

    texts = [c["text"] for c in candidates]
    embeddings = embedder.get_text_embeddings(texts)

    rows = [
        {**candidate, "embedding": embedding.tolist()}
        for candidate, embedding in zip(candidates, embeddings)
    ]

    client.upsert(collection_name=PDF_COLLECTION_NAME, data=rows)
    return True