from datetime import datetime

from storage.milvus_schema import get_milvus_client
from text_utils import preprocess_wechat_text
from settings import WECHAT_COLLECTION_NAME, PDF_COLLECTION_NAME


def _sanitize(value: str) -> str:
    """Strip single quotes so a value can't break a Milvus filter expression."""
    return str(value).replace("'", "")


# --------------------------------------------------------------------------- #
# Chat memory (wechat_messages collection)
# --------------------------------------------------------------------------- #

def get_surrounding_context(
    chat_id: str,
    client,
    target_timestamp: int,
    window_seconds: int = 300,
) -> list:
    """
    Queries Milvus for surrounding conversational context around a timestamp.
    """
    if client is None:
        client = get_milvus_client()

    min_ts = target_timestamp - window_seconds
    max_ts = target_timestamp + window_seconds

    expr = f"chat_id == '{_sanitize(chat_id)}' and timestamp >= {min_ts} and timestamp <= {max_ts}"

    results = client.query(
        collection_name=WECHAT_COLLECTION_NAME,
        filter=expr,
        output_fields=["sender", "text", "timestamp"],
        limit=20
    )

    sorted_context = sorted(results, key=lambda x: x["timestamp"])
    return [
        {
            "sender": item["sender"],
            "text": item["text"],
            "time": datetime.fromtimestamp(item["timestamp"]).strftime("%H:%M:%S")
        }
        for item in sorted_context
    ]


def retrieve_conversation_memory(
    query: str,
    client,
    embedder,
    contact_name: str = None,
    start_timestamp: int = None,
    end_timestamp: int = None,
    top_k: int = 5,
    include_context_window: bool = True,
) -> list:
    """
    Performs vector similarity search over the CHAT collection using MilvusClient.
    """
    if client is None:
        client = get_milvus_client()

    cleaned_query = preprocess_wechat_text(query)
    query_embedding = embedder.get_text_embedding(cleaned_query).tolist()

    conditions = []
    if contact_name:
        safe = _sanitize(contact_name)
        conditions.append(f"(sender == '{safe}' or receiver == '{safe}' or chat_id == '{safe}')")
    if start_timestamp:
        conditions.append(f"timestamp >= {start_timestamp}")
    if end_timestamp:
        conditions.append(f"timestamp <= {end_timestamp}")

    expr = " and ".join(conditions) if conditions else ""

    search_results = client.search(
        collection_name=WECHAT_COLLECTION_NAME,
        data=[query_embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["sender", "receiver", "chat_id", "text", "sentiment_scores", "timestamp"],
        filter=expr,
        search_params={"metric_type": "L2", "params": {"nprobe": 100}}
    )

    retrieved_memory = []
    if search_results and len(search_results) > 0:
        for match in search_results[0]:
            entity = match.get("entity", match)
            hit_data = {
                "id": match.get("id"),
                "distance": match.get("distance"),
                "sender": entity.get("sender"),
                "receiver": entity.get("receiver"),
                "chat_id": entity.get("chat_id"),
                "text": entity.get("text"),
                "timestamp": entity.get("timestamp"),
                "datetime": datetime.fromtimestamp(entity.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
            }

            if include_context_window:
                hit_data["surrounding_chat"] = get_surrounding_context(
                    chat_id=hit_data["chat_id"],
                    target_timestamp=hit_data["timestamp"],
                    window_seconds=300,
                    client=client
                )

            retrieved_memory.append(hit_data)

    return retrieved_memory


# --------------------------------------------------------------------------- #
# PDF document memory (pdf_documents collection)
# --------------------------------------------------------------------------- #

def get_neighboring_chunks(
    file_name: str,
    page_number: int,
    chunk_index: int,
    client,
    window: int = 1,
) -> list:
    """
    Fetch the chunks immediately around a hit (same file and page, adjacent
    chunk indexes) so a matched snippet comes back with its local context —
    the document equivalent of get_surrounding_context().
    """
    if client is None:
        client = get_milvus_client()

    expr = (
        f"file_name == '{_sanitize(file_name)}' "
        f"and page_number == {int(page_number)} "
        f"and chunk_index >= {int(chunk_index) - window} "
        f"and chunk_index <= {int(chunk_index) + window}"
    )

    results = client.query(
        collection_name=PDF_COLLECTION_NAME,
        filter=expr,
        output_fields=["page_number", "chunk_index", "text"],
        limit=2 * window + 1,
    )

    return sorted(results, key=lambda x: x["chunk_index"])


def retrieve_pdf_memory(
    query: str,
    client,
    embedder,
    file_name: str = None,
    top_k: int = 5,
    include_neighbor_chunks: bool = True,
    neighbor_window: int = 1,
) -> list:
    """
    Vector similarity search over the PDF collection.

    Args:
        file_name: optional restriction to a single document (exact match on
            the stored file name, e.g. "annual_report.pdf").
        include_neighbor_chunks: attach adjacent chunks from the same page so
            each hit carries its surrounding context.
    """
    if client is None:
        client = get_milvus_client()

    cleaned_query = preprocess_wechat_text(query)
    query_embedding = embedder.get_text_embedding(cleaned_query).tolist()

    expr = f"file_name == '{_sanitize(file_name)}'" if file_name else ""

    search_results = client.search(
        collection_name=PDF_COLLECTION_NAME,
        data=[query_embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["file_name", "file_path", "text", "page_number", "chunk_index", "timestamp"],
        filter=expr,
        search_params={"metric_type": "L2", "params": {"nprobe": 100}}
    )

    retrieved = []
    if search_results and len(search_results) > 0:
        for match in search_results[0]:
            entity = match.get("entity", match)
            ts = entity.get("timestamp") or 0
            hit_data = {
                "id": match.get("id"),
                "distance": match.get("distance"),
                "file_name": entity.get("file_name"),
                "file_path": entity.get("file_path"),
                "text": entity.get("text"),
                "page_number": entity.get("page_number"),
                "chunk_index": entity.get("chunk_index"),
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            }

            if include_neighbor_chunks:
                hit_data["neighboring_chunks"] = get_neighboring_chunks(
                    file_name=hit_data["file_name"],
                    page_number=hit_data["page_number"],
                    chunk_index=hit_data["chunk_index"],
                    client=client,
                    window=neighbor_window,
                )

            retrieved.append(hit_data)

    return retrieved


def list_ingested_files(client, max_rows: int = 4096) -> list:
    """
    Return the sorted, de-duplicated list of file names currently stored in
    the PDF collection. Used to tell the agent which documents it can search.
    """
    if client is None:
        client = get_milvus_client()

    rows = client.query(
        collection_name=PDF_COLLECTION_NAME,
        filter="id != ''",
        output_fields=["file_name"],
        limit=max_rows,
    )
    return sorted({r["file_name"] for r in rows if r.get("file_name")})