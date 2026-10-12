from datetime import datetime
from storage.milvus_schema import get_milvus_client
from text_utils import preprocess_wechat_text
from settings import WECHAT_COLLECTION_NAME

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

    expr = f"chat_id == '{chat_id}' and timestamp >= {min_ts} and timestamp <= {max_ts}"

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
    Performs vector similarity search using MilvusClient.
    """
    if client is None:
        client = get_milvus_client()

    cleaned_query = preprocess_wechat_text(query)
    query_embedding = embedder.get_text_embedding(cleaned_query).tolist()

    conditions = []
    if contact_name:
        conditions.append(f"(sender == '{contact_name}' or receiver == '{contact_name}' or chat_id == '{contact_name}')")
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


