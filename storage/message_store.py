from typing import Any
from pymilvus import MilvusClient
from text_utils import generate_message_id, preprocess_wechat_text
from storage.milvus_schema import create_or_load_wechat_collection
from settings import WECHAT_COLLECTION_NAME

def insert_wechat_messages(
    messages: list[dict[str, Any]],
    client: MilvusClient,
    embedder,
    sentiment_model,
) -> bool:
    """Upsert CHAT messages into the wechat_messages collection.

    PDF chunks no longer pass through here — they go to
    storage.pdf_store.insert_pdf_chunks and the pdf_documents collection.
    """
    create_or_load_wechat_collection(client)

    candidates = []
    seen_ids = set()

    for message in messages:
        cleaned_text = preprocess_wechat_text(message["text"])
        if not cleaned_text:
            continue

        timestamp = message.get("timestamp")
        message_id = message.get("msg_id") or generate_message_id(
            message["chat_id"],
            message["sender"],
            message["receiver"],
            cleaned_text,
            timestamp,
        )

        if message_id in seen_ids:
            continue

        seen_ids.add(message_id)
        candidates.append({
            "id": message_id,
            "sender": message["sender"],
            "receiver": message["receiver"],
            "chat_id": message["chat_id"],
            "text": cleaned_text,
            "timestamp": timestamp,
        })

    if not candidates:
        return False

    # Existing-ID filtering goes here.

    texts = [message["text"] for message in candidates]
    embeddings = embedder.get_text_embeddings(texts)
    sentiments = sentiment_model.sent_analysis_batch(texts)

    rows = [
        {
            **message,
            "embedding": embedding.tolist(),
            "sentiment_scores": sentiment,
        }
        for message, embedding, sentiment in zip(candidates, embeddings, sentiments)
    ]

    client.upsert(collection_name=WECHAT_COLLECTION_NAME, data=rows)
    return True