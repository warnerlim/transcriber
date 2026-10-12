from storage.milvus_schema import get_milvus_client, create_or_load_wechat_collection
from storage.message_store import insert_wechat_messages
from storage.conversation_retrieval import retrieve_conversation_memory
from settings import WECHAT_COLLECTION_NAME
from nlp.embeddings import TextEmbedder
from nlp.sentiment import SentimentModel

client = get_milvus_client()
embedder = TextEmbedder()
sentiment_model = SentimentModel()


def test_run():
    
    # Recreate collection to apply new schema if switching from INT64 schema
    if client.has_collection(WECHAT_COLLECTION_NAME):
        client.drop_collection(WECHAT_COLLECTION_NAME)

    create_or_load_wechat_collection(client)

    sample_messages = [
        {
            "sender": "Me",
            "receiver": "Sarah",
            "chat_id": "Sarah",
            "text": "Hey Sarah, are we still meeting for coffee on Friday?",
            "timestamp": 1700000000
        },
        {
            "sender": "Sarah",
            "receiver": "Me",
            "chat_id": "Sarah",
            "text": "Yes! Let's meet at Starbucks at 3 PM to discuss the new project design.",
            "timestamp": 1700000060
        }
    ]

    print("--- First Insertion ---")

    insert_wechat_messages(
        sample_messages,
        client=client,
        embedder=embedder,
        sentiment_model=sentiment_model,
    )

    print("\n--- Second Insertion (Duplicates) ---")
    insert_wechat_messages(
        sample_messages,
        client=client,
        embedder=embedder,
        sentiment_model=sentiment_model,
    )

    # Verify count
    res = client.query(collection_name=WECHAT_COLLECTION_NAME, filter="timestamp > 0", output_fields=["id"])
    print(f"\nTotal records stored in Milvus: {len(res)} (Expected: 2)")

def test_run_2():
    """
    Test Run #2: Searches the vector database using semantic vector search
    to verify relevant message retrieval and surrounding conversation context.
    """

    print("==========================================")
    print("      TEST RUN #2: VECTOR SEARCH TEST     ")
    print("==========================================")

    # 1. Ensure sample dataset is populated
    sample_messages = [
        {
            "sender": "Me",
            "receiver": "Sarah",
            "chat_id": "Sarah",
            "text": "Hey Sarah, are we still meeting for coffee on Friday?",
            "timestamp": 1700000000
        },
        {
            "sender": "Sarah",
            "receiver": "Me",
            "chat_id": "Sarah",
            "text": "Yes! Let's meet at Starbucks at 3 PM to discuss the new project design.",
            "timestamp": 1700000060
        },
        {
            "sender": "Me",
            "receiver": "Sarah",
            "chat_id": "Sarah",
            "text": "Awesome, see you then!",
            "timestamp": 1700000120
        }
    ]

    print("\n--- Ingesting Sample Context ---")
    insert_wechat_messages(
        sample_messages,
        client=client,
        embedder=embedder,
        sentiment_model=sentiment_model,
    )

    # 2. Test semantic search queries
    test_queries = [
        "When and where are we meeting for coffee?",
        "What project are we going to discuss at 3 PM?"
    ]

    for query in test_queries:
        print(f"\n------------------------------------------")
        print(f"Search Query: '{query}'")
        print(f"------------------------------------------")

        results = retrieve_conversation_memory(
            query=query,
            client=client,
            embedder=embedder,
            contact_name="Sarah",
            top_k=2,
            include_context_window=True,
        )

        if not results:
            print("  No relevant records retrieved.")
            continue

        for idx, hit in enumerate(results, start=1):
            print(f"\n  [Hit #{idx}]")
            print(f"    Message ID : {hit['id'][:16]}...")
            print(f"    Sender     : {hit['sender']}")
            print(f"    Text       : \"{hit['text']}\"")
            print(f"    Datetime   : {hit['datetime']}")
            print(f"    L2 Distance: {hit['distance']:.4f}")

            if "surrounding_chat" in hit and hit["surrounding_chat"]:
                print("    Surrounding Context Window:")
                for msg in hit["surrounding_chat"]:
                    print(f"      [{msg['time']}] {msg['sender']}: {msg['text']}")

def test_run_3():
    """
    Test Run #3: Verifies that your ingested PDF lecture notes 
    are successfully stored and retrievable from Milvus.
    """

    print("\n==================================================")
    print("   TEST RUN #3: PDF INGESTION VERIFICATION TEST   ")
    print("==================================================")

    pdf_files = [
        "1_LeadUp_Confidentiality.pdf",
        "2_AES_SCA.pdf",
        "3_SecretSharing_Boolean.pdf",
        "4_Boolean_MPC.pdf"
    ]
    # 1. Inspect collection row statistics
    try:
        stats = client.get_collection_stats(collection_name=WECHAT_COLLECTION_NAME)
        print(f"\n📊 Collection Stats: {stats}")
    except Exception as exc:
        print(f"\n📊 Collection Stats could not be fetched: {exc}")

    # 2. Check records stored per file name (chat_id)
    for file_name in pdf_files:
        results = client.query(
            collection_name=WECHAT_COLLECTION_NAME,
            filter=f"chat_id == '{file_name}'",
            output_fields=["id", "chat_id", "text"],
            limit=3
        )
        print(f"\n📁 File: {file_name}")
        if results:
            print(f"  ✅ Found chunks stored in Milvus! (Showing sample results)")
            for idx, record in enumerate(results, start=1):
                print(f"    - [Chunk {idx}]: \"{record['text']}\"")
        else:
            print("  ❌ No records found for this file.")
    # 3. Test semantic search against lecture concepts
    test_queries = [
        "One-time pad and perfect secrecy",
        "Differential Power Analysis on AES",
        "Boolean secret sharing scheme",
        "Ishai Sahai Wagner multiplication algorithm",
        "REAL VS RANDOM SECURITY"
    ]

    print("\n--------------------------------------------------")
    print("      Testing Semantic Retrieval on PDFs          ")
    print("--------------------------------------------------")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieve_conversation_memory(
            query=query,
            client=client,
            embedder=embedder,
            top_k=2,
            include_context_window=False
        )
        if not results:
            print("  No relevant records retrieved.")
            continue
        for idx, hit in enumerate(results, start=1):
            print(f"  [Hit #{idx}] File: {hit['chat_id']} (Distance: {hit['distance']:.4f})")
            print(f"    Text: \"{hit['text']}\"")

def drop_all_collections(client):
    """
    Finds and drops ALL existing collections in Milvus to perform a complete reset.
    """
    if client is None:
        client = get_milvus_client()

    collections = client.list_collections()
    if not collections:
        print("No collections found. Database is already empty.")
        return

    print(f"Found {len(collections)} collection(s) to delete: {collections}")
    for collection_name in collections:
        client.drop_collection(collection_name)
        print(f"  - Dropped collection: '{collection_name}'")

    print("Database reset complete. All collections deleted.")

if __name__ == "__main__":
    # test_run()
    # print("\n")
    # test_run_2()
    test_run_3()