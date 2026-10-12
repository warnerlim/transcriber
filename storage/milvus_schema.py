from pymilvus import MilvusClient, DataType
from settings import WECHAT_COLLECTION_NAME

def get_milvus_client(host="localhost", port=19530) -> MilvusClient:
    return MilvusClient(uri=f"http://{host}:{port}")

def create_or_load_wechat_collection(client: MilvusClient = None) -> MilvusClient:
    """
    Initializes the unified WeChat message memory collection using MilvusClient.
    Uses a deterministic VARCHAR primary key (id) to prevent duplicate insertions.
    """
    if client is None:
        client = get_milvus_client()

    if client.has_collection(WECHAT_COLLECTION_NAME):
        return client

    # Define Schema with deterministic string primary key
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="sender", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="receiver", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="chat_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="sentiment_scores", datatype=DataType.FLOAT_VECTOR, dim=3)
    schema.add_field(field_name="timestamp", datatype=DataType.INT64)

    # Define Index parameters
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    index_params.add_index(
        field_name="sentiment_scores",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 64}
    )

    client.create_collection(
        collection_name=WECHAT_COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    return client