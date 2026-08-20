from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    text_collection: str = "text_chunks"
    image_collection: str = "image_chunks"

    # Triton
    triton_url: str = "localhost:8001"
    use_triton: bool = False

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Models
    text_embed_model: str = "bge-m3"
    image_embed_model: str = "siglip"
    vlm_model: str = "qwen2-vl"
    reranker_model: str = "bge-reranker"

    # Dimensions
    text_embed_dim: int = 1024
    image_embed_dim: int = 1152

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "visionrag"
    langsmith_tracing: bool = False

    # Query Rewriting
    rewrite_strategy: str = "none"  # none, simple, multi_query, hyde
    rewrite_openai_api_key: str = ""
    rewrite_llm_model: str = "gpt-4o-mini"

    # OpenAI (RAG 답변 생성 fallback)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ClearML
    clearml_api_host: str = "http://localhost:8081"
    clearml_web_host: str = "http://localhost:8008"
    clearml_files_host: str = "http://localhost:8082"

    class Config:
        env_file = ".env"


settings = Settings()
