# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db" # Path to store ChromaDB data
COLLECTION_NAME = "medical_rag_chroma" # Use a different name if needed

# Data Configuration
DATA_FILE = "./data/processed_data.json"

# Model Configuration
# Example: 'all-MiniLM-L6-v2' (dim 384), 'thenlper/gte-large' (dim 1024)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# EMBEDDING_MODEL_NAME = "thenlper/gte-large"
# EMBEDDING_MODEL_NAME = "Qwen/Qwen2.5-7B"

# EMBEDDING_DIM = 384 # Must match EMBEDDING_MODEL_NAME
EMBEDDING_DIM = 384
# EMBEDDING_DIM = 1024
# EMBEDDING_DIM = 4096


GENERATION_MODEL_NAME = "gpt2"
# GENERATION_MODEL_NAME = "THUDM/chatglm3-6b"
# GENERATION_MODEL_NAME = "Qwen/Qwen2.5-7B"


# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 1000
TOP_K = 50
# Milvus index parameters (adjust based on data size and needs)
INDEX_METRIC_TYPE = "L2" # Or "IP"
INDEX_TYPE = "IVF_FLAT"  # Milvus Lite 支持的索引类型
# HNSW index params (adjust as needed)
INDEX_PARAMS = {"nlist": 128}
# HNSW search params (adjust as needed)
SEARCH_PARAMS = {"nprobe": 16}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
# Key: document ID (int), Value: dict {'title': str, 'abstract': str, 'content': str}
id_to_doc_map = {}


# Reranking Configuration
RERANKING_ENABLED = True  # 控制是否启用重排序
RERANKING_MODEL_NAME = "BAAI/bge-reranker-base"  # 重排序模型名称（交叉编码器）
# RERANKING_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
# RERANKING_MODEL_NAME = "Qwen/Qwen2.5-7B"
# RERANKING_MODEL_NAME = "Qwen/Qwen2.5-3B"

RERANKING_NUM = 20


GENERATION_TEMPERATURE = 0.7  # 控制生成随机性（0~1，越大越随机）
NUM_CANDIDATES = 3            # 生成候选答案数量

# 新增评估配置
EVAL_DATA_PATH = "./data/eval_queries.json"  # 评估数据集路径
RECALL_TOP_K = [10]  # 不同top K值的召回率评估