# core/rag/retrieval.py

import os
import redis
import json
import requests # 需要导入 requests
from typing import List, Dict, Optional # 需要导入 Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- 配置区 ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config_loader import load_config

config = load_config()
RAG_CONFIG = config.get("rag", {})
REDIS_CONFIG = config.get("redis", {})

EMBEDDING_MODEL_NAME = RAG_CONFIG.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
QDRANT_URL = RAG_CONFIG.get("qdrant_url", "http://localhost:6333")
COLLECTION_NAME = RAG_CONFIG.get("collection_name", "default_collection")

REDIS_HOST = REDIS_CONFIG.get("host", "localhost")
REDIS_PORT = REDIS_CONFIG.get("port", 6379)
REDIS_DB = REDIS_CONFIG.get("db", 0)
PARENT_DOCS_KEY_PREFIX = REDIS_CONFIG.get("parent_docs_key_prefix", "parent_doc:")

CHILD_K = RAG_CONFIG.get("child_k", 20) # 粗排可以召回更多候选者
PARENT_K = RAG_CONFIG.get("parent_k", 5) # 精排后返回的数量
SCORE_THRESHOLD = RAG_CONFIG.get("score_threshold", 0.5)
# --- 配置区结束 ---

class RerankRetrievalEngine:
    """
    一个实现了“父文档检索器”+“重排序”的、生产级的RAG引擎。
    """
    def __init__(self):
        print("正在初始化 RetrievalEngine...")
        self.redis_client = self._connect_to_redis()
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embedding_model,
        )
        print("RetrievalEngine 初始化完成。")

    def _connect_to_redis(self):
        try:
            client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            client.ping()
            print(f"成功连接到 Redis at {REDIS_HOST}:{REDIS_PORT}")
            return client
        except redis.exceptions.ConnectionError as e:
            print(f"严重错误: 无法连接到 Redis. 检索功能将不可用. Error: {e}")
            return None

    def rerank_documents(self, query: str, documents: List[str], top_n: int) -> Optional[List[str]]:
        """
        重排序阶段（精排）
        调用 DashScope Rerank API 对检索结果进行重排序
        """
        API_KEY = "sk-977514b96730495e811f87d2b70f22c5" # 请确保这个Key是有效的

        request_body = {
            "model": "gte-rerank-v2",
            "input": { "query": query, "documents": documents },
            "parameters": { "return_documents": True, "top_n": top_n }
        }
        headers = { "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json" }

        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            result = response.json()
            return [doc["document"]["text"] for doc in result["output"]["results"]]
        except requests.exceptions.RequestException as e:
            print(f"Rerank API 调用失败: {e}")
            return None

    def search(self, query: str) -> List[str]:
        if not self.redis_client:
            print("错误: Redis未连接，无法执行父文档检索。")
            return []
            
        try:
            retrieved_children = self.vectorstore.similarity_search(query, k=CHILD_K)
        except Exception as e:
            print(f"在Qdrant中进行向量搜索时出错: {e}")
            return []

        if not retrieved_children:
            return []

        parent_ids = [doc.metadata['parent_id'] for doc in retrieved_children if 'parent_id' in doc.metadata]
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        
        prefixed_parent_ids = [f"{PARENT_DOCS_KEY_PREFIX}{pid}" for pid in unique_parent_ids]
        retrieved_parents_contents = self.redis_client.mget(prefixed_parent_ids)
        
        # 过滤掉None的结果
        candidate_parents = [content for content in retrieved_parents_contents if content]

        if not candidate_parents:
            return []

        # === 重排序步骤 ===
        print(f"正在对 {len(candidate_parents)} 个候选父文档进行重排序...")
        reranked_parents = self.rerank_documents(query, candidate_parents, top_n=PARENT_K)

        # 如果重排序失败，则退回使用粗排的结果
        final_parents = reranked_parents if reranked_parents is not None else candidate_parents[:PARENT_K]
        
        formatted_docs = []
        for i, content in enumerate(final_parents):
            formatted_docs.append(f"--- 参考资料 {i+1} ---\n{content}")

        print(f"检索成功，精排后返回 {len(formatted_docs)} 个父文档作为上下文。")
        return formatted_docs