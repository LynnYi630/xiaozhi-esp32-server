from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
import requests
import yaml
from typing import List, Dict, Optional
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 构建配置文件路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "config.yaml")

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
# 从配置中获取RAG相关设置
RAG_CONFIG = config["RAG"]
CHUNK_SIZE = RAG_CONFIG.get("chunk_size", 500)
CHUNK_OVERLAP = RAG_CONFIG.get("chunk_overlap", 50)
EMBEDDING_MODEL_NAME = RAG_CONFIG.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
QDRANT_URL = RAG_CONFIG.get("qdrant_url", "http://localhost:6333")
COLLECTION_NAME = RAG_CONFIG.get("collection_name", "default_collection")
SCORE_THRESHOLD = RAG_CONFIG.get("score_threshold", 0.5)
K = RAG_CONFIG.get("k", 15)

class RetrievalEngine:
    """
    RetrievalEngine 封装了 Qdrant 向量检索功能，
    根据用户查询返回对应类别的检索内容（文本列表）。
    """
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        # 初始化中文嵌入模型（请确保与数据导入时使用的模型一致）
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # 连接到 Qdrant 向量数据库
        self.qdrant_client = QdrantClient(url=qdrant_url)
        # 注意这里传入的是 embeddings 参数，直接使用 embedding_model 实例
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="素材库",
            embedding=self.embedding_model
        )

    def get_retrieved_content(self, query: str, k: int = 15) -> list:
        """
        向量检索阶段（粗排）
        根据查询内容返回检索到的文本内容列表。
        :param query: 用户查询（中文）
        :param k: 检索返回的文档数量，默认15
        :return: 检索到的文本内容列表
        """
        docs = self.vectorstore.similarity_search(query, k=k, score_threshold=SCORE_THRESHOLD)

        return [doc.page_content for doc in docs]

    def rerank_documents(self, query: str, documents: List[str], top_n: int = 5) -> Optional[List[str]]:
        """
        重排序阶段（精排）
        调用 DashScope Rerank API 对检索结果进行重排序
        :param query: 查询文本
        :param documents: 待排序的文档列表
        :param top_n: 返回前 N 个最相关的文档
        :return: 重排序后的文档列表（按相关性降序）
        """
        API_KEY = "sk-977514b96730495e811f87d2b70f22c5"

        request_body = {
            "model": "gte-rerank-v2",
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "return_documents": True,
                "top_n": top_n
            }
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()  # 检查 HTTP 错误
            result = response.json()
            return [doc["document"]["text"] for doc in result["output"]["results"]]
        except requests.exceptions.RequestException as e:
            print(f"Rerank API 调用失败: {e}")
            return None

    def search(self, query: str, k: int = 15, rerank_top_n: int = 5) -> List[str]:
        """
        完整检索流程：向量检索 + 重排序
        :param query: 查询文本
        :param k: 向量检索返回数量
        :param rerank_top_n: 重排序返回数量
        :return: 最终检索结果
        """
        # 第一阶段：向量检索
        retrieved_docs = self.get_retrieved_content(query, k=k)

        # 第二阶段：重排序
        if retrieved_docs:
            reranked_docs = self.rerank_documents(query, retrieved_docs, top_n=rerank_top_n)
            return reranked_docs or retrieved_docs[:rerank_top_n]
        return []

    def evaluate_recall(
            self,
            test_queries: Dict[str, List[str]],
            k: int = 15,
            rerank_top_n: int = 5,
            verbose: bool = True
    ) -> Dict[str, float]:
        """
        召回率评估
        :param test_queries: {查询: [相关文档1, 相关文档2...]}
        :param k: 向量检索数量
        :param rerank_top_n: 重排序数量
        :param verbose: 是否打印详细结果
        :return: 包含每个查询和平均召回率的字典
        """
        recall_results = {}

        for query, expected_docs in test_queries.items():
            retrieved_docs = self.search(query, k=k, rerank_top_n=rerank_top_n)

            # 计算召回率
            relevant_retrieved = len(set(retrieved_docs) & set(expected_docs))
            recall = relevant_retrieved / len(expected_docs) if expected_docs else 0
            recall_results[query] = round(recall, 4)

            if verbose:
                print(f"\nQuery: {query}")
                print(f"Expected: {expected_docs}")
                print(f"Retrieved: {retrieved_docs}")
                print(f"Recall: {recall:.2f}")

        # 计算平均召回率
        recall_results["average_recall"] = round(np.mean(list(recall_results.values())), 4)

        if verbose:
            print(f"\n=== Average Recall: {recall_results['average_recall']} ===")

        return recall_results