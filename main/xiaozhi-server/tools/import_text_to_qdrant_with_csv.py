# tools/import_text_to_qdrant_with_csv.py

import hashlib
import os
import sys
import uuid
import redis
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# --- 配置区 ---
# 将项目根目录添加到Python路径，以便能够导入config模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import load_config

config = load_config()
RAG_CONFIG = config.get("rag", {})
REDIS_CONFIG = config.get("redis", {})

# 模型和数据库配置
EMBEDDING_MODEL_NAME = RAG_CONFIG.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
QDRANT_URL = RAG_CONFIG.get("qdrant_url", "http://localhost:6333")
COLLECTION_NAME = RAG_CONFIG.get("collection_name", "default_collection")

# Redis 连接信息
REDIS_HOST = REDIS_CONFIG.get("host", "localhost")
REDIS_PORT = REDIS_CONFIG.get("port", 6379)
REDIS_DB = REDIS_CONFIG.get("db", 0)
REDIS_PARENT_DOCS_KEY_PREFIX = REDIS_CONFIG.get("parent_docs_key_prefix", "parent_docs:")
# --- 配置区结束 ---

def process_csv(file_path, redis_client):
    """
    使用CSVLoader处理单个CSV文件，每一行都成为一个独立的父/子文档。
    这种方法是通用的，不需要硬编码任何列名。
    """
    try:
        # LangChain的CSVLoader会自动处理编码问题，并将每一行转换为一个Document
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        print(f"错误: 使用CSVLoader加载文件 {file_path} 失败: {e}")
        return [], 0

    child_docs_for_qdrant = []
    parent_docs_count = 0
    
    print(f"正在处理CSV文件: {os.path.basename(file_path)}")
    for doc in tqdm(documents, desc=f"Processing rows from {os.path.basename(file_path)}"):
        # CSVLoader已经将每一行格式化为 page_content
        # 例如："姓名: 黄耀斌, 电话: 13425996302, ..."
        parent_content = doc.page_content
        
        if not parent_content.strip():
            continue

        # 1. 根据内容生成确定性的父文档ID
        h_parent = hashlib.sha256(parent_content.encode('utf-8'))
        parent_id = str(uuid.UUID(bytes=h_parent.digest()[:16]))
        redis_key = f"{REDIS_PARENT_DOCS_KEY_PREFIX}{parent_id}"
        
        # 2. 只有当父文档ID在Redis中不存在时，才执行写入，避免重复
        if not redis_client.exists(redis_key):
            redis_client.set(redis_key, parent_content)
            parent_docs_count += 1
        
        # 3. 准备子文档用于Qdrant
        # CSVLoader生成的metadata已经包含了source和row
        child_metadata = doc.metadata
        child_metadata['parent_id'] = parent_id
        child_metadata['parent_redis_key'] = redis_key
        # 确保source是文件名，而不是完整路径
        child_metadata['source'] = os.path.basename(file_path)

        child_doc = Document(page_content=parent_content, metadata=child_metadata)
        child_docs_for_qdrant.append(child_doc)
        
    return child_docs_for_qdrant, parent_docs_count

def main():
    """主函数，处理所有CSV文件并入库"""
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')
    if not os.path.isdir(csv_dir):
        print(f"错误: CSV目录未找到于 {csv_dir}")
        print("请确保 'data/csv' 目录存在，并且包含了通过 excel_to_csv.py 生成的文件。")
        return

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        redis_client.ping() # 检查连接
        print(f"成功连接到 Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        print(f"错误: 无法连接到 Redis. 请确保 Redis 服务正在运行. Error: {e}")
        return

    all_child_docs = []
    total_parent_docs_added = 0

    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_dir, filename)
            child_docs, parent_count = process_csv(file_path, redis_client)
            all_child_docs.extend(child_docs)
            total_parent_docs_added += parent_count

    if not all_child_docs:
        print("在CSV目录中没有找到可处理的内容。")
        return

    print(f"本次运行新增了 {total_parent_docs_added} 个父文档到 Redis。")

    print("初始化嵌入模型和Qdrant客户端...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    vector_size = embeddings._client.get_sentence_embedding_dimension()

    try:
        collections_response = client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]
        
        if COLLECTION_NAME not in existing_collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"Qdrant集合 '{COLLECTION_NAME}' 已成功创建。")
        else:
            print(f"Qdrant集合 '{COLLECTION_NAME}' 已存在，跳过创建步骤。")
            
    except Exception as e:
        print(f"错误: 在检查或创建Qdrant集合时发生异常: {e}")
        return

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    
    print(f"准备将 {len(all_child_docs)} 个子文档从CSV上传到Qdrant...")
    ids_for_qdrant = []
    for doc in all_child_docs:
        h = hashlib.sha256(doc.page_content.encode('utf-8'))
        doc_id = str(uuid.UUID(bytes=h.digest()[:16]))
        ids_for_qdrant.append(doc_id)

    vector_store.add_documents(all_child_docs, ids=ids_for_qdrant, batch_size=128)
    print("来自CSV的子文档已成功上传！")

if __name__ == "__main__":
    main()