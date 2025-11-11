# tools/import_text_to_qdrant_with_txt.py

import hashlib
import os
import sys
import uuid
import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
import redis
from tqdm import tqdm

# --- 配置区 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import load_config

config = load_config()
RAG_CONFIG = config.get("rag", {})
REDIS_CONFIG = config.get("redis", {})

EMBEDDING_MODEL_NAME = RAG_CONFIG.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
QDRANT_URL = RAG_CONFIG.get("qdrant_url", "http://localhost:6333")
COLLECTION_NAME = RAG_CONFIG.get("collection_name", "default_collection")

# 针对TXT文件的父子切分策略
PARENT_CHUNK_SIZE = 800  # 父块可以稍大，以包含更多上下文
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 200   # 子块保持较小，用于精确检索
CHILD_CHUNK_OVERLAP = 50

# Redis 连接信息
REDIS_HOST = REDIS_CONFIG.get("host", "localhost")
REDIS_PORT = REDIS_CONFIG.get("port", 6379)
REDIS_DB = REDIS_CONFIG.get("db", 0)
REDIS_PARENT_DOCS_KEY_PREFIX = REDIS_CONFIG.get("parent_docs_key_prefix", "parent_docs:")
# --- 配置区结束 ---

def process_txt(file_path, redis_client):
    """
    处理单个TXT文件。将其分割成父文档块，再将每个父文档块分割成子文档块。
    """
    loader = TextLoader(file_path, encoding='utf-8')
    raw_doc = loader.load()[0]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    parent_chunks = parent_splitter.split_documents([raw_doc])
    
    child_docs_for_qdrant = []
    parent_docs_count = 0

    source_filename = os.path.basename(file_path)
    print(f"正在处理TXT文件: {source_filename}")
    
    for parent_chunk in tqdm(parent_chunks, desc=f"Splitting {source_filename}"):
        parent_content = parent_chunk.page_content
        # 根据父文档块的内容生成一个确定性的UUID作为其ID
        h = hashlib.sha256(parent_content.encode('utf-8'))
        parent_id = str(uuid.UUID(bytes=h.digest()[:16]))
        redis_key = f"{REDIS_PARENT_DOCS_KEY_PREFIX}{parent_id}"

        # 只有当这个ID不存在时，才添加到store中
        if not redis_client.exists(redis_key):
            redis_client.set(redis_key, parent_content)
            parent_docs_count += 1
        
        child_chunks = child_splitter.split_text(parent_content)
        
        for chunk in child_chunks:
            metadata = {
                "parent_id": parent_id,
                "parent_redis_key": redis_key,
                "source": source_filename
            }
            child_doc = Document(page_content=chunk, metadata=metadata)
            child_docs_for_qdrant.append(child_doc)
            
    return child_docs_for_qdrant, parent_docs_count

def main():
    """主函数，处理所有TXT文件并入库"""
    txt_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'txt')
    if not os.path.isdir(txt_dir):
        print(f"错误: TXT目录未找到于 {txt_dir}")
        return

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        redis_client.ping() # 检查连接
        print(f"成功连接到 Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        print(f"错误: 无法连接到 Redis. 请确保 Redis 服务正在运行. Error: {e}")
        return
    
    all_child_docs = []
    total_parent_docs = 0

    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_dir, filename)
            child_docs, parent_count = process_txt(file_path, redis_client)
            all_child_docs.extend(child_docs)
            total_parent_docs += parent_count

    if not all_child_docs:
        print("没有找到可处理的TXT文件。")
        return

    print(f"总计 {total_parent_docs} 个父文档已存入 Redis。")

    # 初始化向量数据库和嵌入模型
    print("初始化嵌入模型和Qdrant客户端...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    vector_size = embeddings._client.get_sentence_embedding_dimension()
    
    # 确保集合存在，先检查集合是否已存在
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"Qdrant集合 '{COLLECTION_NAME}' 已创建。")
        else:
            print(f"Qdrant集合 '{COLLECTION_NAME}' 已存在，跳过创建步骤。")
    except Exception as e:
        print(f"检查或创建集合时出错: {e}")
        return

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    
    # 将子文档批量上传到Qdrant
    print(f"准备将 {len(all_child_docs)} 个子文档从TXT上传到Qdrant...")
    # 新增：为每个文档生成确定性ID
    ids_for_qdrant = []
    for doc in all_child_docs:
        # 使用内容的SHA256哈希值作为唯一且稳定的ID
        h = hashlib.sha256(doc.page_content.encode('utf-8'))
        doc_id = str(uuid.UUID(bytes=h.digest()[:16]))
        ids_for_qdrant.append(doc_id)
    
    try:
        # 在 add_documents 时传入ids列表，使用更大的批处理大小提高性能
        vector_store.add_documents(all_child_docs, ids=ids_for_qdrant, batch_size=256)
        print("来自TXT的子文档已成功上传！")
    except Exception as e:
        print(f"上传文档到Qdrant时发生错误: {e}")
    finally:
        # 确保资源正确关闭
        if redis_client:
            redis_client.close()
            print("Redis连接已关闭")
    

if __name__ == "__main__":
    main()