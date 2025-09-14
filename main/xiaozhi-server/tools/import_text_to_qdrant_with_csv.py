# tools/import_text_to_qdrant.py

"""
该代码用于处理csv文件的数据
"""

import os
import sys
import glob
from datetime import datetime
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
import re
import yaml

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 构建配置文件路径
config_path = os.path.join(project_root, "config.yaml")

# with open(config_path, "r", encoding="utf-8") as file:
#     config = yaml.safe_load(file)

from config.config_loader import load_config

config = load_config()

# 从配置中获取RAG相关设置
RAG_CONFIG = config.get("rag", {})
CHUNK_SIZE = RAG_CONFIG.get("chunk_size", 500)
CHUNK_OVERLAP = RAG_CONFIG.get("chunk_overlap", 50)
EMBEDDING_MODEL_NAME = RAG_CONFIG.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
QDRANT_URL = RAG_CONFIG.get("qdrant_url", "http://localhost:6333")
COLLECTION_NAME = RAG_CONFIG.get("collection_name", "default_collection")

def split_text_by_headings_with_metadata(original_docs: list[Document]) -> list[Document]:
    """
    根据标题分割，并为每个块添加父标题作为元数据。
    """
    new_docs = []
    if not original_docs:
        return []
        
    full_text = original_docs[0].page_content
    original_metadata = original_docs[0].metadata

    # 按换行符分割成行
    lines = full_text.split('\n')
    
    # 找到主标题，通常是第一个非空行
    parent_heading = ""
    for line in lines:
        if line.strip():
            parent_heading = line.strip()
            break
            
    # 使用与之前相同的正则表达式进行分割
    chunks = re.split(r'(?=\n\d\.\d\s)', full_text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    for chunk_content in chunks:
        # 提取当前块自己的标题
        # 通常是内容的第一行
        current_heading = chunk_content.split('\n')[0].strip()

        new_meta = original_metadata.copy()
        # 在元数据中添加层级信息
        new_meta["parent_heading"] = parent_heading
        new_meta["current_heading"] = current_heading
        
        new_doc = Document(
            page_content=chunk_content,
            metadata=new_meta
        )
        new_docs.append(new_doc)
        
    return new_docs

def import_csv_to_qdrant(file_path: str, collection_name: str, qdrant_url: str = QDRANT_URL):
    """
    从文本文件中加载文档信息，
    切分后导入 Qdrant 向量数据库，并在 metadata 中记录更新时间。
    :param file_path: 文本文件路径
    :param qdrant_url: Qdrant 服务地址
    :return: Qdrant 向量库实例
    """
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()  # 返回 Document 对象列表

    # 2. 切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
        )
    docs = text_splitter.split_documents(documents)

    # 3. 在每个文档块的 metadata 中添加更新时间和类别信息
    current_date = datetime.now().strftime("%Y-%m-%d")
    for doc in docs:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["updated_at"] = current_date

    # 4. 初始化中文嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
        )

    # 5. 创建 Qdrant 向量库
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=qdrant_url,
        collection_name=collection_name
    )

    print(f"成功导入 {len(docs)} 个文本块到 Qdrant，更新时间：{current_date}。")

def import_txt_to_qdrant(file_path: str, collection_name: str, qdrant_url: str = QDRANT_URL):
    """专门处理txt文件的版本"""

    # 1. 加载整个文件为一个 Document
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()

    # 2. 使用我们自定义的函数按标题切分文本
    docs = split_text_by_headings_with_metadata(documents)

    # 3. 在每个文档块的 metadata 中添加更新时间
    current_date = datetime.now().strftime("%Y-%m-%d")
    for doc in docs:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["updated_at"] = current_date
        print(f"--- 切分出的块 ---\n{doc.page_content}\n-----------------\n") # 打印出来检查一下

    # 4. 初始化中文嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 5. 创建 Qdrant 向量库
    if docs: # 确保有内容再导入
        vectorstore = Qdrant.from_documents(
            documents=docs,
            embedding=embedding_model,
            url=qdrant_url,
            collection_name=collection_name
        )
        print(f"成功导入 {len(docs)} 个文本块到 Qdrant 的集合 {collection_name}，更新时间：{current_date}。")
    else:
        print("没有可导入的文本块。")

if __name__ == "__main__":
    # 假设文本文档位于 data/txt/sample.txt
    data_dir = os.path.join(project_root, "data")
    files_path = glob.glob(os.path.join(data_dir, "txt", "*.txt"))
    for file_path in files_path:
        import_txt_to_qdrant(file_path, COLLECTION_NAME, QDRANT_URL)