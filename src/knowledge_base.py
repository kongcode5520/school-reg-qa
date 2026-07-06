"""
知识库管理模块
- 文档增删
- TF-IDF 索引构建与重建
- 持久化存储（JSON + pickle）
"""
import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer

from .document_loader import load_document
from .text_processor import split_chunks, tokenize

# 默认数据存储目录
def _default_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store")


def _file_hash(filepath: str) -> str:
    """计算文件的 MD5 哈希，用于重复检测"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class KnowledgeBase:
    """
    知识库：
      - chunks: List[Dict] — 所有文档分块（线性表）
      - vectorizer: TfidfVectorizer
      - tfidf_matrix: 稀疏矩阵
      - doc_registry: 已导入文档的注册信息
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = _default_data_dir()
        self.data_dir = data_dir
        self.registry_file = os.path.join(data_dir, "doc_registry.json")
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.index_file = os.path.join(data_dir, "index.pkl")

        os.makedirs(data_dir, exist_ok=True)

        self.chunks: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        # 尝试从磁盘加载已有数据
        self._load_from_disk()

    # ==================== 持久化 ====================

    def _load_from_disk(self):
        """从 data_store 加载已有知识库"""
        if os.path.exists(self.chunks_file):
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []

        if os.path.exists(self.index_file) and self.chunks:
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                    self.vectorizer = data.get("vectorizer")
                    self.tfidf_matrix = data.get("matrix")
            except Exception:
                self.vectorizer = None
                self.tfidf_matrix = None

        self._loaded = True

    def _save_to_disk(self):
        """保存知识库到磁盘"""
        try:
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                with open(self.index_file, "wb") as f:
                    pickle.dump({
                        "vectorizer": self.vectorizer,
                        "matrix": self.tfidf_matrix,
                    }, f)
            except Exception:
                pass

    # ==================== 文档管理 ====================

    def get_documents(self) -> List[Dict]:
        """
        返回已导入文档列表（从注册表读取）
        """
        if not os.path.exists(self.registry_file):
            return []
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_registry(self, registry: List[Dict]):
        """保存文档注册表"""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def add_document(self, filepath: str) -> Dict:
        """
        导入单个文档：
          1. 检测重复
          2. 解析（Word/PDF/TXT）
          3. 清洗分块
          4. 加入 chunks 列表
          5. 重建索引
          6. 持久化
        返回 {"success": bool, "filename": str, "chunk_count": int, "message": str}
        """
        import time
        filename = os.path.basename(filepath)

        # 检查是否已导入
        registry = self.get_documents()
        file_hash_val = _file_hash(filepath)
        for entry in registry:
            if entry.get("hash") == file_hash_val:
                return {
                    "success": False,
                    "filename": filename,
                    "chunk_count": 0,
                    "message": f"文档 '{filename}' 已存在（内容相同），无需重复导入",
                }

        # 等待文件系统完全释放句柄（Windows 兼容）
        time.sleep(0.3)

        # 解析文档
        doc = load_document(filepath)
        if doc is None:
            return {
                "success": False,
                "filename": filename,
                "chunk_count": 0,
                "message": f"无法解析文档 '{filename}'，请检查文件是否为有效的 Word/PDF/TXT 格式",
            }

        # 解析失败（返回了错误信息）
        if "error" in doc:
            return {
                "success": False,
                "filename": filename,
                "chunk_count": 0,
                "message": doc["error"],
            }

        # 分块
        new_chunks = split_chunks(doc)
        if not new_chunks:
            return {
                "success": False,
                "filename": filename,
                "chunk_count": 0,
                "message": f"文档 '{filename}' 内容为空",
            }

        # 加入 chunks 列表
        self.chunks.extend(new_chunks)

        # 更新注册表
        registry.append({
            "filename": filename,
            "hash": file_hash_val,
            "source_type": doc.get("source_type"),
            "chunk_count": len(new_chunks),
            "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save_registry(registry)

        # 重建索引
        self.rebuild_index()

        # 持久化
        self._save_to_disk()

        return {
            "success": True,
            "filename": filename,
            "chunk_count": len(new_chunks),
            "message": f"成功导入 '{filename}'，共 {len(new_chunks)} 个文本块",
        }

    def remove_document(self, filename: str) -> Dict:
        """
        从知识库中移除指定文档：
          1. 从 chunks 中移除所有属于该文件的块
          2. 从注册表中移除
          3. 重建索引
          4. 持久化
        返回 {"success": bool, "message": str}
        """
        # 更新注册表
        registry = self.get_documents()
        entry = None
        for e in registry:
            if e.get("filename") == filename:
                entry = e
                break

        if entry is None:
            return {"success": False, "message": f"文档 '{filename}' 不在知识库中"}

        new_registry = [e for e in registry if e.get("filename") != filename]
        self._save_registry(new_registry)

        # 从 chunks 中移除
        removed_count = 0
        new_chunks = []
        for chunk in self.chunks:
            if chunk.get("filename") == filename:
                removed_count += 1
            else:
                new_chunks.append(chunk)
        self.chunks = new_chunks

        # 重建索引
        if self.chunks:
            self.rebuild_index()
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

        # 持久化
        self._save_to_disk()

        return {
            "success": True,
            "message": f"已移除 '{filename}'（删除 {removed_count} 个文本块）",
        }

    # ==================== 索引管理 ====================

    def rebuild_index(self):
        """
        重建 TF-IDF 索引。
        对所有 chunks 的文本进行 jieba 分词后构建 TF-IDF 矩阵。
        """
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        # 对每个 chunk 做分词，用空格连接供 TfidfVectorizer 处理
        corpus = []
        for chunk in self.chunks:
            words = tokenize(chunk.get("text", ""))
            corpus.append(" ".join(words))

        # 构建 TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            sublinear_tf=True,  # 使用 1 + log(tf)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def get_stats(self) -> Dict:
        """返回知识库统计信息"""
        registry = self.get_documents()
        return {
            "doc_count": len(registry),
            "chunk_count": len(self.chunks),
            "total_chars": sum(len(c.get("text", "")) for c in self.chunks),
            "documents": [e.get("filename") for e in registry],
        }

    def has_documents(self) -> bool:
        """检查知识库是否为空"""
        return len(self.chunks) > 0
