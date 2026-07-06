"""
测试知识库管理
"""
import os
import sys
import tempfile
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBase
from src.text_processor import clean_text, split_chunks, tokenize


class TestTextProcessor:
    """文本处理测试"""

    def test_clean_text_strips_whitespace(self):
        """测试去除多余空白"""
        raw = "  第一条  内容  \n\n\n第二条  内容  "
        cleaned = clean_text(raw)
        assert cleaned == "第一条  内容\n\n第二条  内容"

    def test_split_chunks(self):
        """测试文本分块"""
        doc = {
            "filename": "test.docx",
            "text": "A" * 600 + "\n" + "B" * 400,
            "source_type": "docx",
        }
        chunks = split_chunks(doc, chunk_size=500, overlap=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert chunk["filename"] == "test.docx"
            assert chunk["source_type"] == "docx"

    def test_split_chunks_empty(self):
        """测试空文本分块"""
        doc = {"filename": "empty.txt", "text": "", "source_type": "txt"}
        chunks = split_chunks(doc)
        assert len(chunks) == 0

    def test_tokenize_removes_stopwords(self):
        """测试分词去停用词"""
        words = tokenize("我是一名学生")
        # "我"、"是"、"一" 是停用词，应该被过滤
        assert "学生" in words
        assert "我" not in words
        assert "是" not in words


class TestKnowledgeBase:
    """知识库管理测试 — 每个测试使用独立的临时目录避免互相干扰"""

    @staticmethod
    def _make_kb():
        """创建使用临时目录的 KnowledgeBase"""
        tmp_dir = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=tmp_dir)
        # 把临时目录记下来供清理
        kb._tmp_dir = tmp_dir
        return kb

    def test_init_empty(self):
        """测试空知识库初始化"""
        kb = self._make_kb()
        try:
            assert not kb.has_documents()
            assert kb.chunks == []
        finally:
            shutil.rmtree(kb._tmp_dir, ignore_errors=True)

    def test_add_txt_document(self):
        """测试添加 TXT 文档"""
        content = "第一条 本条例适用于全体在校学生。\n第二条 学生应遵守校规校纪。"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            kb = self._make_kb()
            try:
                result = kb.add_document(tmp_path)
                assert result["success"]
                assert result["chunk_count"] > 0
                assert kb.has_documents()
            finally:
                shutil.rmtree(kb._tmp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_add_duplicate_document(self):
        """测试重复导入检测"""
        content = "测试规章制度内容"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            kb = self._make_kb()
            try:
                result1 = kb.add_document(tmp_path)
                assert result1["success"]
                result2 = kb.add_document(tmp_path)
                assert not result2["success"]
                assert "已存在" in result2["message"]
            finally:
                shutil.rmtree(kb._tmp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_remove_document(self):
        """测试删除文档"""
        content = "测试制度文本"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            kb = self._make_kb()
            try:
                result = kb.add_document(tmp_path)
                filename = result["filename"]
                old_count = kb.get_stats()["chunk_count"]

                remove_result = kb.remove_document(filename)
                assert remove_result["success"]
                new_count = kb.get_stats()["chunk_count"]
                assert new_count < old_count
            finally:
                shutil.rmtree(kb._tmp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_remove_nonexistent(self):
        """测试删除不存在的文档"""
        kb = self._make_kb()
        try:
            result = kb.remove_document("不存在的文件.docx")
            assert not result["success"]
        finally:
            shutil.rmtree(kb._tmp_dir, ignore_errors=True)

    def test_get_stats(self):
        """测试统计信息"""
        content = "测试制度"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            kb = self._make_kb()
            try:
                kb.add_document(tmp_path)
                stats = kb.get_stats()
                assert stats["doc_count"] >= 1
                assert stats["chunk_count"] >= 1
                assert "documents" in stats
            finally:
                shutil.rmtree(kb._tmp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_rebuild_index_after_remove(self):
        """测试删除后索引重建"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("包含推免条件的内容：学分绩点不低于3.5")
            tmp_path1 = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("包含奖学金的内容：每生每年8000元")
            tmp_path2 = f.name

        try:
            kb = self._make_kb()
            try:
                result1 = kb.add_document(tmp_path1)
                result2 = kb.add_document(tmp_path2)

                # 删除第一个文档后，索引应该重建
                kb.remove_document(result1["filename"])
                assert kb.vectorizer is not None  # 还有第二个文档的索引
                assert kb.tfidf_matrix is not None
            finally:
                shutil.rmtree(kb._tmp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path1)
            os.unlink(tmp_path2)
