"""
测试检索模块
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBase
from src.retriever import Retriever


@pytest.fixture
def sample_kb():
    """创建包含测试数据的知识库"""
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    kb = KnowledgeBase(data_dir=tmp_dir)
    kb._tmp_dir = tmp_dir
    kb.chunks = [
        {
            "id": "doc1_0",
            "text": "推免研究生条件：学分绩点不低于3.5，无不及格科目，通过英语六级。",
            "filename": "推免办法.docx",
            "source_type": "docx",
            "chapter": "第二章 推免条件",
        },
        {
            "id": "doc1_1",
            "text": "推免流程：学生提交申请后，由学院推免工作小组进行资格审查和综合排名。",
            "filename": "推免办法.docx",
            "source_type": "docx",
            "chapter": "第三章 推免流程",
        },
        {
            "id": "doc2_0",
            "text": "国家奖学金奖励标准为每生每年8000元，评审时间为每年9-10月。",
            "filename": "奖学金管理办法.pdf",
            "source_type": "pdf",
            "chapter": "第三条",
        },
        {
            "id": "doc3_0",
            "text": "考试作弊者，取消该科目成绩，并视情节给予严重警告至开除学籍处分。",
            "filename": "考试违规处理办法.txt",
            "source_type": "txt",
            "chapter": "",
        },
    ]
    kb.rebuild_index()
    return kb


class TestTFIDFSearch:
    """TF-IDF 检索测试"""

    def test_search_relevant(self, sample_kb):
        """测试相关检索"""
        retriever = Retriever(sample_kb)
        results = retriever.search("推免需要什么条件", top_k=3)
        assert len(results) > 0
        # 最相关的结果应该包含推免相关内容
        found = any("推免" in r["text"] for r in results)
        assert found

    def test_search_returns_correct_fields(self, sample_kb):
        """测试返回字段完整性"""
        retriever = Retriever(sample_kb)
        results = retriever.search("奖学金", top_k=2)
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "filename" in r
            assert "score" in r

    def test_search_empty_query(self, sample_kb):
        """测试空查询不会崩溃"""
        retriever = Retriever(sample_kb)
        results = retriever.search("   ", top_k=3)
        assert isinstance(results, list)

    def test_has_relevant_content(self, sample_kb):
        """测试相关性判断"""
        retriever = Retriever(sample_kb)
        assert retriever.has_relevant_content("推免条件")
        # 一个肯定不相关的问题
        assert not retriever.has_relevant_content("火星土壤成分分析", threshold=0.1)

    def test_top_k_limit(self, sample_kb):
        """测试 top_k 限制"""
        retriever = Retriever(sample_kb)
        results = retriever.search("考试", top_k=2)
        assert len(results) <= 2


class TestKeywordSearch:
    """关键词匹配测试"""

    def test_keyword_match_finds(self, sample_kb):
        """测试关键词匹配能找到结果"""
        retriever = Retriever(sample_kb)
        results = retriever._keyword_search("作弊处分", top_k=3)
        assert len(results) > 0

    def test_keyword_match_no_result(self, sample_kb):
        """测试无匹配关键词"""
        retriever = Retriever(sample_kb)
        results = retriever._keyword_search("火星探险", top_k=3)
        assert len(results) == 0


class TestFormat:
    """格式化输出测试"""

    def test_format_context(self, sample_kb):
        """测试上下文格式化"""
        retriever = Retriever(sample_kb)
        results = retriever.search("推免", top_k=2)
        context = retriever.format_context(results)
        assert "参考资料" in context
        assert "推免办法.docx" in context

    def test_format_offline_response(self, sample_kb):
        """测试离线响应格式化"""
        retriever = Retriever(sample_kb)
        results = retriever.search("推免", top_k=2)
        response = retriever.format_offline_response(results)
        assert "推免" in response

    def test_format_offline_empty(self, sample_kb):
        """测试离线模式无结果时的响应"""
        retriever = Retriever(sample_kb)
        response = retriever.format_offline_response([])
        assert "未能检索到" in response


class TestEmptyKB:
    """空知识库测试"""

    def test_search_empty_kb(self):
        """测试空知识库不会崩溃"""
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=tmp_dir)
        kb.chunks = []
        kb.rebuild_index()
        retriever = Retriever(kb)
        results = retriever.search("推免条件")
        assert results == []
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_has_relevant_empty_kb(self):
        """测试空知识库的相关性判断"""
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=tmp_dir)
        kb.chunks = []
        kb.rebuild_index()
        retriever = Retriever(kb)
        assert not retriever.has_relevant_content("任何问题")
        shutil.rmtree(tmp_dir, ignore_errors=True)
