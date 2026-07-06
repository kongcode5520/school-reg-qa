"""
测试文档加载器
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import (
    load_txt,
    load_word,
    load_pdf,
    load_document,
    load_documents_from_dir,
    _detect_encoding,
)


class TestTXTLoader:
    """TXT 文件加载测试"""

    def test_load_txt_utf8(self):
        """测试加载 UTF-8 编码的 TXT 文件"""
        content = "第一条 本条例适用于全体在校学生。\n第二条 学生应当遵守校规校纪。"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = load_txt(tmp_path)
            assert result["filename"].endswith(".txt")
            assert "第一条" in result["text"]
            assert "第二条" in result["text"]
            assert result["source_type"] == "txt"
        finally:
            os.unlink(tmp_path)

    def test_load_txt_gbk(self):
        """测试加载 GBK 编码的 TXT 文件"""
        content = "奖学金评定细则\n第一条 学分绩点不低于3.0"
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            f.write(content.encode("gbk"))
            tmp_path = f.name

        try:
            result = load_txt(tmp_path)
            assert "奖学金" in result["text"]
            assert result["source_type"] == "txt"
        finally:
            os.unlink(tmp_path)

    def test_encoding_detection(self):
        """测试编码自动检测"""
        content = "测试内容"
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            f.write(content.encode("utf-8"))
            tmp_path = f.name

        try:
            encoding = _detect_encoding(tmp_path)
            assert encoding is not None
            assert len(encoding) > 0
        finally:
            os.unlink(tmp_path)


class TestLoadDocument:
    """自动路由测试"""

    def test_load_document_txt(self):
        """测试自动识别 TXT 文件"""
        content = "测试规章制度内容"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = load_document(tmp_path)
            assert result is not None
            assert "测试规章制度内容" in result["text"]
            assert result["source_type"] == "txt"
        finally:
            os.unlink(tmp_path)

    def test_load_document_unsupported(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("test")
            tmp_path = f.name

        try:
            result = load_document(tmp_path)
            assert result is not None
            assert "error" in result
        finally:
            os.unlink(tmp_path)

    def test_load_document_nonexistent(self):
        """测试不存在的文件"""
        result = load_document("/nonexistent/path/12345.docx")
        # 返回 None（文件不存在）或 error dict 都视为正确处理
        assert result is None or "error" in result


class TestBatchLoad:
    """批量加载测试"""

    def test_load_from_dir(self):
        """测试从目录批量加载"""
        tmp_dir = tempfile.mkdtemp()
        try:
            # 创建测试文件
            with open(os.path.join(tmp_dir, "test1.txt"), "w", encoding="utf-8") as f:
                f.write("测试制度一")
            with open(os.path.join(tmp_dir, "test2.txt"), "w", encoding="utf-8") as f:
                f.write("测试制度二")

            results, errors = load_documents_from_dir(tmp_dir)
            assert len(results) == 2
            assert len(errors) == 0
        finally:
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)

    def test_load_from_nonexistent_dir(self):
        """测试不存在的目录"""
        results, errors = load_documents_from_dir("/nonexistent_path_12345/")
        assert len(results) == 0
        assert len(errors) == 0
