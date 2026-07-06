"""
文本处理模块
- 文本清洗
- 滑动窗口分块（保留文档名+章节信息）
- jieba 分词
"""
import re
import jieba
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    清洗文本：去除多余空白、空行、乱码字符
    """
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除每行首尾空白
    lines = [line.strip() for line in text.split("\n")]
    # 合并多个连续空行为一个
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    return "\n".join(cleaned_lines).strip()


def _detect_chapter(text: str) -> str:
    """
    尝试从文本中提取章节信息。
    匹配中文数字序号或"第X章/条/节"等模式。
    """
    patterns = [
        r"(第[一二三四五六七八九十百千\d]+章[^\n]*)",
        r"(第[一二三四五六七八九十百千\d]+条[^\n]*)",
        r"(第[一二三四五六七八九十百千\d]+节[^\n]*)",
        r"([一二三四五六七八九十]、[^\n]*)",
        r"(\d+[\.、][^\n]{2,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def split_chunks(document: Dict, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    滑动窗口分块。
    每个 chunk 包含：
      - text: 文本内容
      - filename: 来源文件名
      - source_type: 文档类型
      - chapter: 所在章节（如果有）
    """
    text = clean_text(document.get("text", ""))
    filename = document.get("filename", "未知文档")
    source_type = document.get("source_type", "unknown")

    chunks = []
    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]

        # 尝试在 chunk 边界处避免截断句子
        if end < text_len:
            # 在 chunk 末尾尽量断在句号/换行处
            last_period = max(
                chunk_text.rfind("。"),
                chunk_text.rfind("\n"),
                chunk_text.rfind("；"),
            )
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk_text = text[start:end]

        chapter = _detect_chapter(chunk_text)

        chunks.append({
            "id": f"{filename}_{chunk_index}",
            "text": chunk_text.strip(),
            "filename": filename,
            "source_type": source_type,
            "chapter": chapter,
        })

        chunk_index += 1
        start = end - overlap if end < text_len else end
        # 防止死循环
        if start >= end:
            start = end

    return chunks


def tokenize(text: str) -> List[str]:
    """
    jieba 分词，用于 TF-IDF 构建和关键词匹配。
    返回去停用词的词列表。
    """
    # 基础停用词（可根据需要扩展）
    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "所", "为", "所以", "因为", "但是", "然而", "而且", "或", "或者",
        "并", "并且", "虽然", "如果", "可以", "被", "把", "从", "对", "与",
        "及", "其", "等", "之", "中", "将", "已", "该", "此", "各", "每",
        "年", "月", "日", "时", "分", "第", "共", "条", "款", "项",
    }

    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and w.strip() not in stop_words and len(w.strip()) > 1]
