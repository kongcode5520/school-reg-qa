"""
检索模块
- TF-IDF 余弦相似度检索
- jieba 关键词匹配（双路召回）
- 堆排序 top-k
"""
import heapq
from typing import List, Dict, Tuple

from .text_processor import tokenize


class Retriever:
    """检索引擎"""

    def __init__(self, knowledge_base):
        """
        knowledge_base: KnowledgeBase 实例
        """
        self.kb = knowledge_base

    # ==================== TF-IDF 检索 ====================

    def _tfidf_search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """
        TF-IDF + 余弦相似度检索。
        返回 [(score, chunk), ...] 按相似度降序排列
        """
        if self.kb.vectorizer is None or self.kb.tfidf_matrix is None:
            return []

        # 将查询转为分词后的字符串
        query_words = tokenize(query)
        if not query_words:
            return []
        query_text = " ".join(query_words)

        # 用已训练好的 vectorizer 转换 query
        try:
            query_vec = self.kb.vectorizer.transform([query_text])
        except Exception:
            return []

        # 计算余弦相似度
        similarities = (self.kb.tfidf_matrix * query_vec.T).toarray().flatten()

        # 堆排序取 top-k（小顶堆）
        heap = []
        for idx, score in enumerate(similarities):
            if score <= 0:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (score, idx))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, idx))

        # 降序排列
        results = []
        while heap:
            score, idx = heapq.heappop(heap)
            results.append((float(score), dict(self.kb.chunks[idx])))

        results.reverse()
        return results

    # ==================== 关键词匹配检索 ====================

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """
        基于 jieba 分词的倒排索引关键词匹配。
        作为 TF-IDF 的补充召回通路。
        返回 [(score, chunk), ...]
        """
        query_words = set(tokenize(query))
        if not query_words:
            return []

        scored = []
        for chunk in self.kb.chunks:
            chunk_words = set(tokenize(chunk.get("text", "")))
            if not chunk_words:
                continue
            # Jaccard 相似系数
            intersection = query_words & chunk_words
            if not intersection:
                continue
            union = query_words | chunk_words
            score = len(intersection) / len(union)
            if score > 0:
                scored.append((score, chunk))

        # 堆排序取 top-k（用 counter 做 tiebreaker 避免 dict 比较）
        heap = []
        counter = 0
        for item in scored:
            score = item[0]
            if len(heap) < top_k:
                heapq.heappush(heap, (score, counter, item))
                counter += 1
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, item))
                counter += 1

        results = []
        while heap:
            score, _, item = heapq.heappop(heap)
            results.append((float(score), dict(item[1])))

        results.reverse()
        return results

    # ==================== 综合检索 ====================

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        双路召回 + 去重合并 + 排序。
        1. TF-IDF 检索
        2. 关键词匹配检索
        3. 按 id 去重，取分数较高的
        4. 最终排序返回 top_k
        返回 [{"id": ..., "text": ..., "filename": ..., "chapter": ..., "score": ...}, ...]
        """
        tfidf_results = self._tfidf_search(query, top_k * 2)
        kw_results = self._keyword_search(query, top_k * 2)

        # 合并去重（同一个 chunk 取两者中较高的分数）
        merged = {}
        for score, chunk in tfidf_results:
            cid = chunk.get("id")
            merged[cid] = (score, chunk)

        for score, chunk in kw_results:
            cid = chunk.get("id")
            if cid in merged:
                # 取较高分：TF-IDF 得分 * 0.7 + 关键词得分 * 0.3
                existing_score, existing_chunk = merged[cid]
                merged[cid] = (max(existing_score, score * 0.3), existing_chunk)
            else:
                merged[cid] = (score * 0.3, chunk)

        # 按分数排序
        sorted_results = sorted(merged.values(), key=lambda x: x[0], reverse=True)

        # 取 top_k
        output = []
        for score, chunk in sorted_results[:top_k]:
            output.append({
                "id": chunk.get("id"),
                "text": chunk.get("text"),
                "filename": chunk.get("filename"),
                "chapter": chunk.get("chapter", ""),
                "score": round(score, 4),
            })

        return output

    def has_relevant_content(self, query: str, threshold: float = 0.05) -> bool:
        """判断是否有相关结果"""
        results = self.search(query, top_k=1)
        if not results:
            return False
        return results[0].get("score", 0) >= threshold

    def format_context(self, results: List[Dict]) -> str:
        """
        将检索结果格式化为 LLM 上下文。
        包含出处标注。
        """
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("filename", "未知文档")
            chapter = r.get("chapter", "")
            source_str = f"【{source}】" + (f" {chapter}" if chapter else "")
            parts.append(f"[参考资料 {i}] 来源：{source_str}\n{r.get('text', '')}")
        return "\n\n".join(parts)

    def format_offline_response(self, results: List[Dict]) -> str:
        """
        离线模式下格式化返回内容。
        直接返回检索到的原文片段 + 出处。
        """
        if not results:
            return "⚠️ 未能检索到有效信息。请确认相关制度文档已导入知识库。"

        parts = []
        for i, r in enumerate(results, 1):
            filename = r.get("filename", "未知文档")
            chapter = r.get("chapter", "")
            text = r.get("text", "")
            score = r.get("score", 0)

            chapter_str = f" — {chapter}" if chapter else ""
            parts.append(
                f"📄 **{filename}**{chapter_str}（相关度：{score:.0%}）\n\n"
                f"{text}"
            )

        return "\n\n---\n\n".join(parts)
