"""
对话历史管理器
- 基于 collections.deque 的固定容量队列
- 每条消息有唯一 ID，支持单独删除
- 支持上下文获取和清空
"""
import uuid
from collections import deque
from typing import List, Dict, Optional


class ConversationHistory:
    """对话历史管理器"""

    def __init__(self, max_size: int = 20):
        """
        max_size: 最大保留轮数（一"轮" = 用户消息 + 助手消息）
        """
        self.max_size = max_size
        # deque 存储格式:
        # [{"id": ..., "role": "user"/"assistant", "content": ..., "timestamp": ...}, ...]
        self._messages = deque(maxlen=max_size * 2)  # 每条消息独立存储

    def add(self, user_msg: str, assistant_msg: str) -> tuple:
        """
        添加一轮对话。
        返回 (user_msg_id, assistant_msg_id)
        """
        user_id = str(uuid.uuid4())[:8]
        assistant_id = str(uuid.uuid4())[:8]

        import time
        ts = time.strftime("%H:%M:%S")

        user_entry = {"id": user_id, "role": "user", "content": user_msg, "timestamp": ts}
        assistant_entry = {"id": assistant_id, "role": "assistant", "content": assistant_msg, "timestamp": ts}

        self._messages.append(user_entry)
        self._messages.append(assistant_entry)

        return user_id, assistant_id

    def remove(self, msg_id: str) -> bool:
        """
        按 ID 删除单条消息。
        同时尝试删除配对的消息（同一轮的 user/assistant）。
        返回是否删除成功。
        """
        target_msg = None
        for m in self._messages:
            if m["id"] == msg_id:
                target_msg = m
                break

        if target_msg is None:
            return False

        target_role = target_msg["role"]
        target_idx = None
        for i, m in enumerate(self._messages):
            if m["id"] == msg_id:
                target_idx = i
                break

        if target_idx is None:
            return False

        # 如果是 user 消息，尝试删除紧跟着的 assistant 消息（同一轮）
        # 如果是 assistant 消息，尝试删除紧跟着的 user 消息（同一轮）
        indices_to_remove = {target_idx}

        if target_role == "user" and target_idx + 1 < len(self._messages):
            if self._messages[target_idx + 1]["role"] == "assistant":
                indices_to_remove.add(target_idx + 1)
        elif target_role == "assistant" and target_idx > 0:
            if self._messages[target_idx - 1]["role"] == "user":
                indices_to_remove.add(target_idx - 1)

        # 从 deque 中移除（按索引从大到小移除，避免索引偏移）
        new_messages = deque(maxlen=self.max_size * 2)
        for i, m in enumerate(self._messages):
            if i not in indices_to_remove:
                new_messages.append(m)
        self._messages = new_messages

        return True

    def get_context(self, n: int = 10) -> List[Dict]:
        """
        返回最近 n 条消息，供 LLM 使用。
        格式：[{"role": "user"/"assistant", "content": ...}, ...]
        """
        messages = list(self._messages)
        context = []
        for m in messages[-n:]:
            context.append({"role": m["role"], "content": m["content"]})
        return context

    def get_all(self) -> List[Dict]:
        """返回全部消息（含 ID，供 UI 渲染）"""
        return list(self._messages)

    def clear(self):
        """一键清空全部历史"""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0
