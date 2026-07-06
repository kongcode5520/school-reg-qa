"""
测试对话历史管理器
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation import ConversationHistory


class TestConversationBasics:
    """基础功能测试"""

    def test_add_and_get(self):
        """测试添加对话和获取上下文"""
        conv = ConversationHistory(max_size=20)
        conv.add("推免条件是什么？", "推免需要学分绩点不低于3.5。")
        context = conv.get_context(10)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "推免条件是什么？"
        assert context[1]["role"] == "assistant"
        assert "3.5" in context[1]["content"]

    def test_get_all_returns_ids(self):
        """测试 getAll 返回消息 ID"""
        conv = ConversationHistory(max_size=20)
        uid, aid = conv.add("问题", "回答")
        messages = conv.get_all()
        assert len(messages) == 2
        assert messages[0]["id"] == uid
        assert messages[1]["id"] == aid

    def test_clear(self):
        """测试清空"""
        conv = ConversationHistory(max_size=20)
        conv.add("问题1", "回答1")
        conv.add("问题2", "回答2")
        assert len(conv) == 4
        conv.clear()
        assert len(conv) == 0
        assert conv.is_empty

    def test_is_empty(self):
        """测试空判断"""
        conv = ConversationHistory(max_size=20)
        assert conv.is_empty
        conv.add("问题", "回答")
        assert not conv.is_empty


class TestRemove:
    """删除功能测试"""

    def test_remove_user_msg_also_removes_assistant(self):
        """删除用户消息时，配对的助手消息也被删除"""
        conv = ConversationHistory(max_size=20)
        uid, aid = conv.add("问题A", "回答A")
        conv.add("问题B", "回答B")

        assert len(conv) == 4
        result = conv.remove(uid)
        assert result is True
        assert len(conv) == 2  # 删了2条
        remaining = conv.get_all()
        assert remaining[0]["content"] == "问题B"

    def test_remove_assistant_msg_also_removes_user(self):
        """删除助手消息时，配对的用户消息也被删除"""
        conv = ConversationHistory(max_size=20)
        conv.add("问题A", "回答A")
        uid_b, aid_b = conv.add("问题B", "回答B")

        assert len(conv) == 4
        conv.remove(aid_b)
        assert len(conv) == 2
        remaining = conv.get_all()
        assert remaining[0]["content"] == "问题A"

    def test_remove_nonexistent(self):
        """删除不存在的 ID"""
        conv = ConversationHistory(max_size=20)
        conv.add("问题", "回答")
        assert not conv.remove("fake_id_12345")


class TestMaxSize:
    """容量限制测试"""

    def test_max_size_enforced(self):
        """测试 deque 容量限制：超出后旧消息被移除"""
        conv = ConversationHistory(max_size=3)  # 最多 3 轮 = 6 条消息
        for i in range(5):  # 5 轮 = 10 条 > 6 条上限
            conv.add(f"问题{i}", f"回答{i}")

        assert len(conv) <= 6
        # 最旧的消息应该已被丢弃
        all_msgs = conv.get_all()
        contents = [m["content"] for m in all_msgs]
        assert "问题0" not in contents
        assert "问题4" in contents


class TestContext:
    """上下文获取测试"""

    def test_get_context_limit(self):
        """测试上下文数量限制"""
        conv = ConversationHistory(max_size=20)
        for i in range(5):
            conv.add(f"问题{i}", f"回答{i}")
        # 只取最近 4 条
        context = conv.get_context(n=4)
        assert len(context) == 4
        assert context[0]["content"] == "问题3"

    def test_get_context_roles(self):
        """测试上下文保留正确角色"""
        conv = ConversationHistory(max_size=20)
        conv.add("用户问题", "助手回答")
        context = conv.get_context(10)
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
