"""
大模型 API 调用客户端
- OpenAI 兼容接口
- 容错处理（超时自动降级）
- Prompt 构造（含检索结果上下文和对话历史）
"""
from typing import List, Dict, Optional


class LLMClient:
    """大模型 API 客户端"""

    def __init__(self, settings):
        """
        settings: config.Settings 实例
        """
        self.settings = settings

    def _create_client(self):
        """创建 OpenAI 兼容客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=15.0,
            max_retries=1,
        )

    def validate(self) -> str:
        """
        校验 API 配置，返回错误信息字符串。
        无错误返回空字符串。
        """
        if not self.settings.api_key:
            return "API Key 未设置"
        if not self.settings.model_name.strip():
            return "模型名称未设置，请在侧边栏填写（如 gpt-3.5-turbo / deepseek-chat）"
        return ""

    def generate(
        self,
        question: str,
        context: str = "",
        history: List[Dict] = None,
        has_relevant: bool = True,
    ) -> str:
        """
        调用大模型生成回答。

        参数:
          - question: 用户当前问题
          - context: 检索到的制度条文上下文（含出处）
          - history: 对话历史 [{"role": "user", "content": ...}, ...]
          - has_relevant: 是否检索到了相关内容

        返回: 模型回答文本
        """
        # 离线模式
        if not self.settings.api_enabled:
            return self._offline_fallback(context, has_relevant)

        # 配置校验
        validation_error = self.validate()
        if validation_error:
            return f"⚠️ {validation_error}\n\n{self._offline_fallback(context, has_relevant)}"

        messages = self._build_messages(question, context, history, has_relevant)

        try:
            client = self._create_client()
            model = self.settings.model_name.strip()
            # 构造 API 调用参数
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": False,
            }
            # DeepSeek v4-pro / r1 模型启用深度思考
            if any(m in model for m in ("deepseek-v4", "deepseek-r1", "deepseek-reasoner")):
                api_params["reasoning_effort"] = "medium"
                api_params["extra_body"] = {"thinking": {"type": "enabled"}}
            response = client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)[:120]
            return (
                f"⚠️ API 调用失败（{error_msg}），已自动降级为离线模式。\n\n"
                f"{self._offline_fallback(context, has_relevant)}"
            )

    def _build_messages(
        self,
        question: str,
        context: str,
        history: List[Dict],
        has_relevant: bool,
    ) -> List[Dict]:
        """构建发送给 LLM 的消息列表"""

        system_prompt = """你是一个校园规章制度问答助手。你的职责是根据提供的制度文档内容来回答学生的问题。

规则：
1. 如果参考资料中有相关内容，请基于资料回答问题，并在回答末尾注明出处（文档名+章节）。
2. 如果参考资料中没有相关内容，但问题是闲聊性质（如问候、自我介绍询问），可以自由简短回答。
3. 如果问题是制度相关问题但没有找到对应资料，请明确回复"未能检索到有效信息，建议查阅相关规章制度原件。"
4. 回答要简洁、准确，不要编造资料中没有的信息。
5. 用中文回答。"""

        messages = [{"role": "system", "content": system_prompt}]

        # 注入对话历史（最近 N 轮）
        if history:
            for h in history[-10:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})

        # 构造用户消息（含检索上下文）
        if context:
            user_message = f"""参考资料：
{context}

---
用户问题：{question}

请根据以上参考资料回答问题。"""
        else:
            user_message = question

        messages.append({"role": "user", "content": user_message})

        return messages

    def _offline_fallback(self, context: str, has_relevant: bool) -> str:
        """离线模式下的响应"""
        if not context:
            return "⚠️ 未能检索到有效信息。请确认相关制度文档已导入知识库。"
        return f"📋 *以下为检索到的原文片段：*\n\n{context}"
