"""
校园规章制度问答AI助手 — Streamlit Web 界面
"""
import os
import sys
import time
import streamlit as st

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Settings
from src.knowledge_base import KnowledgeBase
from src.retriever import Retriever
from src.llm_client import LLMClient
from src.conversation import ConversationHistory

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="校园规章制度问答助手",
    page_icon="📚",
    layout="wide",
)

# ==================== 初始化 ====================

@st.cache_resource
def init_knowledge_base():
    return KnowledgeBase()

def reset_conversation():
    st.session_state.conversation.clear()
    st.session_state.messages = []

# 初始化 session state
if "settings" not in st.session_state:
    st.session_state.settings = Settings()
if "kb" not in st.session_state:
    st.session_state.kb = init_knowledge_base()
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationHistory(max_size=20)
if "messages" not in st.session_state:
    st.session_state.messages = []  # UI 显示用 [{id, role, content, source}, ...]
if "editing_key" not in st.session_state:
    st.session_state.editing_key = False

settings = st.session_state.settings
kb = st.session_state.kb
conv = st.session_state.conversation

# ==================== 侧边栏 ====================

with st.sidebar:
    st.title("📚 校园规章制度助手")

    # --- 文档管理 ---
    st.header("📁 文档管理")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
    os.makedirs(data_dir, exist_ok=True)

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    uploaded_files = st.file_uploader(
        "上传制度文档（Word / PDF / TXT）",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
        help="支持批量上传",
        key="file_uploader_widget",
    )

    if uploaded_files:
        something_changed = False
        for uf in uploaded_files:
            # 跳过本周期已处理的文件（避免重复导入）
            if uf.name in st.session_state.processed_files:
                continue
            try:
                # 保存到磁盘
                tmp_path = os.path.join(data_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.getbuffer())

                # 导入知识库
                result = kb.add_document(tmp_path)
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    # 显示导入预览
                    with st.expander(f"📄 预览：{uf.name}", expanded=False):
                        for c in kb.chunks:
                            if c["filename"] == uf.name:
                                preview = c["text"][:200]
                                st.code(preview, language=None)
                                break
                    something_changed = True
                else:
                    st.warning(f"⚠️ {result['message']}")

                # 标记已处理
                st.session_state.processed_files.add(uf.name)
            except Exception as e:
                st.error(f"❌ 导入 '{uf.name}' 失败：{str(e)[:200]}")

        # 定期清理 processed_files（避免无限增长）
        if len(st.session_state.processed_files) > 100:
            st.session_state.processed_files.clear()

        if something_changed:
            st.rerun()

    # 已导入文档列表
    st.subheader("已导入文档")
    registry = kb.get_documents()
    if registry:
        for idx, doc in enumerate(registry):
            fname = doc.get("filename", "未知")
            ftype = doc.get("source_type", "")
            chunk_n = doc.get("chunk_count", 0)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {fname} ({ftype}, {chunk_n}块)")
            with col2:
                if st.button("🗑", key=f"del_{idx}_{fname[:20]}", help=f"删除 {fname}"):
                    result = kb.remove_document(fname)
                    if result["success"]:
                        st.success(result["message"])
                        # 清理已处理记录，允许重新导入
                        if fname in st.session_state.processed_files:
                            st.session_state.processed_files.discard(fname)
                        # 从文件系统中删除
                        doc_path = os.path.join(data_dir, fname)
                        if os.path.exists(doc_path):
                            try:
                                os.remove(doc_path)
                            except Exception:
                                pass
                    else:
                        st.error(result["message"])
                    st.rerun()
    else:
        st.caption("暂无文档，请上传制度文件")

    st.divider()

    # --- API 设置 ---
    st.header("⚙️ API 设置")

    # 开关
    api_enabled = st.toggle(
        "启用 API（在线模式）",
        value=settings.api_enabled,
        help="关闭后使用离线检索模式",
    )
    if api_enabled != settings.api_enabled:
        settings.api_enabled = api_enabled

    # API Key 输入
    st.caption("API Key")
    if st.session_state.editing_key:
        new_key = st.text_input(
            "输入 API Key",
            value=settings.api_key,
            type="password",
            key="api_key_input",
            label_visibility="collapsed",
        )
        if st.button("保存 Key", key="save_key"):
            settings.api_key = new_key
            st.session_state.editing_key = False
            st.rerun()
    else:
        masked = settings.mask_key()
        display_key = masked if masked else "未设置"
        st.code(display_key, language=None)
        if st.button("编辑 API Key", key="edit_key"):
            st.session_state.editing_key = True
            st.rerun()

    # Base URL
    base_url = st.text_input(
        "Base URL",
        value=settings.base_url,
        help="OpenAI 兼容 API 地址",
    )
    if base_url != settings.base_url:
        settings.base_url = base_url

    # Model
    model_name = st.text_input(
        "模型名称",
        value=settings.model_name,
        help="如 gpt-3.5-turbo / deepseek-chat / qwen-turbo",
    )
    if model_name != settings.model_name:
        settings.model_name = model_name

    st.divider()

    # --- 状态信息 ---
    st.header("📊 状态")
    stats = kb.get_stats()
    st.caption(f"文档数：{stats['doc_count']}")
    st.caption(f"文本块：{stats['chunk_count']}")
    st.caption(f"对话轮数：{len(conv) // 2}")
    mode = "🟢 在线" if settings.api_enabled else "🔵 离线"
    st.caption(f"当前模式：{mode}")

    st.divider()

    # --- 清空对话 ---
    if st.button("🗑 清空全部对话", use_container_width=True):
        conv.clear()
        st.session_state.messages = []
        st.rerun()

# ==================== 主区域 ====================

# 初始化和更新 retriever（每次渲染都重新引用，确保数据最新）
retriever = Retriever(kb)
llm_client = LLMClient(settings)

# 标题
st.title("📚 校园规章制度问答助手")
st.caption("上传校规文件后，即可提问。支持在线 AI 问答和离线检索两种模式。")

# 示例问题
if not st.session_state.messages:
    with st.expander("💡 示例问题（点击展开）"):
        examples = [
            "推免研究生的条件是什么？",
            "国家奖学金的申请流程是怎样的？",
            "考试作弊如何处理？",
            "你可以帮我做些什么？",
            "介绍一下学校的选课规则",
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"example_{i}"):
                st.session_state.example_question = ex

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 每条消息单独删除按钮
        col_text, col_btn = st.columns([20, 1])
        with col_btn:
            if st.button("✕", key=f"del_msg_{msg['id']}", help="删除此条消息"):
                conv.remove(msg["id"])
                st.session_state.messages = [
                    m for m in st.session_state.messages if m["id"] != msg["id"]
                ]
                st.rerun()

# 处理示例问题点击
if "example_question" in st.session_state and st.session_state.example_question:
    user_input = st.session_state.example_question
    st.session_state.example_question = ""
else:
    user_input = st.chat_input("请输入您的问题...")

# ==================== 核心逻辑 ====================

if user_input and user_input.strip():
    user_question = user_input.strip()

    # 1. 检索
    search_results = retriever.search(user_question, top_k=settings.top_k)
    has_relevant = len(search_results) > 0 and search_results[0].get("score", 0) >= 0.05
    context = retriever.format_context(search_results) if search_results else ""

    # 2. 生成回答（带 spinner）
    with st.spinner("思考中..."):
        if settings.api_enabled and settings.api_key:
            # 在线模式
            history = conv.get_context(n=10)
            reply = llm_client.generate(
                question=user_question,
                context=context,
                history=history,
                has_relevant=has_relevant,
            )
        else:
            # 离线模式
            if kb.has_documents():
                reply = retriever.format_offline_response(search_results)
            else:
                reply = "⚠️ 知识库为空，请先在侧边栏上传制度文档。开启 API 在线模式后可直接对话。"

    # 3. 更新对话历史
    user_id, assistant_id = conv.add(user_question, reply)

    # 4. 更新 UI
    st.session_state.messages.append({
        "id": user_id,
        "role": "user",
        "content": user_question,
    })
    st.session_state.messages.append({
        "id": assistant_id,
        "role": "assistant",
        "content": reply,
    })

    st.rerun()
