# 校园规章制度问答AI助手 📚

基于 Streamlit + TF-IDF + jieba 的校园规章制度智能检索问答系统。支持**在线 AI 模式**（调用大模型 API）和**离线检索模式**，开箱即用。

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/kongcode5520/school-reg-qa.git
cd school-reg-qa
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

### 3. 安装依赖

```bash
.venv\Scripts\pip install -r requirements.txt
```

### 4. 复制配置文件

```bash
copy settings.example.json settings.json
```

### 5. （可选）配置在线 AI 模式

编辑 `settings.json`，填入你的 API Key：

```json
{
  "api_key": "sk-你的密钥",
  "base_url": "https://api.deepseek.com",
  "model_name": "deepseek-chat",
  "api_enabled": true,
  "top_k": 5
}
```

支持所有 OpenAI 兼容接口（DeepSeek、通义千问、GLM 等）。

### 6. 启动

```bash
.venv\Scripts\python run.py
```

浏览器会自动打开 http://localhost:8501

## 📖 使用说明

1. 在左侧边栏**上传制度文档**（支持 Word / PDF / TXT）
2. 在输入框**输入问题**即可查询
3. **在线模式**：开启 API 后由大模型生成智能回答
4. **离线模式**：直接从文档检索最相关原文片段

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| 搜索引擎 | TF-IDF（scikit-learn） + Jaccard 关键词 |
| 中文分词 | jieba |
| 文档解析 | python-docx / PyMuPDF / chardet |
| LLM 调用 | OpenAI SDK（兼容接口） |
| 数据结构 | 线性表、堆排序、倒排索引、固定容量队列 |

## 📁 项目结构

```
school-reg-qa/
├── app.py                  # Streamlit 主界面
├── config.py               # 配置管理（API Key 加密存储）
├── run.py                  # 一键启动入口
├── requirements.txt        # 依赖列表
├── settings.example.json   # 配置文件模板
├── src/
│   ├── knowledge_base.py   # 知识库管理（文档增删、索引构建）
│   ├── retriever.py        # 双路检索引擎（TF-IDF + 关键词）
│   ├── llm_client.py       # 大模型 API 客户端
│   ├── document_loader.py  # 文档解析器（Word/PDF/TXT）
│   ├── text_processor.py   # 文本清洗、分词、分块
│   └── conversation.py     # 对话历史管理
└── tests/                  # 单元测试
```

## ❓ 常见问题

**Q: 启动报错端口被占用？**
修改 `run.py` 中的 `--server.port 8501` 为其他端口。

**Q: 离线模式找不到内容？**
先确认已在侧边栏上传了相关制度文档，且文档内容清晰可读。
