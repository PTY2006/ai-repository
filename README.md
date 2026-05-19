# AI 智能体开发学习项目 – practice03

本项目在 `practice02` 的基础上实现了聊天记忆的**自动压缩**、**5W信息提取**与**智能检索**功能。

## 🚀 新特性

### 1. 自动上下文压缩（`llm_test_compressed.py`）
- 检测条件：对话轮数 > 5 **或** 历史消息总字符数 > 3000
- 压缩策略：将前70%的对话轮次交给LLM生成摘要，保留后30%的原始对话
- 压缩后对话历史变为：`(摘要) + 最近N轮完整对话`
- 保证长对话不会超限，同时保留关键信息

### 2. 5W信息持久化（`llm_test_search.py`）
- 每次完成历史压缩后，自动从当前对话中抽取 **Who / What / When / Where / Why**
- 信息追加写入 `D:\chat-log\log.txt`，并附带时间戳
- 形成可检索的结构化长期记忆

### 3. 智能检索（function call 风格）
- 模型可自主输出 `{"action": "search", "query": "..."}` 来请求检索历史日志
- 用户也可以直接使用 `/search <关键词>` 主动查找
- 系统读取 `log.txt` 并用LLM查找相关内容，最后结合用户原始问题生成回答

## 📦 环境要求
- Python 3.8+
- 根目录下配置 `.env` 文件（与 practice02 相同）：
  ```env
  LLM_BASE_URL=https://api.openai.com/v1
  LLM_MODEL=gpt-3.5-turbo
  LLM_API_KEY=sk-xxxx
  LLM_TEMPERATURE=0.7