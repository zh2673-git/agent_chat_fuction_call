# Agent Chat with Function Calling

一个基于大语言模型（LLM）的智能对话Agent，支持动态工具调用（Function Calling）功能。

## 项目架构

```
agent_chat_fuction_call/
├── .env                    # 环境变量（API密钥等，从env.example复制）
├── env.example             # 环境变量示例模板
├── model.json              # 模型供应商配置（API URL、模型列表）
├── agent.py                # Agent核心逻辑（工具注册与调用）
├── LLM.py                  # 大语言模型封装（支持Function Calling）
├── main.py                 # 项目入口文件
└── tools/                  # 工具目录
    ├── __init__.py         # 空文件（Python包标识）
    ├── weather.py          # 示例：天气查询工具
    ├── translation.py      # 示例：翻译工具
    └── config.json         # 工具配置（API密钥等）
```

## 核心原理

1. **动态工具调用**：
   - Agent通过LLM的Function Calling能力，自动解析用户输入，决定是否调用工具（如天气、翻译）。
   - 工具调用结果由LLM整合后返回给用户。

2. **配置驱动**：
   - **模型配置**：`model.json`定义供应商（如OpenAI/DeepSeek）及其模型。
   - **工具配置**：`tools/config.json`注册工具参数。
   - **环境变量**：`.env`管理敏感信息（API密钥）。

3. **扩展性**：
   - 新增工具：在`tools/`下添加工具类，并在`config.json`注册。
   - 新增模型：在`model.json`中添加供应商配置。

## 快速开始

### 1. 安装依赖
```bash
pip install requests python-dotenv
```

### 2. 配置环境
复制`env.example`为`.env`并填写API密钥：
```env
# .env
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-3.5-turbo
```

### 3. 运行Agent
```bash
python main.py
```

## 示例交互
```python
# main.py
from agent import ChatAgent
agent = ChatAgent("openai")

# 调用天气工具
print(agent.run("北京天气怎么样？"))

# 调用翻译工具
print(agent.run("翻译成英文：你好！"))
```

## 注意事项
- 确保`.env`不被提交到版本控制（已默认在`.gitignore`中）。
- 工具类需继承`tools.base.Tool`并实现`execute`方法。

## 如何贡献
1. Fork 本仓库
2. 创建分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 提交 Pull Request

## 许可证
MIT License (见 `LICENSE` 文件)
