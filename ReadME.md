# Agent Chat with Function Calling

一个基于大语言模型（LLM）的智能对话Agent，支持动态工具调用（Function Calling）和流式输出功能。通过简洁的架构设计，实现多模型供应商支持和灵活的工具扩展能力。

## 主要特性

- **多模型供应商支持**：支持OpenAI、ModelScope等多种模型供应商，可轻松扩展
- **智能工具调用**：根据用户意图自动选择并调用合适的工具
- **流式输出**：支持流式生成响应，提供更自然的对话体验
- **可扩展工具集**：易于添加自定义工具，不需修改核心代码
- **自动参数提取**：从用户输入中智能提取工具所需参数
- **健壮的错误处理**：优雅处理各类异常情况
- **清晰的代理层设计**：LLM作为代理层，简化模型切换和管理

## 项目架构

```
agent_chat_fuction_call/
├── .env                    # 环境变量（API密钥等，从env.example复制）
├── env.example             # 环境变量示例模板
├── agent.py                # Agent核心逻辑（工具注册与调用）
├── LLM.py                  # 大语言模型封装（作为模型供应商的代理层）
├── main.py                 # 项目入口文件
├── config/                 # 配置文件目录
│   └── provider_config.json # 供应商配置（API URL、支持模型列表）
├── providers/              # 模型供应商实现
│   ├── __init__.py         # 供应商导出
│   ├── base.py             # 供应商基类定义
│   ├── openai.py           # OpenAI API实现
│   └── modelscope.py       # ModelScope API实现
└── tools/                  # 工具目录
    ├── __init__.py         # 工具包初始化
    ├── tool.py             # 工具基类（自动参数提取）
    ├── weather.py          # 示例：天气查询工具
    ├── translation.py      # 示例：翻译工具
    └── config.json         # 工具配置（API密钥等）
```

## 核心原理

### 1. 代理层设计
LLM类作为代理层，负责连接Agent和各模型供应商，不处理具体的工具逻辑，只负责消息传递。

### 2. 工具调用流程
1. 用户向Agent提问
2. Agent准备工具列表，传递给LLM
3. LLM将请求转发给对应的模型提供商
4. 模型决定是否调用工具，返回工具调用信息
5. Agent执行工具，将结果再次传给LLM
6. LLM生成最终回复

### 3. 配置驱动架构
- **供应商配置**：`config/provider_config.json`定义供应商、支持的模型和默认设置
- **工具配置**：`tools/config.json`注册工具参数
- **环境变量**：`.env`管理敏感信息（API密钥）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制`env.example`为`.env`并填写API密钥：

```bash
# .env
MODELSCOPE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. 运行Agent

```bash
# 正常模式
python main.py

# 调试模式（显示详细日志）
python main.py -d

# 指定模型供应商和模型
python main.py -p modelscope -m "deepseek-ai/DeepSeek-V3-0324"
```

## 示例交互

```
用户输入: 北京今天天气怎么样
Agent回复: [工具调用] 使用工具: weather
[工具参数] {"city": "北京"}
[工具结果] [TEST] Weather for 北京: Sunny, 25°C
[模型回复] 北京今天天气晴朗，温度是25°C，很适合外出活动！

用户输入: 翻译"你好"成英文
Agent回复: [工具调用] 使用工具: translation
[工具参数] {"text": "你好", "target_lang": "en"}
[工具结果] [TEST] Translation result: Hello
[模型回复] "你好"翻译成英文是"Hello"。
```

## 如何添加新工具

1. **创建工具类**：在`tools/`目录下创建新的工具类，继承`Tool`基类：

```python
from .tool import Tool

class MyNewTool(Tool):
    def __init__(self, config):
        super().__init__(config)
        self.description = "这是一个新工具的描述"
        
    def execute(self, param1: str, param2: int = 0) -> str:
        """
        工具说明（将用作模型的工具描述）
        
        :param param1: 参数1的说明
        :param param2: 参数2的说明
        :return: 执行结果
        """
        # 工具实现逻辑
        return f"Result: {param1}, {param2}"
```

2. **注册工具**：在`tools/config.json`中添加工具配置：

```json
{
  "my_new_tool": {
    "api_key": "your_api_key",
    "api_url": "https://api.example.com/v1/endpoint"
  }
}
```

3. **重启Agent**：重新启动应用程序以加载新工具

## 如何添加新的模型供应商

1. **创建供应商类**：在`providers/`目录下创建新的供应商类，继承`BaseProvider`：

```python
from .base import BaseProvider
import os

class NewProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        return cls(
            api_key=os.getenv("NEW_PROVIDER_API_KEY", config.get("api_key")),
            api_url=config.get("api_url", "https://api.default.com/v1"),
            model_name=config.get("model_name", "default-model")
        )
        
    def __init__(self, api_key: str, api_url: str, model_name: str):
        # 初始化代码
        
    def generate_response(self, prompt: str, tools: list, stream: bool = False):
        # 实现响应生成逻辑
        # 返回格式必须符合BaseProvider接口规范
```

2. **更新供应商导出**：在`providers/__init__.py`中导出新供应商：

```python
from .base import BaseProvider
from .openai import OpenaiProvider
from .modelscope import ModelscopeProvider
from .new_provider import NewProvider

__all__ = ['BaseProvider', 'OpenaiProvider', 'ModelscopeProvider', 'NewProvider']
```

3. **添加供应商配置**：在`config/provider_config.json`中添加供应商配置：

```json
{
  "new_provider": {
    "api_url": "https://api.new-provider.com/v1",
    "supported_models": ["model-1", "model-2"],
    "default_model": "model-1",
    "timeout": 30,
    "max_retries": 3
  }
}
```

## 上传到GitHub的注意事项

1. **创建.gitignore文件**，排除敏感文件：

```
# .gitignore
.env
__pycache__/
*.pyc
.vscode/
```

2. **使用环境变量示例**，而非实际值：
   - 确保`env.example`中不包含真实API密钥
   - 确保`config/provider_config.json`中不包含敏感信息

3. **上传命令**：

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/zh2673-git/agent_chat_fuction_call.git
git push -u origin main
```

## 调试与故障排除

- **启用调试模式**：`python main.py -d`显示详细日志
- **检查工具调用**：查看控制台输出的工具调用信息
- **API密钥问题**：确保`.env`文件中的API密钥正确
- **工具参数缺失**：检查用户输入是否包含所需参数信息
- **模型兼容性**：确认所选模型支持Function Calling功能

## 未来计划

- [ ] 支持更多模型供应商（如Claude、DeepSeek等）
- [ ] 增强工具参数提取能力
- [ ] 添加更多实用工具
- [ ] 支持多轮工具调用
- [ ] 添加Web界面

## 如何贡献

1. Fork 本仓库
2. 创建分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 提交 Pull Request

## 许可证

MIT License (见 `LICENSE` 文件)
