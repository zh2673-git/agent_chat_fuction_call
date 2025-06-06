# Agent Chat with Function Calling

一个基于大语言模型（LLM）的智能对话Agent，支持动态工具调用（Function Calling）和流式输出功能。

## 主要特性

- **智能工具调用**：根据用户意图自动选择并调用合适的工具
- **流式输出**：支持流式生成响应，提供更自然的对话体验
- **模型兼容性**：支持多种模型供应商（OpenAI、ModelScope等）
- **可扩展工具集**：易于添加自定义工具，不需修改核心代码
- **详细工具调用展示**：清晰展示工具调用过程和结果
- **自动参数提取**：从用户输入中智能提取工具所需参数
- **健壮的错误处理**：优雅处理各类异常情况

## 项目架构

```
agent_chat_fuction_call/
├── .env                    # 环境变量（API密钥等，从env.example复制）
├── env.example             # 环境变量示例模板
├── agent.py                # Agent核心逻辑（工具注册与调用）
├── LLM.py                  # 大语言模型封装（支持Function Calling）
├── main.py                 # 项目入口文件
├── config/                 # 配置文件目录
│   ├── model.json          # 模型供应商配置（API URL、模型列表）
│   └── provider_config.json # 供应商特定配置
├── providers/              # 模型供应商实现
│   ├── openai.py           # OpenAI API实现
│   └── modelscope.py       # ModelScope API实现
└── tools/                  # 工具目录
    ├── __init__.py         # 空文件（Python包标识）
    ├── tool.py             # 工具基类（自动参数提取）
    ├── weather.py          # 示例：天气查询工具
    ├── translation.py      # 示例：翻译工具
    └── config.json         # 工具配置（API密钥等）
```

## 核心原理

1. **动态工具调用**：
   - Agent通过LLM的Function Calling能力，自动解析用户输入，决定是否调用工具
   - 工具参数自动从用户输入中提取，减少多轮交互
   - 支持多种工具类型，如天气查询、文本翻译等

2. **智能响应生成**：
   - 基于工具调用结果生成自然、友好的回复
   - 流式输出支持，提供更自然的对话体验

3. **配置驱动架构**：
   - **模型配置**：`config/model.json`定义供应商及其模型
   - **工具配置**：`tools/config.json`注册工具参数
   - **环境变量**：`.env`管理敏感信息（API密钥）

4. **扩展性设计**：
   - 统一的供应商接口：便于添加新的模型供应商
   - 工具自动注册：只需创建新工具类并在配置中注册
   - 参数自动提取：通过反射机制自动识别工具参数

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境
复制`env.example`为`.env`并填写API密钥：
```env
# .env
MODELSCOPE_API_KEY=your_key_here
# 可选：指定使用的模型名称
# MODEL_NAME=Qwen/Qwen3-235B-A22B
```

### 3. 运行Agent
```bash
# 正常模式
python main.py

# 调试模式（显示详细日志）
python main.py -d

# 详细输出模式
python main.py -v
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
[工具参数] {"prompt": "翻译"你好"成英文"}
[工具结果] [TEST] Translation result: Hello
[模型回复] "你好"翻译成英文是"Hello"。
```

## 如何添加新工具

1. **创建工具类**：在`tools/`目录下创建新的工具类，继承`Tool`基类：

```python
from .tool import Tool

class MyNewTool(Tool):
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

## 调试与故障排除

- **启用调试模式**：`python main.py -d`显示详细日志
- **检查工具调用**：查看控制台输出的工具调用信息
- **API密钥问题**：确保`.env`文件中的API密钥正确
- **工具参数缺失**：检查用户输入是否包含所需参数信息

## 如何贡献

1. Fork 本仓库
2. 创建分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 提交 Pull Request

## 许可证

MIT License (见 `LICENSE` 文件)
