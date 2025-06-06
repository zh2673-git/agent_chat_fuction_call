from typing import Dict, Optional
import json
from LLM import LargeLanguageModel
from tools.base import Tool  # 假设有一个基础工具类

class ChatAgent:
    def __init__(self, model_provider: str, tool_config_file: str = "tools/config.json"):
        """
        聊天机器人Agent，集成LLM和工具调用。
        :param model_provider: 模型提供商（如 openai）
        :param tool_config_file: 工具配置文件路径
        """
        self.llm = LargeLanguageModel(model_provider)
        self.tools = self._load_tools(tool_config_file)

    def _load_tools(self, config_file: str) -> Dict[str, Tool]:
        """
        动态加载所有工具。
        :param config_file: 工具配置文件路径
        :return: 工具名称到工具对象的映射
        """
        with open(config_file, "r") as f:
            configs = json.load(f)
        
        tools = {}
        for name, config in configs.items():
            # 动态导入工具类
            module_name = f"tools.{name}"
            class_name = f"{name.capitalize()}Tool"
            module = __import__(module_name, fromlist=[class_name])
            tool_class = getattr(module, class_name)
            tools[name] = tool_class(config)
        return tools

    def run(self, prompt: str) -> str:
        """
        处理用户输入，自动调用工具或LLM。
        :param prompt: 用户输入
        :return: 工具或LLM生成的响应
        """
        # 调用 LLM 的 Function Calling 功能
        llm_response = self.llm.generate_response(
            prompt,
            tools=[{"name": name} for name in self.tools.keys()]  # 告诉 LLM 可用的工具列表
        )

        # 检查是否需要调用工具
        if llm_response.get("tool_call"):
            tool_name = llm_response["tool_call"]["name"]
            tool_args = llm_response["tool_call"]["arguments"]
            return self.tools[tool_name].execute(tool_args)
        else:
            return llm_response["content"]

# 示例用法
if __name__ == "__main__":
    # 初始化Agent
    agent = ChatAgent("openai")

    # 示例1：调用天气工具
    print(agent.run("北京天气怎么样？"))  # LLM 可能返回 {"tool_call": {"name": "weather", "arguments": {"city": "北京"}}}

    # 示例2：调用LLM
    print(agent.run("你好！"))           # LLM 直接生成自然语言响应

    # 示例3：调用其他工具（假设已注册）
    print(agent.run("翻译成英文"))       # 输出翻译工具的响应
