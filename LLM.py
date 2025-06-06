import json
import os
from typing import Dict, Any, Optional
import requests

class ModelConfig:
    def __init__(self, config_file: str = "model.json"):
        """
        加载模型配置文件。
        :param config_file: 配置文件路径，默认为 model.json
        """
        self.config_file = config_file
        self.models = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载并解析配置文件"""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"配置文件 {self.config_file} 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {self.config_file} 格式错误")

    def get_model_config(self, model_provider: str) -> Dict[str, str]:
        """
        获取指定模型的配置，支持通过环境变量动态选择模型。
        :param model_provider: 模型提供商名称（如 openai、deepseek）
        """
        config = self.models.get(model_provider)
        if not config:
            raise ValueError(f"未找到模型配置: {model_provider}")

        # 从环境变量读取模型名称，默认为配置中的第一个模型
        model_name = os.getenv("MODEL_NAME", list(config["models"].keys())[0])
        
        # 检查模型是否存在
        if model_name not in config["models"]:
            raise ValueError(f"未找到模型: {model_name}")

        return {
            "api_key": config["api_key"],
            "model_name": model_name,
            "api_url": config["models"][model_name]
        }

class LargeLanguageModel:
    def __init__(self, model_provider: str, config_file: str = "model.json"):
        """
        初始化大模型调用类。
        :param model_provider: 模型提供商名称（如 openai、deepseek）
        :param config_file: 配置文件路径，默认为 model.json
        """
        self.model_provider = model_provider
        self.config = ModelConfig(config_file).get_model_config(model_provider)

        # 初始化请求参数
        self.api_key = self.config["api_key"]
        self.model_name = self.config["model_name"]
        self.api_url = self.config["api_url"]
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_response(self, prompt: str, tools: Optional[list] = None) -> dict:
        """
        支持 Function Calling 的 LLM 响应生成。
        :param prompt: 用户输入
        :param tools: 可用工具列表（如 [{"name": "weather"}, {"name": "translation"}]）
        :return: 包含工具调用或自然语言响应的字典
        """
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools  # 告诉 LLM 可用的工具
        }

        response = requests.post(self.api_url, headers=self.headers, json=data)
        response_data = response.json()

        # 解析 LLM 响应，判断是否需要调用工具
        if "tool_calls" in response_data:
            return {
                "tool_call": {
                    "name": response_data["tool_calls"][0]["function"]["name"],
                    "arguments": json.loads(response_data["tool_calls"][0]["function"]["arguments"])
                }
            }
        else:
            return {"content": response_data["choices"][0]["message"]["content"]}

# 示例用法
if __name__ == "__main__":
    try:
        # 指定模型提供商（如 openai 或 deepseek）
        model = LargeLanguageModel("openai")

        # 检查当前使用的模型
        print(f"当前模型: {model.model_name}")

        # 调用生成方法
        response = model.generate_response("你好，请介绍一下你自己。")
        print(response)
    except ValueError as e:
        print(f"初始化失败: {e}")
