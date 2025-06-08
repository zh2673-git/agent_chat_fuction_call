import json
import os
import importlib
from typing import Dict, Any, Optional, Union, Generator, Iterator, List
import requests
from pathlib import Path
from providers import BaseProvider, get_provider

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
    def __init__(self, provider_name: str, model_name: Optional[str] = None, debug: bool = False):
        """
        初始化大语言模型
        :param provider_name: 提供商名称
        :param model_name: 模型名称，如果为None则使用提供商的默认模型
        :param debug: 是否启用调试模式
        """
        self.provider_name = provider_name
        self.debug = debug
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化提供商
        provider_config = self.config.get(provider_name, {})
        
        # 保存模型名称，如果未指定则使用默认模型
        if model_name:
            provider_config["model_name"] = model_name
            self.model_name = model_name
        else:
            # 从配置文件中获取默认模型名称
            self.model_name = provider_config.get("default_model", "未知模型")
            # 确保provider_config中也包含model_name
            provider_config["model_name"] = self.model_name
        
        if debug:
            print(f"初始化LLM，提供商: {provider_name}, 模型: {self.model_name}")
            print(f"提供商配置: {provider_config}")
        
        provider_config["debug"] = debug
        
        # 使用get_provider函数获取提供商实例
        self.provider = get_provider(provider_name, provider_config)

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = os.path.join("config", "provider_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def generate_response(self, prompt: str, tools: Optional[List[Dict]] = None, stream: bool = False) -> Union[str, Dict, Generator]:
        """
        生成响应
        :param prompt: 提示文本
        :param tools: 可用工具列表
        :param stream: 是否使用流式输出
        :return: 响应文本、工具调用或流式生成器
        """
        return self.provider.generate_response(prompt, tools or [], stream)
    
    def get_supported_models(self, provider_name: Optional[str] = None) -> Dict[str, list]:
        """
        获取支持的模型列表
        :param provider_name: 提供商名称，如果为None则返回所有提供商的模型
        :return: 提供商名称到模型列表的映射
        """
        # 这里可以实现一个获取支持模型的逻辑
        # 暂时返回一个简单的映射
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "openai_compatible": ["自定义模型"],
            "siliconflow": ["deepseek-ai/DeepSeek-V3", "01-ai/Yi-1.5-34B"],
            "openrouter": ["deepseek/deepseek-r1-0528", "anthropic/claude-3-opus", "google/gemini-pro"]
        }
        
        if provider_name:
            return {provider_name: models.get(provider_name, [])}
        return models

# 示例用法
if __name__ == "__main__":
    try:
        # 指定模型提供商和具体模型名称
        model = LargeLanguageModel("openai", model_name="gpt-4", debug=True)

        # 调用生成方法
        response = model.generate_response("你好，请介绍一下你自己。")
        print(response)
    except ValueError as e:
        print(f"初始化失败: {e}")
