import json
import os
import importlib
from typing import Dict, Any, Optional, Union, Generator, Iterator
import requests
from pathlib import Path
from providers import BaseProvider

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
    def __init__(self, model_provider: str, model_name: Optional[str] = None, debug: bool = False):
        """
        初始化大语言模型调用类
        :param model_provider: 模型供应商名称（如 openai/modelscope）
        :param model_name: 具体模型名称（如 gpt-4/qwen-max），如果为None则使用配置文件中的默认值
        :param debug: 是否启用调试模式
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.debug = debug  # 确保此行在provider初始化之前
        self.provider = self._load_provider(model_provider, model_name)

    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)

    def _load_provider(self, provider_name: str, model_name: Optional[str] = None) -> 'BaseProvider':
        """
        动态加载模型供应商实现
        :param provider_name: 供应商名称（对应 providers/ 下的模块名）
        :param model_name: 具体模型名称，如果为None则使用配置文件中的默认值
        :return: 供应商实例
        """
        try:
            # 导入providers包
            import providers
            
            # 获取对应的提供商类
            provider_class_name = f"{provider_name.capitalize()}Provider"
            if not hasattr(providers, provider_class_name):
                raise ValueError(f"未找到供应商类: {provider_class_name}")
                
            provider_class = getattr(providers, provider_class_name)
            
            # 从配置文件加载配置
            provider_config = self._load_provider_config(provider_name)
            
            # 添加模型名称到配置中
            if model_name:
                # 检查模型是否在支持列表中
                if "supported_models" in provider_config and model_name not in provider_config["supported_models"]:
                    self._debug_print(f"警告: 模型 {model_name} 不在 {provider_name} 的支持列表中")
                provider_config["model_name"] = model_name
            elif "default_model" in provider_config:
                provider_config["model_name"] = provider_config["default_model"]
                
            # 创建提供商实例
            return provider_class.from_config(provider_config)
            
        except ImportError as e:
            raise ValueError(f"导入提供商模块失败: {e}")
        except Exception as e:
            raise ValueError(f"加载提供商 {provider_name} 失败: {e}")

    def _load_provider_config(self, provider_name: str) -> dict:
        """
        加载提供商配置
        :param provider_name: 提供商名称
        :return: 配置字典
        """
        try:
            with open("config/provider_config.json") as f:
                provider_configs = json.load(f)
                config = provider_configs.get(provider_name, {})
        except (FileNotFoundError, json.JSONDecodeError):
            self._debug_print("警告: 无法加载提供商配置文件，使用空配置")
            config = {}
        return config

    def get_supported_models(self, provider_name: Optional[str] = None) -> Dict[str, list]:
        """
        获取支持的模型列表
        :param provider_name: 提供商名称，如果为None则返回所有提供商的模型
        :return: 提供商名称到模型列表的映射
        """
        try:
            with open("config/provider_config.json") as f:
                provider_configs = json.load(f)
                
            result = {}
            if provider_name:
                if provider_name not in provider_configs:
                    return {provider_name: []}
                config = provider_configs[provider_name]
                result[provider_name] = config.get("supported_models", [])
            else:
                for name, config in provider_configs.items():
                    result[name] = config.get("supported_models", [])
            
            return result
        except (FileNotFoundError, json.JSONDecodeError):
            self._debug_print("警告: 无法加载提供商配置文件")
            return {}

    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[Dict[str, Any], Generator[str, None, None], str]:
        """
        生成响应，直接返回提供商的原始响应。
        :param prompt: 用户输入
        :param tools: 可用工具列表
        :param stream: 是否启用流式输出
        :return: 提供商返回的原始响应
            - 工具调用: {"tool_call": {"name": str, "arguments": str}}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
        """
        try:
            # 将参数传递给provider，返回原始响应
            response = self.provider.generate_response(prompt, tools or [], stream=stream)
            self._debug_print(f"原始响应类型: {type(response)}")
            return response
        except Exception as e:
            self._debug_print(f"生成响应错误: {e}")
            return {"error": str(e)}

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
