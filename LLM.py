import json
import os
import importlib
from typing import Dict, Any, Optional, Union, Generator, Iterator
import requests
from pathlib import Path

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
    def __init__(self, model_provider: str, tool_config_file: str = "tools/config.json", debug: bool = False):
        """
        初始化大语言模型调用类
        :param model_provider: 模型供应商名称（如 openai/modelscope）
        :param tool_config_file: 工具配置文件路径
        :param debug: 是否启用调试模式
        """
        self.model_provider = model_provider
        self.debug = debug  # 确保此行在provider初始化之前
        self.provider = self._load_provider(model_provider)
        self.tools = self._load_tools(tool_config_file) if tool_config_file else {}

    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)

    def _load_provider(self, provider_name: str) -> 'BaseProvider':
        """
        动态加载模型供应商实现
        :param provider_name: 供应商名称（对应 providers/ 下的模块名）
        """
        try:
            # 1. 动态导入供应商模块
            module = importlib.import_module(f"providers.{provider_name}")
            
            # 2. 获取供应商类（命名约定：{ProviderName}Provider）
            provider_class = getattr(module, f"{provider_name.capitalize()}Provider")
            
            # 3. 从配置初始化
            with open("config/model.json") as f:
                config = json.load(f)[provider_name]
                
            return provider_class.from_config(config)
            
        except ImportError as e:
            raise ValueError(f"未找到供应商实现: {provider_name}") from e
        except AttributeError:
            raise ValueError(f"供应商类命名不规范，应为 {provider_name.capitalize()}Provider")
        except KeyError:
            raise ValueError(f"config.json 中缺少 {provider_name} 的配置")

    def _load_tools(self, config_file: str) -> Dict[str, 'Tool']:
        """增强版工具加载，自动提取工具参数信息"""
        with open(config_file, "r") as f:
            configs = json.load(f)
        
        tools = {}
        for name, config in configs.items():
            try:
                module = __import__(f"tools.{name}", fromlist=[f"{name.capitalize()}Tool"])
                tool_class = getattr(module, f"{name.capitalize()}Tool")
                tools[name] = tool_class(config)
                if self.debug:
                    print(f"成功加载工具: {name}")
            except Exception as e:
                if self.debug:
                    print(f"加载工具 {name} 失败: {e}")
        return tools

    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> dict:
        """
        生成响应，支持流式输出和工具调用。
        :param stream: 是否启用流式输出
        :return: {
            "content": str,                 # 非流式模式
            "stream": Generator[str, None], # 流式模式
            "tool_call": dict               # 工具调用
        }
        """
        try:
            # 将stream参数传递给provider，由provider负责处理流式输出
            response = self.provider.generate_response(prompt, tools or [], stream=stream)
            self._debug_print(f"原始响应类型: {type(response)}")
            
            # 1. 检查是否为工具调用
            if isinstance(response, dict) and "tool_call" in response:
                self._debug_print(f"检测到工具调用: {response['tool_call']}")
                return {"tool_call": response["tool_call"]}
            
            # 2. 检查是否为流式响应
            if stream and isinstance(response, Generator):
                self._debug_print("收到流式响应")
                return {"stream": response}
            
            # 3. 处理普通字符串响应
            if isinstance(response, str):
                self._debug_print(f"收到字符串响应: {response[:30]}...")
                return {"content": response}
            
            # 4. 处理其他类型响应(兜底)
            self._debug_print(f"未识别的响应类型: {type(response)}")
            return {"content": str(response)}
            
        except Exception as e:
            self._debug_print(f"生成响应错误: {e}")
            return {"error": str(e)}

# 抽象基类（也可放在 providers/base.py）
class BaseProvider:
    @classmethod
    def from_config(cls, config: dict) -> 'BaseProvider':
        """从配置创建实例（子类必须实现）"""
        raise NotImplementedError
        
    def generate_response(self, prompt: str, tools: list, stream: bool = False) -> Union[str, dict, Generator[str, None, None]]:
        """生成响应（子类必须实现）"""
        raise NotImplementedError

# 示例用法
if __name__ == "__main__":
    try:
        # 指定模型提供商（如 openai 或 deepseek）
        model = LargeLanguageModel("openai", debug=True)

        # 调用生成方法
        response = model.generate_response("你好，请介绍一下你自己。")
        print(response)
    except ValueError as e:
        print(f"初始化失败: {e}")
