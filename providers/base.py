from abc import ABC, abstractmethod
from typing import Optional, Union, Generator, Dict, Any

class BaseProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成模型响应，所有提供商必须实现此方法
        
        :param prompt: 用户输入的提示文本
        :param tools: 可用工具列表
        :param stream: 是否使用流式输出
        :return: 
            - 工具调用: {"tool_call": {"name": str, "arguments": str}}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        pass

    @classmethod
    def from_config(cls, config: dict):
        """根据配置创建实例"""
        return cls(**config)
