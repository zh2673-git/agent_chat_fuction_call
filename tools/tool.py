from abc import ABC, abstractmethod
from typing import Dict, Any
import requests

class Tool(ABC):
    """
    基础工具类，所有工具需继承此类并实现 execute 方法。
    """
    def __init__(self, config: dict):
        """
        工具基类，所有具体工具需继承此类。
        :param config: 工具配置（如API Key、URL等）
        """
        self.config = config

    @abstractmethod
    def execute(self, prompt: str) -> str:
        """
        执行工具逻辑。
        :param prompt: 用户输入
        :return: 工具生成的响应
        """
        pass
