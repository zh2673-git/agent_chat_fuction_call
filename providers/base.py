from abc import ABC, abstractmethod
from typing import Optional

class BaseProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, tools: Optional[list] = None) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        """根据配置创建实例"""
        return cls(**config)
