from abc import ABC, abstractmethod
from typing import Dict, Any, List, get_type_hints
import requests
import inspect

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
    def execute(self, *args, **kwargs) -> str:
        """
        执行工具逻辑。
        :param args: 位置参数
        :param kwargs: 关键字参数
        :return: 工具生成的响应
        """
        pass

    @property
    def description(self) -> str:
        """获取工具描述"""
        return self.execute.__doc__.strip() if self.execute.__doc__ else "No description provided."

    @property
    def parameters(self) -> Dict:
        """自动提取工具参数信息"""
        # 获取execute方法的参数信息
        sig = inspect.signature(self.execute)
        param_types = get_type_hints(self.execute)
        
        properties = {}
        required = []
        
        # 跳过self参数
        for name, param in list(sig.parameters.items())[1:]:
            param_info = {
                "type": self._get_json_type(param_types.get(name, Any)),
                "description": f"Parameter {name}"
            }
            
            # 如果参数没有默认值，则为必需参数
            if param.default is param.empty:
                required.append(name)
                
            properties[name] = param_info
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _get_json_type(self, type_hint):
        """将Python类型转换为JSON Schema类型"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return type_map.get(type_hint, "string")
