from abc import ABC, abstractmethod
from typing import Optional, Union, Generator, Dict, Any, List

class BaseProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成模型响应，所有提供商必须实现此方法
        
        :param prompt: 用户输入的提示文本
        :param tools: 可用工具列表
        :param stream: 是否使用流式输出
        :return: 
            - 工具调用: {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        pass

    @classmethod
    def from_config(cls, config: dict):
        """根据配置创建实例"""
        return cls(**config)
    
    def _extract_tool_calls(self, response_json: dict) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        从响应JSON中提取工具调用信息，统一格式
        :param response_json: API响应的JSON数据
        :return: 统一格式的工具调用信息，如果没有则返回None
        """
        if ("choices" in response_json and 
            response_json["choices"] and 
            response_json["choices"][0].get("message", {}).get("tool_calls")):
            
            tool_calls = response_json["choices"][0]["message"]["tool_calls"]
            result = []
            
            for i, tool_call in enumerate(tool_calls):
                # 提取工具调用ID，如果没有则生成一个
                tool_id = tool_call.get("id", f"call_{i}")
                
                # 提取工具名称和参数
                if "function" in tool_call:
                    tool_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                else:
                    # 兼容其他可能的格式
                    tool_name = tool_call.get("name", f"unknown_tool_{i}")
                    arguments = tool_call.get("arguments", "{}")
                
                result.append({
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": arguments
                })
            
            return {"tool_calls": result}
        
        return None
