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
    
    def _extract_tool_calls(self, response_data) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        从响应中提取工具调用信息，统一格式
        支持OpenAI SDK对象和JSON字典格式
        :param response_data: API响应数据(OpenAI SDK对象或字典)
        :return: 统一格式的工具调用信息，如果没有则返回None
        """
        try:
            # 处理OpenAI SDK对象
            if hasattr(response_data, 'choices') and response_data.choices:
                message = response_data.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    result = []
                    for i, tool_call in enumerate(message.tool_calls):
                        result.append({
                            "id": getattr(tool_call, 'id', f'call_{i}'),
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
                    return {"tool_calls": result}
            
            # 处理字典格式
            elif isinstance(response_data, dict):
                if ("choices" in response_data and 
                    response_data["choices"] and 
                    "message" in response_data["choices"][0] and
                    "tool_calls" in response_data["choices"][0]["message"]):
                    
                    tool_calls = response_data["choices"][0]["message"]["tool_calls"]
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
        except Exception as e:
            # 添加调试信息打印
            if hasattr(self, '_debug_print'):
                self._debug_print(f"提取工具调用时出错: {str(e)}")
            return None
