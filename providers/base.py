from abc import ABC, abstractmethod
from typing import Optional, Union, Generator, Dict, Any, List
import os
import json
import re

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
        """
        根据配置创建实例
        默认实现，子类可以根据需要重写
        """
        # 获取API密钥，优先使用环境变量
        env_key = config.get("env_key", "API_KEY")
        api_key = os.getenv(env_key, config.get("api_key"))
        
        # 获取API URL
        api_url = config.get("api_url", "")
        
        # 获取模型名称
        model_name = config.get("model_name", config.get("default_model", ""))
        
        # 获取调试模式
        debug = config.get("debug", False)
        
        # 获取超时和重试设置
        timeout = config.get("timeout", 60)
        max_retries = config.get("max_retries", 3)
        
        # 获取额外的请求头
        extra_headers = config.get("extra_headers", {})
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            debug=debug,
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=extra_headers
        )
    
    def __init__(self, api_key: str, api_url: str, model_name: str, debug: bool = False, 
                 timeout: int = 60, max_retries: int = 3, extra_headers: dict = None):
        """
        初始化提供商基类
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称
        :param debug: 是否启用调试模式
        :param timeout: 请求超时时间(秒)
        :param max_retries: 最大重试次数
        :param extra_headers: 额外的请求头
        """
        if not api_key or api_key == "use_env_variable":
            raise ValueError("API密钥未配置或无效")
            
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        
        if self.debug:
            print(f"初始化{self.__class__.__name__}，使用模型: {model_name}")
    
    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)
    
    def _convert_tools(self, tools: list) -> list:
        """
        将通用工具格式转换为OpenAI格式
        子类可以根据需要重写此方法以适应不同的API格式
        """
        if not tools:
            return []
            
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            for tool in tools
        ]
    
    def _extract_tool_calls(self, response_data) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        从响应中提取工具调用信息，统一格式
        支持OpenAI SDK对象和JSON字典格式
        :param response_data: API响应数据(OpenAI SDK对象或字典)
        :return: 统一格式的工具调用信息，如果没有则返回None
        """
        try:
            # 首先检查response_data是否为None
            if response_data is None:
                self._debug_print("警告: 收到空的响应数据")
                return None
                
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
                # 确保所有必要的键都存在
                if ("choices" in response_data and 
                    response_data["choices"] and 
                    isinstance(response_data["choices"], list) and
                    len(response_data["choices"]) > 0 and
                    "message" in response_data["choices"][0] and
                    isinstance(response_data["choices"][0]["message"], dict) and
                    "tool_calls" in response_data["choices"][0]["message"]):
                    
                    tool_calls = response_data["choices"][0]["message"]["tool_calls"]
                    # 确保tool_calls是列表且不为空
                    if not isinstance(tool_calls, list) or not tool_calls:
                        return None
                        
                    result = []
                    
                    for i, tool_call in enumerate(tool_calls):
                        # 提取工具调用ID，如果没有则生成一个
                        tool_id = tool_call.get("id", f"call_{i}")
                        
                        # 提取工具名称和参数
                        if "function" in tool_call:
                            tool_name = tool_call["function"].get("name")
                            arguments = tool_call["function"].get("arguments", "{}")
                            
                            # 确保工具名称不为空
                            if not tool_name:
                                continue
                        else:
                            # 兼容其他可能的格式
                            tool_name = tool_call.get("name", f"unknown_tool_{i}")
                            arguments = tool_call.get("arguments", "{}")
                        
                        result.append({
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": arguments
                        })
                    
                    # 只有当结果不为空时才返回
                    if result:
                        return {"tool_calls": result}
            
            return None
        except Exception as e:
            # 添加调试信息打印
            if hasattr(self, '_debug_print'):
                self._debug_print(f"提取工具调用时出错: {str(e)}")
            return None

    def get_default_system_message(self) -> str:
        """获取默认的系统提示，用于工具调用增强"""
        return """
        你是一个能够调用工具的助手。当用户询问可以用工具解决的问题时，请确保：
        1. 正确识别用户查询中的关键信息
        2. 选择合适的工具
        3. 提取查询中的所有相关参数值并填入工具参数中
        4. 不要遗漏用户提到的任何关键信息
        5. 如果用户提出了多个需要不同工具解决的问题，可以按顺序调用多个工具
        """

    def _split_multiple_responses(self, content: str) -> list:
        """
        尝试将包含多个回复的内容分割成单独的回复列表
        :param content: 包含多个回复的文本内容
        :return: 回复列表
        """
        # 常见的多回复分隔模式
        patterns = [
            r"问题\s*\d+[:：]", 
            r"回答\s*\d+[:：]",
            r"问题\s*\(?\d+\)?[:：]",
            r"回答\s*\(?\d+\)?[:：]",
            r"\d+\.\s*问[:：]",
            r"\d+\.\s*答[:：]"
        ]
        
        # 尝试使用不同的模式分割
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if len(matches) > 1:
                # 找到了多个匹配项，使用该模式分割
                parts = re.split(pattern, content)
                # 去掉第一个空元素（如果存在）
                if parts and not parts[0].strip():
                    parts = parts[1:]
                
                # 将分隔符添加回每个部分
                responses = []
                for i, part in enumerate(parts):
                    if i < len(matches):
                        responses.append(f"{matches[i]}{part.strip()}")
                    else:
                        responses.append(part.strip())
                
                return responses
        
        # 如果没有找到明确的分隔符，尝试按段落分割
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        
        # 如果无法分割，返回原始内容作为单个元素
        return [content]
