from openai import OpenAI
from .base import BaseProvider
from typing import Optional, Union, Generator, Dict, Any, List
import os
import json

class OpenaiProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        """
        从配置创建实例
        :param config: 配置字典，包含api_key, api_url, model_name等
        """
        # 获取API密钥，优先使用环境变量
        api_key = os.getenv("OPENAI_API_KEY", config.get("api_key"))
        
        # 获取API URL，默认使用官方API
        api_url = config.get("api_url", "https://api.openai.com/v1")
        
        # 获取模型名称，默认使用gpt-3.5-turbo
        model_name = config.get("model_name", "gpt-3.5-turbo")
        
        # 获取调试模式
        debug = config.get("debug", False)
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            debug=debug
        )
        
    def __init__(self, api_key: str, api_url: str, model_name: str, debug: bool = False):
        """
        初始化OpenAI提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如gpt-3.5-turbo, gpt-4等
        :param debug: 是否启用调试模式
        """
        if not api_key or api_key == "use_env_variable":
            raise ValueError("OPENAI_API_KEY 未配置或无效")
            
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name
        self.debug = debug
        
        if self.debug:
            print(f"初始化OpenAI提供商，使用模型: {model_name}")
    
    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)

    def _convert_tools(self, tools: list) -> list:
        """将通用工具格式转换为OpenAI格式"""
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

    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        使用OpenAI API生成响应
        :return: 
            - 工具调用: {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
        """
        try:
            # 转换工具格式为OpenAI格式
            openai_tools = self._convert_tools(tools or [])
            
            # 添加系统提示以增强参数提取能力
            system_message = """
            你是一个能够调用工具的助手。当用户询问可以用工具解决的问题时，请确保：
            1. 正确识别用户查询中的关键信息
            2. 选择合适的工具
            3. 提取查询中的所有相关参数值并填入工具参数中
            4. 不要遗漏用户提到的任何关键信息
            5. 如果用户提出了多个需要不同工具解决的问题，可以按顺序调用多个工具
            """
            
            # 构建请求
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # 判断是否需要工具调用
            if stream:
                # 流式输出
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    stream=True
                )
                
                # 检查是否为工具调用
                first_chunk = next(response)
                if hasattr(first_chunk.choices[0], 'delta') and hasattr(first_chunk.choices[0].delta, 'tool_calls'):
                    # 收集完整的工具调用信息
                    tool_calls = []
                    tool_call_chunks = [first_chunk]
                    
                    for chunk in response:
                        tool_call_chunks.append(chunk)
                        if chunk.choices[0].finish_reason == "tool_calls":
                            break
                    
                    # 提取所有工具调用
                    result_tool_calls = []
                    for i, tool_call in enumerate(tool_call_chunks[-1].choices[0].delta.tool_calls):
                        result_tool_calls.append({
                            "id": tool_call.id or f"call_{i}",
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
                    
                    return {"tool_calls": result_tool_calls}
                else:
                    # 返回流式生成器
                    def generate_stream():
                        yield first_chunk.choices[0].delta.content or ""
                        for chunk in response:
                            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    
                    return generate_stream()
            else:
                # 非流式输出
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                )
                
                # 检查是否为工具调用
                if response.choices[0].message.tool_calls:
                    # 提取所有工具调用
                    result_tool_calls = []
                    for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                        result_tool_calls.append({
                            "id": tool_call.id or f"call_{i}",
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
                    
                    return {"tool_calls": result_tool_calls}
                else:
                    # 返回文本内容
                    return response.choices[0].message.content
                
        except Exception as e:
            self._debug_print(f"OpenAI API错误: {str(e)}")
            return {"error": f"OpenAI API错误: {str(e)}"}
