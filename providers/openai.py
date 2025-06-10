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
        
        # 获取超时和重试设置
        timeout = config.get("timeout", 60)
        max_retries = config.get("max_retries", 3)
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            debug=debug,
            timeout=timeout,
            max_retries=max_retries
        )
        
    def __init__(self, api_key: str, api_url: str, model_name: str, debug: bool = False,
                 timeout: int = 60, max_retries: int = 3, extra_headers: dict = None):
        """
        初始化OpenAI提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如gpt-3.5-turbo, gpt-4等
        :param debug: 是否启用调试模式
        :param timeout: 请求超时时间(秒)
        :param max_retries: 最大重试次数
        :param extra_headers: 额外的请求头
        """
        # 调用父类初始化方法
        super().__init__(api_key, api_url, model_name, debug, timeout, max_retries, extra_headers)
            
        # 初始化OpenAI客户端
        self.client = OpenAI(base_url=api_url, api_key=api_key)
    
    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)

    def _convert_tools(self, tools: list) -> list:
        """
        将通用工具格式转换为OpenAI格式
        这是对基类方法的覆盖，因为OpenAI需要特定的工具格式
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

    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成模型响应，实现BaseProvider抽象方法
        
        :param prompt: 用户输入的提示文本
        :param tools: 可用工具列表
        :param stream: 是否使用流式输出
        :return: 
            - 工具调用: {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        try:
            # 确保API URL格式正确
            self._debug_print(f"使用API URL: {self.api_url}")
            self._debug_print(f"使用模型: {self.model_name}")
            
            # 转换工具格式为OpenAI格式
            openai_tools = self._convert_tools(tools or [])
            
            # 使用基类中的系统提示
            system_message = self.get_default_system_message()
            
            # 构建请求
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            self._debug_print(f"请求消息: {json.dumps(messages, ensure_ascii=False)}")
            if openai_tools:
                self._debug_print(f"工具数量: {len(openai_tools)}")
            
            # 判断是否需要工具调用
            if stream:
                # 流式输出
                try:
                    self._debug_print("发送流式请求")
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=openai_tools if openai_tools else None,
                        stream=True
                    )
                    
                    # 检查是否为工具调用
                    try:
                        first_chunk = next(response)
                        self._debug_print(f"收到首个流式块: {first_chunk}")
                        
                        if hasattr(first_chunk.choices[0], 'delta') and hasattr(first_chunk.choices[0].delta, 'tool_calls'):
                            # 收集完整的工具调用信息
                            self._debug_print("检测到工具调用")
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
                            
                            self._debug_print(f"提取的工具调用: {result_tool_calls}")
                            return {"tool_calls": result_tool_calls}
                        else:
                            # 返回流式生成器
                            def generate_stream():
                                content = first_chunk.choices[0].delta.content
                                if content:
                                    yield content
                                
                                for chunk in response:
                                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                        yield chunk.choices[0].delta.content
                            
                            return generate_stream()
                    except StopIteration:
                        self._debug_print("流式请求没有返回任何数据")
                        return {"error": "流式请求没有返回任何数据"}
                except Exception as e:
                    self._debug_print(f"流式请求错误: {str(e)}")
                    return {"error": f"流式请求错误: {str(e)}"}
            else:
                # 非流式输出
                try:
                    self._debug_print("发送非流式请求")
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=openai_tools if openai_tools else None,
                    )
                    
                    self._debug_print(f"收到响应: {response}")
                    
                    # 使用基类中的工具调用提取方法
                    tool_calls = self._extract_tool_calls(response)
                    if tool_calls:
                        self._debug_print(f"提取的工具调用: {tool_calls}")
                        return tool_calls
                    else:
                        # 返回文本内容
                        content = response.choices[0].message.content
                        if not content:
                            self._debug_print("警告: 响应中没有找到文本内容")
                            return "无法获取有效回复"
                        return content
                except Exception as e:
                    self._debug_print(f"非流式请求错误: {str(e)}")
                    return {"error": f"非流式请求错误: {str(e)}"}
                
        except Exception as e:
            self._debug_print(f"OpenAI API错误: {str(e)}")
            import traceback
            self._debug_print(f"错误堆栈: {traceback.format_exc()}")
            return {"error": f"OpenAI API错误: {str(e)}"}
