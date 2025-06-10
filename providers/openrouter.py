from .base import BaseProvider
import os
import re
from typing import Union, Generator, Dict, Any, Optional
import json
from openai import OpenAI

class OpenrouterProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        """
        从配置创建实例
        :param config: 配置字典，包含api_key, api_url, model_name等
        """
        # 获取API密钥，优先使用环境变量
        api_key = os.getenv("OPENROUTER_API_KEY", config.get("api_key"))
        
        # 获取API URL
        api_url = config.get("api_url", "https://openrouter.ai/api/v1").rstrip("/")
        
        # 获取模型名称，默认使用deepseek/deepseek-r1-0528:free
        model_name = config.get("model_name", "deepseek/deepseek-r1-0528:free")
        
        # 获取超时和重试设置
        timeout = config.get("timeout", 60)
        max_retries = config.get("max_retries", 3)
        
        # 获取调试模式
        debug = config.get("debug", False)
        
        # 获取额外的请求头
        extra_headers = config.get("extra_headers", {})
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            debug=debug,
            extra_headers=extra_headers
        )
    
    def __init__(self, api_key: str, api_url: str, model_name: str, timeout: int = 60, 
                 max_retries: int = 3, debug: bool = False, extra_headers: dict = None):
        """
        初始化OpenRouter提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称
        :param timeout: 请求超时时间(秒)
        :param max_retries: 最大重试次数
        :param debug: 是否启用调试模式
        :param extra_headers: 额外的请求头
        """
        # 调用父类初始化方法
        super().__init__(api_key, api_url, model_name, debug, timeout, max_retries, extra_headers)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def generate_response(self, prompt: str, tools: list = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成响应，返回格式与BaseProvider接口一致
        :return: 
            - 工具调用: {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        try:
            # 添加系统提示
            system_message = "你是一个有用的AI助手。"
            if tools:
                system_message = self.get_default_system_message()
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # 准备额外请求头
            extra_headers = {
                "HTTP-Referer": self.extra_headers.get("HTTP-Referer", "https://agent-chat.app"),
                "X-Title": self.extra_headers.get("X-Title", "Agent Chat App")
            }
            
            # 如果存在额外请求头，更新它们
            if self.extra_headers:
                extra_headers.update(self.extra_headers)
            
            # 准备基本请求参数
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "extra_headers": extra_headers
            }
            
            # 处理工具调用
            if tools and not stream:  # 流式输出暂时不使用工具
                try:
                    # 转换工具格式
                    openrouter_tools = self._convert_tools(tools)
                    if openrouter_tools:
                        # 直接使用工具调用，不再先尝试基本调用
                        completion_params["tools"] = openrouter_tools
                        tool_response = self.client.chat.completions.create(**completion_params)
                        
                        # 提取工具调用
                        tool_calls_result = self._extract_tool_calls(tool_response)
                        if tool_calls_result:
                            return tool_calls_result
                        
                        # 如果没有工具调用，返回内容
                        return tool_response.choices[0].message.content
                    
                except Exception as e:
                    self._debug_print(f"工具调用失败，回退到基本调用: {str(e)}")
                    # 移除tools参数，回退到基本调用
                    if "tools" in completion_params:
                        del completion_params["tools"]
            
            # 处理流式响应
            if stream:
                completion_params["stream"] = True
                
                def stream_response():
                    try:
                        response_stream = self.client.chat.completions.create(**completion_params)
                        for chunk in response_stream:
                            if hasattr(chunk, 'choices') and chunk.choices:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    yield delta.content
                    except Exception as e:
                        yield {"error": f"OpenRouter API错误: {str(e)}"}
                
                return stream_response()
            
            # 基本非流式调用
            response = self.client.chat.completions.create(**completion_params)
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                
                # 检查是否包含多个回复
                if content and any(pattern in content for pattern in [
                    "\n\n问题1:", "\n\n问题 1:", "\n\n回答1:", "\n\n回答 1:"
                ]):
                    # 尝试将内容分割成多个回复
                    self._debug_print("检测到多个回复，尝试分割")
                    return {
                        "multi_responses": self._split_multiple_responses(content)
                    }
                
                return content
            
            return "无法获取有效回复"
                
        except Exception as e:
            self._debug_print(f"OpenRouter API错误: {str(e)}")
            return {"error": f"OpenRouter API错误: {str(e)}"} 