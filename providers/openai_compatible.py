from .base import BaseProvider
import os
import requests
from typing import Union, Generator, Dict, Any, Optional
import json
import re

class OpenAICompatibleProvider(BaseProvider):
    """
    OpenAI兼容API提供商，适用于遵循OpenAI API格式的服务
    """
    @classmethod
    def from_config(cls, config: dict):
        """
        从配置创建实例
        :param config: 配置字典，包含api_key, api_url, model_name等
        """
        # 获取API密钥，优先使用环境变量
        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"), config.get("api_key"))
        
        # 获取API URL
        api_url = config.get("api_url", "https://api.openai.com/v1").rstrip("/") + "/"
        
        # 获取模型名称
        model_name = config.get("model_name", "gpt-3.5-turbo")
        
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
        初始化OpenAI兼容API提供商
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
        
        # 构建请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 添加额外的请求头
        if extra_headers:
            self.headers.update(extra_headers)

    def generate_response(self, prompt: str, tools: list, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成响应，返回格式与BaseProvider接口一致
        :return: 
            - 工具调用: {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        try:
            url = f"{self.api_url}chat/completions"
            
            # 使用基类中的系统提示
            system_message = self.get_default_system_message()
            
            # 构建请求数据
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "stream": stream,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            # 添加工具信息
            openai_tools = self._convert_tools(tools)
            if openai_tools:
                data["tools"] = openai_tools
            
            # 对于流式输出，先检查是否有工具调用
            if stream:
                # 先使用非流式请求检查是否有工具调用
                data_no_stream = data.copy()
                data_no_stream["stream"] = False
                check_response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=data_no_stream,
                    timeout=self.timeout
                )
                
                if check_response.status_code == 200:
                    check_result = check_response.json()
                    tool_calls_result = self._extract_tool_calls(check_result)
                    if tool_calls_result:
                        return tool_calls_result
                
                # 如果没有工具调用，使用流式输出
                def stream_response():
                    with requests.post(
                        url, 
                        headers=self.headers, 
                        json=data, 
                        stream=True,
                        timeout=self.timeout
                    ) as response:
                        if response.status_code != 200:
                            yield {"error": f"API错误: {response.status_code}"}
                            return
                            
                        # 读取并解析事件流中的数据
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        json_data = json.loads(data_str)
                                        if "choices" in json_data and json_data["choices"]:
                                            delta = json_data["choices"][0].get("delta", {})
                                            
                                            # 检查是否有工具调用
                                            if "tool_calls" in delta:
                                                # 流式工具调用，提取第一个工具调用
                                                tool_call = delta["tool_calls"][0]
                                                return {
                                                    "tool_calls": [{
                                                        "id": tool_call.get("id", "call_0"),
                                                        "name": tool_call["function"]["name"],
                                                        "arguments": tool_call["function"]["arguments"]
                                                    }]
                                                }
                                            
                                            if "content" in delta and delta["content"]:
                                                yield delta["content"]
                                    except json.JSONDecodeError:
                                        pass
                
                return stream_response()
            else:
                # 非流式处理
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=data,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    # 使用基类中的工具调用提取方法
                    tool_calls_result = self._extract_tool_calls(result)
                    if tool_calls_result:
                        return tool_calls_result
                    
                    # 如果没有工具调用，返回内容
                    if "choices" in result and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                        content = result["choices"][0]["message"]["content"]
                        
                        # 检查是否包含多个回复
                        if "\n\n问题1:" in content or "\n\n问题 1:" in content:
                            # 使用基类中的方法尝试将内容分割成多个回复
                            self._debug_print("检测到多个回复，尝试分割")
                            return {
                                "multi_responses": self._split_multiple_responses(content)
                            }
                        
                        return content
                    
                    return "无法获取有效回复"
                else:
                    return {"error": f"API错误: {response.status_code} - {response.text}"}
            
        except Exception as e:
            self._debug_print(f"OpenAI兼容API错误: {str(e)}")
            return {"error": f"OpenAI兼容API错误: {str(e)}"}