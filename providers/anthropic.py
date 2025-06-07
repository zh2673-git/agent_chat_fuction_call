from .base import BaseProvider
from typing import Optional, Union, Generator, Dict, Any
import os
import json
import requests

class AnthropicProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        """
        从配置创建实例
        :param config: 配置字典，包含api_key, api_url, model_name等
        """
        # 获取API密钥，优先使用环境变量
        api_key = os.getenv("ANTHROPIC_API_KEY", config.get("api_key"))
        
        # 获取API URL，默认使用官方API
        api_url = config.get("api_url", "https://api.anthropic.com")
        
        # 获取模型名称，默认使用claude-3-opus-20240229
        model_name = config.get("model_name", "claude-3-opus-20240229")
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name
        )
        
    def __init__(self, api_key: str, api_url: str, model_name: str):
        """
        初始化Anthropic提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如claude-3-opus-20240229
        """
        if not api_key or api_key == "use_env_variable":
            raise ValueError("ANTHROPIC_API_KEY 未配置或无效")
            
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        
        print(f"初始化Anthropic提供商，使用模型: {model_name}")

    def generate_response(self, prompt: str, tools: Optional[list] = None, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        使用Anthropic API生成响应，保持与OpenAI兼容的接口
        :return: 
            - 工具调用: {"tool_call": {"name": str, "arguments": str}}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
        """
        try:
            # 构建请求头
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # 构建请求体
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000
            }
            
            # 添加工具调用支持
            if tools and len(tools) > 0:
                # 转换工具格式为Anthropic格式
                anthropic_tools = []
                for tool in tools:
                    anthropic_tool = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {})
                    }
                    anthropic_tools.append(anthropic_tool)
                
                payload["tools"] = anthropic_tools
                payload["tool_choice"] = "auto"
            
            # 添加流式支持
            if stream:
                payload["stream"] = True
                
                # 发送请求
                response = requests.post(
                    f"{self.api_url}/v1/messages",
                    headers=headers,
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                
                def generate_stream():
                    buffer = ""
                    tool_call_data = None
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        if line.startswith(b"data: "):
                            data = json.loads(line[6:])
                            
                            # 检查是否为工具调用
                            if "tool_calls" in data:
                                tool_call = data["tool_calls"][0]
                                tool_call_data = {
                                    "tool_call": {
                                        "name": tool_call["name"],
                                        "arguments": json.dumps(tool_call["parameters"])
                                    }
                                }
                                # 工具调用不继续流式输出
                                break
                            
                            # 处理文本内容
                            if "content" in data and data["content"] and len(data["content"]) > 0:
                                for content_block in data["content"]:
                                    if content_block["type"] == "text":
                                        text = content_block["text"]
                                        yield text
                    
                    # 如果有工具调用，返回工具调用数据
                    if tool_call_data:
                        return tool_call_data
                
                return generate_stream()
            else:
                # 非流式请求
                response = requests.post(
                    f"{self.api_url}/v1/messages",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                response_data = response.json()
                
                # 检查是否为工具调用
                if "tool_calls" in response_data:
                    tool_call = response_data["tool_calls"][0]
                    return {
                        "tool_call": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["parameters"])
                        }
                    }
                
                # 处理普通文本响应
                content = ""
                for content_block in response_data["content"]:
                    if content_block["type"] == "text":
                        content += content_block["text"]
                
                return content
                
        except Exception as e:
            return {"error": f"Anthropic API错误: {str(e)}"} 