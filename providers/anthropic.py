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
        初始化Anthropic提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如claude-3-opus-20240229
        :param debug: 是否启用调试模式
        :param timeout: 请求超时时间(秒)
        :param max_retries: 最大重试次数
        :param extra_headers: 额外的请求头
        """
        # 调用父类初始化方法
        super().__init__(api_key, api_url, model_name, debug, timeout, max_retries, extra_headers)

    def _convert_tools(self, tools: list) -> list:
        """
        将通用工具格式转换为Anthropic格式
        这是对基类方法的覆盖，因为Anthropic需要特定的工具格式
        """
        if not tools:
            return []
            
        anthropic_tools = []
        for tool in tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {})
            }
            anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools

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
            # 确保API URL以正确的格式结尾
            if not self.api_url.endswith("/"):
                api_url = f"{self.api_url}/"
            else:
                api_url = self.api_url
                
            self._debug_print(f"使用API URL: {api_url}")
            self._debug_print(f"使用模型: {self.model_name}")
            
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
            
            # 调试输出请求头和请求体
            self._debug_print(f"请求头: {headers}")
            self._debug_print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
            
            # 添加工具调用支持
            if tools and len(tools) > 0:
                # 转换工具格式为Anthropic格式
                anthropic_tools = self._convert_tools(tools)
                payload["tools"] = anthropic_tools
                payload["tool_choice"] = "auto"
            
            # 添加流式支持
            if stream:
                payload["stream"] = True
                
                # 发送请求
                try:
                    self._debug_print(f"发送流式请求到: {api_url}v1/messages")
                    response = requests.post(
                        f"{api_url}v1/messages",
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=self.timeout
                    )
                    
                    if response.status_code != 200:
                        self._debug_print(f"流式响应错误: {response.status_code} - {response.text[:100]}")
                        return {"error": f"API错误: {response.status_code}"}
                except requests.exceptions.RequestException as e:
                    self._debug_print(f"流式请求异常: {str(e)}")
                    return {"error": f"请求异常: {str(e)}"}
                
                def generate_stream():
                    buffer = ""
                    tool_call_data = None
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        if line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                self._debug_print(f"流式数据: {json.dumps(data, ensure_ascii=False)[:100]}...")
                                
                                # 检查是否为工具调用
                                if "tool_calls" in data:
                                    tool_call = data["tool_calls"][0]
                                    tool_call_data = {
                                        "tool_calls": [{
                                            "id": f"call_{0}",
                                            "name": tool_call["name"],
                                            "arguments": json.dumps(tool_call["parameters"])
                                        }]
                                    }
                                    # 工具调用不继续流式输出
                                    break
                                
                                # 处理文本内容
                                if "content" in data and data["content"] and len(data["content"]) > 0:
                                    for content_block in data["content"]:
                                        if content_block["type"] == "text":
                                            text = content_block["text"]
                                            yield text
                            except json.JSONDecodeError as e:
                                self._debug_print(f"JSON解析错误: {str(e)}")
                    
                    # 如果有工具调用，返回工具调用数据
                    if tool_call_data:
                        return tool_call_data
                
                return generate_stream()
            else:
                # 非流式请求
                try:
                    self._debug_print(f"发送非流式请求到: {api_url}v1/messages")
                    response = requests.post(
                        f"{api_url}v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    self._debug_print(f"响应状态码: {response.status_code}")
                    if response.status_code != 200:
                        error_text = response.text[:200] if response.text else "无响应内容"
                        self._debug_print(f"API错误响应: {error_text}")
                        return {"error": f"API错误: {response.status_code} - {error_text}"}
                        
                    response_data = response.json()
                    self._debug_print(f"API响应: {json.dumps(response_data, ensure_ascii=False)[:200]}")
                    
                    # 检查是否为工具调用
                    if "tool_calls" in response_data:
                        tool_calls = []
                        for i, tool_call in enumerate(response_data["tool_calls"]):
                            tool_calls.append({
                                "id": f"call_{i}",
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["parameters"])
                            })
                        
                        return {"tool_calls": tool_calls}
                    
                    # 处理普通文本响应
                    content = ""
                    if "content" in response_data and isinstance(response_data["content"], list):
                        for content_block in response_data["content"]:
                            if isinstance(content_block, dict) and content_block.get("type") == "text":
                                content += content_block["text"]
                    
                    if not content:
                        self._debug_print("警告: 响应中没有找到文本内容")
                        return "无法获取有效回复"
                        
                    return content
                    
                except requests.exceptions.RequestException as e:
                    self._debug_print(f"请求异常: {str(e)}")
                    return {"error": f"请求异常: {str(e)}"}
                
        except Exception as e:
            self._debug_print(f"Anthropic API错误: {str(e)}")
            import traceback
            self._debug_print(f"错误堆栈: {traceback.format_exc()}")
            return {"error": f"Anthropic API错误: {str(e)}"} 