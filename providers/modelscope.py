from openai import OpenAI
from .base import BaseProvider
import os
import requests
from typing import Union, Generator, Dict, Any, Optional, List
import json

class ModelscopeProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        """
        从配置创建实例
        :param config: 配置字典，包含api_key, api_url, model_name等
        """
        # 获取API密钥，优先使用环境变量
        api_key = os.getenv("MODELSCOPE_API_KEY", config.get("api_key"))
        
        # 获取API URL，默认使用DashScope兼容模式
        api_url = config.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1/").rstrip("/") + "/"
        
        # 获取模型名称，默认使用qwen-max
        model_name = config.get("model_name", "qwen-max")
        
        # 获取超时和重试设置
        timeout = config.get("timeout", 60)
        max_retries = config.get("max_retries", 3)
        
        # 获取调试模式
        debug = config.get("debug", False)
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            debug=debug
        )
    
    def __init__(self, api_key: str, api_url: str, model_name: str, 
                 timeout: int = 60, max_retries: int = 3, debug: bool = False, extra_headers: dict = None):
        """
        初始化ModelScope提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如qwen-max, Qwen/Qwen3-235B-A22B等
        :param timeout: 请求超时时间(秒)
        :param max_retries: 最大重试次数
        :param debug: 是否启用调试模式
        :param extra_headers: 额外的请求头
        """
        # 调用父类初始化方法
        super().__init__(api_key, api_url, model_name, debug, timeout, max_retries, extra_headers)
        
        # 添加ModelScope特定的请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Required": "true"
        }

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
            # 确保API URL以正确的格式结尾
            if not self.api_url.endswith("/"):
                api_url = f"{self.api_url}/"
            else:
                api_url = self.api_url
                
            url = f"{api_url}chat/completions"
            self._debug_print(f"使用API URL: {url}")
            self._debug_print(f"使用模型: {self.model_name}")
            
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
                "enable_thinking": False  # 关键参数，非流式调用必须为false
            }
            
            # 调试输出请求头和请求体
            self._debug_print(f"请求头: {self.headers}")
            self._debug_print(f"请求体: {json.dumps(data, ensure_ascii=False)}")
            
            # 添加工具信息
            modelscope_tools = self._convert_tools(tools)
            if modelscope_tools:
                data["tools"] = modelscope_tools
            
            # 对于流式输出，先检查是否有工具调用
            if stream:
                # 先使用非流式请求检查是否有工具调用
                data_no_stream = data.copy()
                data_no_stream["stream"] = False
                try:
                    check_response = requests.post(
                        url, 
                        headers=self.headers, 
                        json=data_no_stream,
                        timeout=self.timeout
                    )
                    
                    self._debug_print(f"非流式检查响应状态码: {check_response.status_code}")
                    if check_response.status_code != 200:
                        self._debug_print(f"API错误响应: {check_response.text}")
                        return {"error": f"API错误: {check_response.status_code} - {check_response.text[:100]}"}
                        
                    check_result = check_response.json()
                    self._debug_print(f"非流式检查响应: {json.dumps(check_result, ensure_ascii=False)[:200]}")
                    
                    tool_calls_result = self._extract_tool_calls(check_result)
                    if tool_calls_result:
                        return tool_calls_result
                except Exception as e:
                    self._debug_print(f"非流式检查请求失败: {str(e)}")
                    return {"error": f"请求失败: {str(e)}"}
                
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
                            self._debug_print(f"流式响应错误: {response.status_code} - {response.text[:100]}")
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
                try:
                    self._debug_print(f"发送非流式请求到: {url}")
                    response = requests.post(
                        url, 
                        headers=self.headers, 
                        json=data,
                        timeout=self.timeout
                    )
                    
                    self._debug_print(f"响应状态码: {response.status_code}")
                    if response.status_code != 200:
                        error_text = response.text[:200] if response.text else "无响应内容"
                        self._debug_print(f"API错误响应: {error_text}")
                        return {"error": f"API错误: {response.status_code} - {error_text}"}
                    
                    result = response.json()
                    self._debug_print(f"API响应: {json.dumps(result, ensure_ascii=False)[:200]}")
                    
                    # 使用基类中的工具调用提取方法
                    tool_calls_result = self._extract_tool_calls(result)
                    if tool_calls_result:
                        return tool_calls_result
                    
                    # 如果没有工具调用，返回内容
                    if "choices" in result and result["choices"] and isinstance(result["choices"], list):
                        choice = result["choices"][0]
                        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                            message = choice["message"]
                            if "content" in message and message["content"]:
                                content = message["content"]
                                
                                # 检查是否包含多个回复
                                if "\n\n问题1:" in content or "\n\n问题 1:" in content:
                                    # 使用基类中的方法尝试将内容分割成多个回复
                                    self._debug_print("检测到多个回复，尝试分割")
                                    return {
                                        "multi_responses": self._split_multiple_responses(content)
                                    }
                                
                                return content
                    
                    return "无法获取有效回复"
                except requests.exceptions.RequestException as e:
                    self._debug_print(f"请求异常: {str(e)}")
                    return {"error": f"请求异常: {str(e)}"}
            
        except Exception as e:
            self._debug_print(f"ModelScope API错误: {str(e)}")
            import traceback
            self._debug_print(f"错误堆栈: {traceback.format_exc()}")
            return {"error": f"ModelScope API错误: {str(e)}"}
