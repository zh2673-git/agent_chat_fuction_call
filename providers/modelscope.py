from openai import OpenAI
from .base import BaseProvider
import os
import requests
from typing import Union, Generator, Dict, Any, Optional
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
        
        return cls(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name
        )
    
    def __init__(self, api_key: str, api_url: str, model_name: str):
        """
        初始化ModelScope提供商
        :param api_key: API密钥
        :param api_url: API URL
        :param model_name: 模型名称，例如qwen-max, Qwen/Qwen3-235B-A22B等
        """
        if not api_key or api_key == "use_env_variable":
            raise ValueError("MODELSCOPE_API_KEY 未配置或无效")
        
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Required": "true"
        }
        
        print(f"初始化ModelScope提供商，使用模型: {model_name}")

    def _convert_tools(self, tools: list) -> list:
        """将通用工具格式转换为ModelScope格式"""
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

    def _check_tool_call(self, response_json: dict) -> Optional[Dict[str, Any]]:
        """检查响应中是否包含工具调用"""
        if ("choices" in response_json and 
            response_json["choices"] and 
            response_json["choices"][0].get("message", {}).get("tool_calls")):
            
            tool_call = response_json["choices"][0]["message"]["tool_calls"][0]
            return {
                "tool_call": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                }
            }
        return None

    def generate_response(self, prompt: str, tools: list, stream: bool = False) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        生成响应，返回格式与BaseProvider接口一致
        :return: 
            - 工具调用: {"tool_call": {"name": str, "arguments": str}}
            - 流式输出: Generator[str, None, None]
            - 普通文本: str
            - 错误信息: {"error": str}
        """
        try:
            url = f"{self.api_url}chat/completions"
            
            # 添加系统提示以增强参数提取能力
            system_message = """
            你是一个能够调用工具的助手。当用户询问可以用工具解决的问题时，请确保：
            1. 正确识别用户查询中的关键信息
            2. 选择合适的工具
            3. 提取查询中的所有相关参数值并填入工具参数中
            4. 不要遗漏用户提到的任何关键信息
            """
            
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
            
            # 添加工具信息
            modelscope_tools = self._convert_tools(tools)
            if modelscope_tools:
                data["tools"] = modelscope_tools
            
            # 对于流式输出，先检查是否有工具调用
            if stream:
                # 先使用非流式请求检查是否有工具调用
                data_no_stream = data.copy()
                data_no_stream["stream"] = False
                check_response = requests.post(url, headers=self.headers, json=data_no_stream)
                
                if check_response.status_code == 200:
                    check_result = check_response.json()
                    tool_call_result = self._check_tool_call(check_result)
                    if tool_call_result:
                        return tool_call_result
                
                # 如果没有工具调用，使用流式输出
                def stream_response():
                    with requests.post(url, headers=self.headers, json=data, stream=True) as response:
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
                                            if "content" in delta and delta["content"]:
                                                yield delta["content"]
                                    except json.JSONDecodeError:
                                        pass
                
                return stream_response()
            else:
                # 非流式处理
                response = requests.post(url, headers=self.headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    
                    # 检查是否有工具调用
                    tool_call_result = self._check_tool_call(result)
                    if tool_call_result:
                        return tool_call_result
                    
                    # 如果没有工具调用，返回内容
                    if "choices" in result and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                        return result["choices"][0]["message"]["content"]
                    
                    return "无法获取有效回复"
                else:
                    return {"error": f"API错误: {response.status_code} - {response.text}"}
            
        except Exception as e:
            return {"error": f"ModelScope API错误: {str(e)}"}
