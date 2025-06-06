from openai import OpenAI
from LLM import BaseProvider
import os
import requests
from typing import Union, Generator
import json

class ModelscopeProvider(BaseProvider):
    @classmethod
    def from_config(cls, config: dict):
        return cls(
            api_key=os.getenv("MODELSCOPE_API_KEY"),
            api_url=config["api_url"].rstrip("/") + "/",
            model_name=config["model_name"]
        )
    
    def __init__(self, api_key: str, api_url: str, model_name: str):
        if not api_key or api_key == "use_env_variable":
            raise ValueError("MODELSCOPE_API_KEY 未配置或无效")
        
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "X-ModelScope-Required": "true"
            }
        )
        self.model_name = model_name

    def generate_response(self, prompt: str, tools: list, stream: bool = False) -> Union[str, dict, Generator[str, None, None]]:
        try:
            # 动态构建工具信息
            openai_tools = []
            if tools and len(tools) > 0:
                openai_tools = [
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
            
            # 判断是否需要进行工具调用处理
            if openai_tools:
                # 使用REST API直接调用
                url = f"{self.client.base_url}chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json",
                    "X-ModelScope-Required": "true"
                }
                
                # 添加系统提示以增强参数提取能力
                system_message = """
                你是一个能够调用工具的助手。当用户询问可以用工具解决的问题时，请确保：
                1. 正确识别用户查询中的关键信息
                2. 选择合适的工具
                3. 提取查询中的所有相关参数值并填入工具参数中
                4. 不要遗漏用户提到的任何关键信息
                
                例如，如果用户询问"查询北京天气"，你应该调用weather工具并将"北京"作为city参数传递。
                """
                
                data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": openai_tools,
                    "stream": stream,
                    "enable_thinking": False  # 关键参数，非流式调用必须为false
                }
                
                # 如果启用流式输出，使用流式处理
                if stream:
                    # 先使用非流式请求检查是否有工具调用
                    data_no_stream = data.copy()
                    data_no_stream["stream"] = False
                    check_response = requests.post(url, headers=headers, json=data_no_stream)
                    
                    if check_response.status_code == 200:
                        check_result = check_response.json()
                        # 检查是否有工具调用
                        if "choices" in check_result and check_result["choices"][0].get("message", {}).get("tool_calls"):
                            tool_call = check_result["choices"][0]["message"]["tool_calls"][0]
                            return {
                                "tool_call": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"]
                                }
                            }
                    
                    # 如果没有工具调用，使用流式输出
                    def stream_response():
                        with requests.post(url, headers=headers, json=data, stream=True) as response:
                            if response.status_code != 200:
                                yield f"API错误: {response.status_code}"
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
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        result = response.json()
                        # 检查是否有工具调用
                        if "choices" in result and result["choices"][0].get("message", {}).get("tool_calls"):
                            tool_call = result["choices"][0]["message"]["tool_calls"][0]
                            return {
                                "tool_call": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"]
                                }
                            }
                        # 如果没有工具调用，返回内容
                        if "choices" in result and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                            return result["choices"][0]["message"]["content"]
                        return "无法获取有效回复"
                    else:
                        return f"API错误: {response.status_code} - {response.text}"
            
            # 没有工具或普通对话时，使用非流式API调用以避免流式问题
            url = f"{self.client.base_url}chat/completions"
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json",
                "X-ModelScope-Required": "true"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "enable_thinking": False
            }
            
            # 如果启用流式输出，使用流式处理
            if stream:
                # 先使用非流式请求检查是否有工具调用
                data_no_stream = data.copy()
                data_no_stream["stream"] = False
                check_response = requests.post(url, headers=headers, json=data_no_stream)
                
                if check_response.status_code == 200:
                    check_result = check_response.json()
                    # 检查是否有工具调用
                    if "choices" in check_result and check_result["choices"][0].get("message", {}).get("tool_calls"):
                        tool_call = check_result["choices"][0]["message"]["tool_calls"][0]
                        return {
                            "tool_call": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"]
                            }
                        }
                
                # 普通对话的流式输出处理
                def stream_chat():
                    with requests.post(url, headers=headers, json=data, stream=True) as response:
                        if response.status_code != 200:
                            yield f"API错误: {response.status_code}"
                            return
                            
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
                
                return stream_chat()
            else:
                # 非流式处理
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                        return result["choices"][0]["message"]["content"]
                    return "无法获取有效回复"
                else:
                    return f"API错误: {response.status_code} - {response.text}"
            
        except Exception as e:
            return f"API错误: {str(e)}"
