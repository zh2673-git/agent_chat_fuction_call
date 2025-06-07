from typing import Dict, Generator, List, Optional, Union
import json
import inspect
from LLM import LargeLanguageModel
from tools.tool import Tool  # 假设有一个基础工具类

class ChatAgent:
    def __init__(self, model_provider: str, model_name: Optional[str] = None, tool_config_file: str = "tools/config.json", debug: bool = False):
        """
        聊天机器人Agent，集成LLM和工具调用。
        :param model_provider: 模型提供商（如 openai, modelscope）
        :param model_name: 具体模型名称（如 gpt-4, qwen-max）
        :param tool_config_file: 工具配置文件路径
        :param debug: 是否启用调试模式，打印调试信息
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.debug = debug  # 调试模式标志
        
        # 初始化LLM
        self.llm = LargeLanguageModel(model_provider, model_name, debug=debug)
        
        # 加载工具
        self.tools = self._load_tools(tool_config_file)
        
        # 初始化对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        if debug:
            # 打印当前使用的模型信息
            print(f"初始化Agent，使用提供商: {model_provider}, 模型: {model_name or '默认'}")
            # 打印可用的模型列表
            supported_models = self.get_supported_models(model_provider)
            print(f"支持的模型列表: {supported_models}")

    def _load_tools(self, config_file: str) -> Dict[str, Tool]:
        """
        动态加载所有工具。
        :param config_file: 工具配置文件路径
        :return: 工具名称到工具对象的映射
        """
        with open(config_file, "r") as f:
            configs = json.load(f)
        
        tools = {}
        for name, config in configs.items():
            # 动态导入工具类
            module = __import__(f"tools.{name}", fromlist=[f"{name.capitalize()}Tool"])
            tool_class = getattr(module, f"{name.capitalize()}Tool")
            tools[name] = tool_class(config)
        return tools
    
    def _debug_print(self, *args, **kwargs):
        """调试信息打印函数，只在调试模式下输出"""
        if self.debug:
            print(*args, **kwargs)
            
    def _stream_text(self, text: str) -> Generator[str, None, None]:
        """将文本转换为流式输出"""
        if not text:
            yield "无响应"
            return
            
        # 按句子分割，确保流式效果更自然
        sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 对于每个句子，按词输出
            words = sentence.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
            
            # 句子结束添加空格
            if sentence[-1] in ['.', '!', '?'] and sentences[-1] != sentence:
                yield " "

    def _prepare_tools_list(self) -> list:
        """准备工具列表，用于传递给LLM"""
        available_tools = []
        for name, tool_obj in self.tools.items():
            available_tools.append({
                "name": name,
                "description": tool_obj.description,
                "parameters": tool_obj.parameters  # 使用自动提取的参数信息
            })
        return available_tools
    
    def _execute_tool(self, tool_name: str, tool_args: dict) -> Union[str, dict]:
        """
        执行工具调用
        :param tool_name: 工具名称
        :param tool_args: 工具参数
        :return: 工具执行结果
        """
        if tool_name not in self.tools:
            return {"error": f"未找到工具: {tool_name}"}
            
        tool = self.tools[tool_name]
        
        # 获取工具的execute方法签名
        sig = inspect.signature(tool.execute)
        
        # 获取必要参数列表(排除self)
        required_params = [
            param.name for param in sig.parameters.values() 
            if param.default == inspect.Parameter.empty and param.name != 'self'
        ]
        
        # 检查是否缺少任何必要参数
        missing_params = [param for param in required_params if param not in tool_args]
        
        if missing_params:
            # 尝试使用可用的参数值填充缺失的参数
            # 这个逻辑会根据参数名称尝试寻找匹配的值
            for missing in missing_params:
                # 策略1: 检查是否有同名的键但大小写不同
                for key in tool_args:
                    if key.lower() == missing.lower():
                        tool_args[missing] = tool_args[key]
                        break
                
                # 策略2: 如果只有一个字符串参数值和一个必需参数，尝试匹配
                if len(required_params) == 1 and len(missing_params) == 1:
                    string_values = [v for v in tool_args.values() if isinstance(v, str)]
                    if len(string_values) == 1:
                        tool_args[missing] = string_values[0]
        
        # 再次检查必要参数是否都已存在
        missing_params = [param for param in required_params if param not in tool_args]
        
        if missing_params:
            # 如果仍然缺少参数，返回错误
            missing_str = ", ".join(missing_params)
            return {"error": f"工具调用缺少必要参数: {missing_str}"}
        
        # 过滤掉多余的参数，只保留工具方法所需的参数
        filtered_args = {k: v for k, v in tool_args.items() if k in sig.parameters}
        
        try:
            # 执行工具调用
            self._debug_print(f"执行工具: {tool_name}，参数: {filtered_args}")
            result = tool.execute(**filtered_args)
            self._debug_print(f"工具执行结果: {result}")
            return result
        except Exception as e:
            self._debug_print(f"工具执行错误: {e}")
            return {"error": f"工具执行失败: {e}"}

    def _handle_tool_call(self, response: dict, original_prompt: str) -> Generator[str, None, None]:
        """
        处理工具调用响应，统一处理所有工具调用，不再区分单工具和多工具
        :param response: 工具调用响应，格式为 {"tool_calls": [{"id": str, "name": str, "arguments": str}, ...]}
        :param original_prompt: 原始用户提问
        :return: 生成器，逐块返回流式输出
        """
        # 确保响应中包含工具调用
        if "tool_calls" not in response or not response["tool_calls"]:
            yield "未找到有效的工具调用"
            return
        
        tool_calls = response["tool_calls"]
        tool_count = len(tool_calls)
        self._debug_print(f"检测到 {tool_count} 个工具调用")
        
        # 存储所有工具执行结果
        tool_results = []
        
        # 依次处理每个工具调用
        for i, tool_call in enumerate(tool_calls):
            tool_id = tool_call.get("id", f"call_{i}")
            tool_name = tool_call["name"]
            tool_args_str = tool_call["arguments"]
            
            yield f"[工具调用 {i+1}/{tool_count}] 使用工具: {tool_name}\n"
            
            try:
                # 尝试解析工具参数为字典
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                
                if not isinstance(tool_args, dict): # 确保解析结果是字典
                    raise ValueError("工具参数不是有效的JSON字典")
                
                # 显示工具参数
                yield f"[工具参数] {json.dumps(tool_args, ensure_ascii=False)}\n"
                
                # 执行工具调用
                tool_result = self._execute_tool(tool_name, tool_args)
                
                # 检查是否有错误
                if isinstance(tool_result, dict) and "error" in tool_result:
                    error_msg = f"[工具执行错误] {tool_result['error']}"
                    yield f"{error_msg}\n"
                    tool_results.append({
                        "id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                        "result": error_msg,
                        "success": False
                    })
                else:
                    # 显示工具结果
                    yield f"[工具结果] {tool_result}\n"
                    tool_results.append({
                        "id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                        "success": True
                    })
                
                # 如果不是最后一个工具调用，添加分隔符
                if i < tool_count - 1:
                    yield "\n---\n\n"
                
            except json.JSONDecodeError:
                error_msg = f"工具参数解析失败，非法的 JSON 格式: {tool_args_str}"
                self._debug_print(f"错误: {error_msg}")
                yield f"[错误] {error_msg}\n"
                tool_results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "args": tool_args_str,
                    "result": error_msg,
                    "success": False
                })
            except Exception as e:
                error_msg = f"工具执行失败: {e}"
                self._debug_print(f"错误: {error_msg}")
                yield f"[错误] {error_msg}\n"
                tool_results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "args": tool_args_str if 'tool_args' not in locals() else tool_args,
                    "result": error_msg,
                    "success": False
                })
        
        # 所有工具都执行完毕后，生成总结性回复
        yield f"[模型回复] "
        
        # 构建带有所有工具执行结果的新提示
        tool_results_summary = []
        for i, result in enumerate(tool_results):
            if result["success"]:
                tool_results_summary.append(
                    f"工具{i+1}: {result['name']}\n"
                    f"参数: {json.dumps(result['args'], ensure_ascii=False)}\n"
                    f"结果: {result['result']}"
                )
            else:
                tool_results_summary.append(
                    f"工具{i+1}: {result['name']}\n"
                    f"参数: {json.dumps(result['args'], ensure_ascii=False)}\n"
                    f"结果: {result['result']} (执行失败)"
                )
        
        multi_tools_prompt = f"""
用户问题: {original_prompt}

以下是工具的执行结果:
{"=" * 40}
{("\n" + "=" * 40 + "\n").join(tool_results_summary)}
{"=" * 40}

请根据以上所有工具的执行结果，生成一个综合性的、自然友好的回答，直接回应用户的问题。
不要提及"工具调用"、"参数"等技术细节，而是像在自然对话中一样回答。
如果有多个相关问题，请分别回答每个问题。
"""
        
        # 将工具执行结果传回给LLM，获取总结性回答
        summary_response = self.llm.generate_response(multi_tools_prompt, stream=False)
        
        # 处理总结响应
        if isinstance(summary_response, str):
            # 直接是文本响应
            for chunk in self._stream_text(summary_response):
                yield chunk
            
            # 更新对话历史
            full_response = f"[工具调用结果]\n{summary_response}"
            self.conversation_history.append({"role": "assistant", "content": full_response})
        else:
            # 如果没有得到有效的总结，返回简单提示
            yield "已完成工具调用，但无法生成综合回复。"

    def run(self, prompt: str) -> Generator[str, None, None]:
        """
        处理用户输入，支持流式输出，并维护对话历史。
        :param prompt: 用户输入
        :return: 生成器，逐块返回流式输出
        """
        # 1. 将用户输入添加到对话历史
        self.conversation_history.append({"role": "user", "content": prompt})

        # 2. 构造完整的对话上下文（包括历史）
        full_context = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        )

        # 3. 准备工具列表
        available_tools = self._prepare_tools_list()
        self._debug_print(f"可用工具列表: {json.dumps(available_tools, indent=2)}")

        # 4. 调用LLM（始终传入工具列表，让模型自行判断是否需要工具）
        response = self.llm.generate_response(
            full_context,
            tools=available_tools,
            stream=True
        )
        self._debug_print(f"LLM原始响应类型: {type(response)}")

        # 5. 处理错误响应
        if isinstance(response, dict) and "error" in response:
            error_msg = f"错误: {response['error']}"
            self._debug_print(error_msg)
            yield error_msg
            return

        # 6. 处理工具调用响应
        if isinstance(response, dict):
            if "tool_calls" in response:
                # 使用统一的方法处理工具调用
                for chunk in self._handle_tool_call(response, prompt):
                    yield chunk
                return
            elif "multi_responses" in response:
                # 处理多个回复
                self._debug_print(f"检测到多个回复: {len(response['multi_responses'])}")
                for i, resp in enumerate(response['multi_responses']):
                    yield f"回复 {i+1}/{len(response['multi_responses'])}:\n{resp}\n\n"
                
                # 更新对话历史
                full_response = "\n\n".join(response['multi_responses'])
                self.conversation_history.append({"role": "assistant", "content": full_response})
                return

        # 7. 处理流式输出
        full_response = []
        if isinstance(response, Generator):
            try:
                for chunk in response:
                    if isinstance(chunk, dict) and "error" in chunk:
                        # 处理流式输出中的错误
                        error_msg = chunk["error"]
                        self._debug_print(f"流式输出错误: {error_msg}")
                        yield f"错误: {error_msg}"
                        return
                        
                    # 检查流式输出中的工具调用
                    if isinstance(chunk, dict) and "tool_calls" in chunk:
                        # 处理流式工具调用
                        self._debug_print("在流式输出中检测到工具调用")
                        for tool_chunk in self._handle_tool_call(chunk, prompt):
                            yield tool_chunk
                        return
                        
                    if chunk:  # 确保块不为空
                        full_response.append(chunk)
                        yield chunk
                        
            except Exception as e:
                self._debug_print(f"流式输出错误: {e}")
                # 如果流式输出失败，尝试返回已收集的内容
                if full_response:
                    yield "".join(full_response)
                else:
                    yield f"生成回复时出错: {e}"
        else: 
            # 处理普通字符串响应
            if isinstance(response, str):
                content = response
            else:
                # 处理其他类型响应
                content = str(response)
                
            if not content:
                self._debug_print("警告: 接收到空的内容回复")
                yield "我没有找到相关的回答。"
                return
                
            # 输出非流式内容
            full_response.append(content)
            yield content

        # 8. 将模型回复添加到对话历史
        if full_response:
            self.conversation_history.append({"role": "assistant", "content": "".join(full_response)})

    def get_supported_models(self, provider_name: Optional[str] = None) -> Dict[str, list]:
        """
        获取支持的模型列表
        :param provider_name: 提供商名称，如果为None则返回所有提供商的模型
        :return: 提供商名称到模型列表的映射
        """
        return self.llm.get_supported_models(provider_name)
    
    def get_current_model_info(self) -> Dict[str, str]:
        """
        获取当前使用的模型信息
        :return: 包含提供商和模型名称的字典
        """
        return {
            "provider": self.model_provider,
            "model": self.model_name or "默认"
        }

# 示例用法
if __name__ == "__main__":
    agent = ChatAgent("modelscope", model_name="deepseek-ai/DeepSeek-V3-0324", debug=True)
    while True:
        user_input = input("\n用户输入: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Agent回复: ", end="", flush=True)
        for chunk in agent.run(user_input):
            print(chunk, end="", flush=True)
        print()
