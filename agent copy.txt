from typing import Dict, Generator, List, Optional, Union
import json
import inspect
from LLM import LargeLanguageModel
from tools.tool import Tool  # 假设有一个基础工具类

class ChatAgent:
    def __init__(self, model_provider: str, tool_config_file: str = "tools/config.json", debug: bool = False):
        """
        聊天机器人Agent，集成LLM和工具调用。
        :param model_provider: 模型提供商（如 openai）
        :param tool_config_file: 工具配置文件路径
        :param debug: 是否启用调试模式，打印调试信息
        """
        self.llm = LargeLanguageModel(model_provider, debug=debug)
        self.tools = self._load_tools(tool_config_file)
        self.conversation_history: List[Dict[str, str]] = []  # 存储对话历史
        self.debug = debug  # 调试模式标志

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

        # 3. 构造工具列表，自动提取工具参数信息
        available_tools = []
        for name, tool_obj in self.tools.items():
            available_tools.append({
                "name": name,
                "description": tool_obj.description,
                "parameters": tool_obj.parameters  # 使用自动提取的参数信息
            })

        # 调试信息：打印工具列表
        self._debug_print(f"可用工具列表: {json.dumps(available_tools, indent=2)}")

        # 4. 调用LLM（始终传入工具列表，让模型自行判断是否需要工具）
        # 不再进行关键词匹配，而是依赖模型的判断能力
        llm_response = self.llm.generate_response(
            full_context,
            tools=available_tools,
            stream=True
        )

        # 调试信息：打印LLM响应
        self._debug_print(f"LLM响应: {llm_response}")

        # 5. 处理错误
        if "error" in llm_response:
            error_msg = f"错误: {llm_response['error']}"
            self._debug_print(error_msg)
            yield error_msg
            return

        # 6. 处理工具调用
        if "tool_call" in llm_response:
            tool_name = llm_response["tool_call"]["name"]
            tool_args_str = llm_response["tool_call"]["arguments"] # 接收到的参数可能是字符串
            
            # 调试信息
            self._debug_print(f"工具调用: {tool_name}")
            self._debug_print(f"工具参数(原始): {tool_args_str}")
            
            try:
                # 尝试解析工具参数为字典
                tool_args = json.loads(tool_args_str)
                
                # 调试信息
                self._debug_print(f"工具参数(解析后): {tool_args}")
                
                if not isinstance(tool_args, dict): # 确保解析结果是字典
                    raise ValueError("Tool arguments are not a valid JSON dictionary.")
                
                # 通用参数处理 - 检查参数和工具需求是否匹配
                if tool_name in self.tools:
                    self._debug_print(f"找到工具: {tool_name}")
                    tool = self.tools[tool_name]
                    # 获取工具的execute方法签名
                    sig = inspect.signature(tool.execute)
                    
                    # 获取必要参数列表(排除self)
                    required_params = [
                        param.name for param in sig.parameters.values() 
                        if param.default == inspect.Parameter.empty and param.name != 'self'
                    ]
                    self._debug_print(f"必需参数: {required_params}")
                    
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
                        # 如果仍然缺少参数，请求用户提供
                        missing_str = ", ".join(missing_params)
                        yield f"工具调用缺少必要参数: {missing_str}，请提供这些信息。"
                        return
                    
                    # 过滤掉多余的参数，只保留工具方法所需的参数
                    filtered_args = {k: v for k, v in tool_args.items() if k in sig.parameters}
                    
                    # 执行工具调用
                    self._debug_print(f"执行工具: {tool_name}，参数: {filtered_args}")
                    print(f"[DIRECT DEBUG] 调用{tool_name}工具前")
                    tool_result = tool.execute(**filtered_args)
                    print(f"[DIRECT DEBUG] 调用{tool_name}工具后，结果: {repr(tool_result)}")
                    self._debug_print(f"工具执行结果: {tool_result}")
                    
                    # 检查工具结果类型
                    print(f"[DIRECT DEBUG] 工具结果类型: {type(tool_result)}, 内容: {repr(tool_result)}")
                    
                    # 显示工具调用信息（非调试模式也显示）
                    final_tool_result = tool_result  # 保存原始工具结果
                    print(f"[DIRECT DEBUG] 最终使用的工具结果: {repr(final_tool_result)}")
                    
                    yield f"[工具调用] 使用工具: {tool_name}\n"
                    yield f"[工具参数] {json.dumps(filtered_args, ensure_ascii=False)}\n"
                    yield f"[工具结果] {final_tool_result}\n"
                    yield f"[模型回复] "
                    
                    # 获取一个基于工具结果的总结性回答
                    summary_prompt = f"""
用户问题: {prompt}
工具名称: {tool_name}
工具参数: {json.dumps(filtered_args, ensure_ascii=False)}
工具结果: {final_tool_result}

请根据以上信息，生成一个自然、友好的回答，直接回应用户的问题。
不要提及"工具调用"、"参数"等技术细节，而是像在自然对话中一样回答。
"""
                    summary_response = self.llm.generate_response(summary_prompt, stream=False)
                    
                    summary = ""
                    if "content" in summary_response:
                        summary = summary_response["content"]
                        for chunk in self._stream_text(summary):
                            yield chunk
                    else:
                        # 直接返回工具结果，不做特殊处理
                        for chunk in self._stream_text(final_tool_result):
                            yield chunk
                        summary = final_tool_result  # 如果没有总结，使用工具结果作为总结
                    
                    # 更新对话历史中的最后一条消息，包含完整的工具调用和模型回复
                    full_response = f"[工具调用] 使用工具: {tool_name}\n[工具参数] {json.dumps(filtered_args, ensure_ascii=False)}\n[工具结果] {final_tool_result}\n[模型回复] {summary}"
                    if len(self.conversation_history) > 0 and self.conversation_history[-1]["role"] == "assistant":
                        self.conversation_history[-1]["content"] = full_response
                    else:
                        self.conversation_history.append({"role": "assistant", "content": full_response})
                else:
                    yield f"未找到工具: {tool_name}"
            except json.JSONDecodeError:
                error_msg = f"工具参数解析失败，非法的 JSON 格式: {tool_args_str}"
                self._debug_print(f"错误: {error_msg}")
                yield error_msg
            except Exception as e:
                error_msg = f"工具执行失败: {e}"
                self._debug_print(f"错误: {error_msg}")
                yield error_msg
            return

        # 7. 处理流式输出
        full_response = []
        if "stream" in llm_response:
            try:
                for chunk in llm_response["stream"]:
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
        else: # 非流式输出
            content = llm_response.get("content", "")
            if not content:
                self._debug_print("警告: 接收到空的内容回复")
                yield "我没有找到相关的回答。"
                return
                
            # 输出非流式内容
            full_response.append(content)
            yield content

        # 8. 将模型回复添加到对话历史
        self.conversation_history.append({"role": "assistant", "content": "".join(full_response)})

# 示例用法
if __name__ == "__main__":
    agent = ChatAgent("openai", debug=False)  # 设置debug=True可以查看详细调试信息
    while True:
        user_input = input("\n用户输入: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Agent回复: ", end="", flush=True)
        for chunk in agent.run(user_input):
            print(chunk, end="", flush=True)
        print()
