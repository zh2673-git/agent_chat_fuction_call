from .tool import Tool
import os
import inspect

class TranslationTool(Tool):
    """
    翻译工具，将输入翻译为目标语言。
    """
    def execute(self, prompt: str) -> str:
        """
        执行翻译逻辑。
        :param prompt: 用户输入（如 "翻译成英文"）
        :return: 翻译结果
        """
        # 详细的调试输出
        print(f"[TRANSLATION DEBUG] Called with prompt: {prompt}")
        print(f"[TRANSLATION DEBUG] Current directory: {os.getcwd()}")
        print(f"[TRANSLATION DEBUG] Translation tool file path: {__file__}")
        print(f"[TRANSLATION DEBUG] Translation tool class: {self.__class__.__name__}")
        print(f"[TRANSLATION DEBUG] Translation tool module: {self.__class__.__module__}")
        
        # 尝试输出调用堆栈，看看是谁调用了这个函数
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        print(f"[TRANSLATION DEBUG] Called from: {caller_frame[1].filename}, line {caller_frame[1].lineno}")
        
        # 保证固定返回Hello
        fixed_result = "[FIXED] Hello from the REAL translation tool"
        print(f"[TRANSLATION DEBUG] Returning: {fixed_result}")
        
        return fixed_result
        
if __name__ == "__main__":
    # 示例用法
    tool = TranslationTool({})
    print(tool.execute("翻译成英文"))