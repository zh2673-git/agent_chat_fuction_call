from .tool import Tool
import os

class WeatherTool(Tool):
    def execute(self, city: str) -> str:
        """
        查询城市天气（示例工具）。
        
        :param city: 要查询天气的城市名称，例如"北京"、"上海"、"苏州"等
        :return: 该城市的天气信息
        """
        # 添加调试输出
        print(f"[WEATHER DEBUG] Called with city: {city}")
        print(f"[WEATHER DEBUG] Current directory: {os.getcwd()}")
        print(f"[WEATHER DEBUG] Weather tool file path: {__file__}")
        
        # 固定结果，确保每个城市都是晴天25°C
        result = f"[TEST] Weather for {city}: Sunny, 25°C"
        print(f"[WEATHER DEBUG] Returning: {result}")
        
        return result
