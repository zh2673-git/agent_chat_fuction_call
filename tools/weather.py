from .tool import Tool

class WeatherTool(Tool):
    def execute(self, city: str) -> str:
        """
        查询城市天气（示例工具）。
        """
        api_url = f"{self.config['api_url']}?city={city}"
        try:
            response = requests.get(api_url, headers={"Authorization": self.config["api_key"]})
            return response.json()["weather"]
        except Exception as e:
            return f"天气查询失败: {e}"
