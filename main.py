import sys
import os
from dotenv import load_dotenv  # 新增环境变量加载
from agent import ChatAgent

def init_environment(provider="modelscope"):
    """
    初始化环境配置
    :param provider: 提供商名称，用于检查对应的API密钥
    """
    # 1. 加载.env文件
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"缺少.env文件，请从env.example复制模板")
    load_dotenv(env_path)
    
    # 2. 根据提供商名称确定API密钥环境变量名
    api_key_map = {
        "modelscope": "MODELSCOPE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "groq": "GROQ_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "baichuan": "BAICHUAN_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY"
    }
    
    # 获取当前提供商对应的API密钥环境变量名
    api_key_env = api_key_map.get(provider, f"{provider.upper()}_API_KEY")
    
    # 3. 验证API密钥是否存在
    if not os.getenv(api_key_env):
        raise ValueError(f"缺少必需的API密钥环境变量: {api_key_env}")

    print("=== 环境变量验证 ===")
    print("当前工作目录:", os.getcwd())
    print(".env 文件路径:", os.path.abspath('.env'))
    print(f"{api_key_env} 是否存在:", api_key_env in os.environ)
    if api_key_env in os.environ:
        api_key_value = os.getenv(api_key_env)
        print(f"{api_key_env} 值:", api_key_value[:4] + "..." if api_key_value else "未设置")

def list_providers():
    """列出可用的提供商"""
    try:
        import json
        with open("config/provider_config.json", 'r', encoding='utf-8') as f:
            providers = json.load(f)
        
        print("\n=== 可用的模型提供商 ===")
        for name, config in providers.items():
            provider_name = config.get("provider_name", name)
            default_model = config.get("default_model", "无默认模型")
            print(f"- {name} ({provider_name}): {default_model}")
        print()
    except Exception as e:
        print(f"获取提供商列表失败: {e}")

def main():
    try:
        # 解析命令行参数
        debug_mode = False
        verbose_mode = False
        provider = "siliconflow"  # 默认使用硅基流动
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg in ['-d', '--debug']:
                debug_mode = True
            elif arg in ['-v', '--verbose']:
                verbose_mode = True
            elif arg in ['-p', '--provider']:
                if i + 1 < len(sys.argv):
                    provider = sys.argv[i + 1]
                    i += 1
                else:
                    print("错误: --provider 参数需要提供商名称")
                    return
            elif arg in ['-l', '--list']:
                list_providers()
                return
            elif arg in ['-h', '--help']:
                print("用法: python main.py [选项]")
                print("选项:")
                print("  -d, --debug         启用调试模式")
                print("  -v, --verbose       启用详细输出")
                print("  -p, --provider NAME 使用指定的提供商")
                print("  -l, --list          列出所有可用的提供商")
                print("  -h, --help          显示此帮助信息")
                return
            i += 1
        
        # 初始化环境（只执行一次）
        init_environment(provider)  
        
        print("=== Agent 初始化 ===")
        
        if debug_mode:
            print("调试模式已启用，将显示详细日志")
        if verbose_mode:
            print("详细输出模式已启用")
            os.environ['VERBOSE'] = '1'
        
        print(f"使用提供商: {provider}")
        
        # 创建Agent实例，传递debug参数
        agent = ChatAgent(provider, debug=debug_mode)
        
        # 获取并显示当前使用的模型
        model_info = agent.get_current_model_info()
        print(f"使用模型: {model_info.get('model', '默认模型')}")
        
        print("Agent已成功初始化，可以开始交流")
        
        while True:
            user_input = input("\n用户输入: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # 强制流式输出
            print("Agent回复: ", end="", flush=True)
            for chunk in agent.run(user_input):  # 直接迭代生成器
                print(chunk, end="", flush=True)
            print()  # 换行
            
    except KeyboardInterrupt:
        print("\n退出交互")
    except Exception as e:
        print(f"系统错误: {str(e)}")
        if debug_mode:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
