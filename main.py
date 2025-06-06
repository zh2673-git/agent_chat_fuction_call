import sys
import os
from dotenv import load_dotenv  # 新增环境变量加载
from agent import ChatAgent

def init_environment():
    """初始化环境配置"""
    # 1. 加载.env文件
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"缺少.env文件，请从env.example复制模板")
    load_dotenv(env_path)
    
    # 2. 验证关键环境变量
    required_vars = ['MODELSCOPE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"缺少必需环境变量: {missing_vars}")

    print("=== 环境变量验证 ===")
    print("当前工作目录:", os.getcwd())
    print(".env 文件路径:", os.path.abspath('.env'))
    print("MODELSCOPE_API_KEY 是否存在:", "MODELSCOPE_API_KEY" in os.environ)
    print("MODELSCOPE_API_KEY 值:", os.getenv("MODELSCOPE_API_KEY")[:4] + "...")

def main():
    try:
        # 初始化环境（只执行一次）
        init_environment()  
        
        print("=== Agent 初始化 ===")
        print(f"使用的Token: {os.getenv('MODELSCOPE_API_KEY')[:4]}...")  # 只显示前4位
        
        # 检查是否启用调试模式
        debug_mode = False
        verbose_mode = False
        for arg in sys.argv[1:]:
            if arg in ['-d', '--debug']:
                debug_mode = True
            elif arg in ['-v', '--verbose']:
                verbose_mode = True
        
        if debug_mode:
            print("调试模式已启用，将显示详细日志")
        if verbose_mode:
            print("详细输出模式已启用")
            os.environ['VERBOSE'] = '1'
        
        # 创建Agent实例，传递debug参数
        agent = ChatAgent("modelscope", debug=debug_mode)
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
