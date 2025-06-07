from .base import BaseProvider
from .openai_compatible import OpenAICompatibleProvider
from .siliconflow import SiliconflowProvider
from .openrouter import OpenrouterProvider

__all__ = [
    "BaseProvider",
    "OpenAICompatibleProvider",
    "SiliconflowProvider",
    "OpenrouterProvider",
    "get_provider"
]

def get_provider(provider_name: str, config: dict = None):
    """
    根据提供商名称获取对应的提供商实例
    :param provider_name: 提供商名称
    :param config: 提供商配置
    :return: 提供商实例
    """
    if not config:
        config = {}
        
    providers = {
        "openai": OpenAICompatibleProvider,
        "openai_compatible": OpenAICompatibleProvider,
        "siliconflow": SiliconflowProvider,
        "openrouter": OpenrouterProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"不支持的提供商: {provider_name}")
        
    return providers[provider_name].from_config(config)
