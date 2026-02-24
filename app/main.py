

from .core.model_loader import ModelConfigLoader

loader = ModelConfigLoader()

print("Available models:", loader.list_models())

config = loader.get_model_config("gemma3_local")
print("Selected model config:", config)