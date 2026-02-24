import yaml
from pathlib import Path


class ModelConfigLoader:
    print("Flag ^^^66")
    def __init__(self, config_path: str = "models/models_config.yaml"):
        base_dir = Path(__file__).resolve().parent
        self.config_path = base_dir.parent/config_path
        self.models = self._load_yaml()

    def _load_yaml(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if "models" not in data:
            raise ValueError("YAML file must contain a 'models' key")

        return data["models"]

    def get_model_config(self, model_key: str) -> dict:
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found in config")

        return self.models[model_key]

    def list_models(self) -> list:
        return list(self.models.keys())