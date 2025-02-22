import yaml

class ConfigLoader:
    def __init__(self, config_path="configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:  # ✅ Force UTF-8 encoding
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)
