import os
import yaml

from nexora.models.openai.gpt35 import GPT35  
from nexora.models.anthropic import BaseAnthropic
from nexora.models.openai.gpt4 import GPT4
from nexora.models.openai.gpt4o import GPT4o
from nexora.models.ollama import Ollama

from nexora.exceptions import ModelNotFoundException

class Config:
    defaults = {
        'defaults': {
            'model': 'gpt-4',
            'provider': "openai",
            # Additional default configuration options can be added here
        },
        "providers": {
            "openai": {
                "api_key": None,
                "models": {
                    "gpt-4": {
                        "temperature": None
                    }
                }
            },
            "anthropic": {
                "api_key": None,
                "models": {}
            }
        }
    }

    example = """defaults:
  model: gpt-4
  provider: openai

# Example configuration for provider and model-specific settings, 
# such as API keys, temperature, etc.

# providers: 
#   openai:
#     api_key: 'sk-thisisasecretkey'
#     models:
#       gpt-4:
#         temperature: 0.75
#   anthropic:
#     api_key: 'sk-thisisalsoasecretkeylol'"""
    
    def __init__(self):
        self.config_path = os.path.expanduser('~/.config/nexora/config.yaml')
        self.settings = self.load_config()
        # self.rc_settings = self.load_rc_config()

    def merge_configs(self, default, existing):
        merged = existing.copy()
        for key, value in default.items():
            if isinstance(value, dict):
                merged[key] = self.merge_configs(value, merged.get(key, {}))
            else:
                merged.setdefault(key, value)
        return merged

    def load_config(self):
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as config_file:
                config_file.write(self.example)
                print('Default config generated at:', self.config_path)
            return self.defaults
        else:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            return self.merge_configs(self.defaults, config)

    def load_rc_config(self):
        rc_file_path = '.luminos_rc.yaml'
        
        if os.path.exists(rc_file_path):
            with open(rc_file_path, 'r') as file:
                self.settings = self.merge_configs(self.settings, yaml.safe_load(file))

    def get(self, key, default=None):
        # Recursively get a key from settings, fallback to default example_config if not available
        def get_recursive(key_list, settings, default):
            if key_list:
                key = key_list.pop(0)
                if key in settings:
                    return get_recursive(key_list, settings[key], default.get(key, None))
                else:
                    return default.get(key, None)
            else:
                return settings

        key_list = key.split('.')
        return get_recursive(key_list, self.settings, self.defaults)

    def get_model_class(self, provider: str, model_name: str):
        """
        Maps a model name and provider to its corresponding class object.
        
        Parameters:
        - provider: The provider of the model (e.g., 'openai', 'anthropic').
        - model_name: The name of the model (e.g., 'gpt-4').
        
        Returns:
        - The class object corresponding to the model and provider.
        
        Throws:
        - ModelNotFoundException: If the combination of model name and provider does not map to any class.
        """
        # Define a nested dictionary mapping providers to models and their respective classes
        model_map = {
            'openai': {
                'gpt-3.5': GPT35,
                'gpt-4': GPT4,
                'gpt-4o': GPT4o,
            },
            'anthropic': {
                'default': BaseAnthropic,  # Assuming a single class for any Anthropics' model
            },
            "ollama": {
                "default": Ollama
            }
        }
        
        if provider not in model_map:
            raise ModelNotFoundException(f"Provider '{provider}' not found.")
        
        # For providers with a single class for all models, use a 'default' key
        provider_models = model_map[provider]
        model_class = provider_models.get(model_name, provider_models.get('default'))
        
        if model_class is None:
            raise ModelNotFoundException(f"Model '{model_name}' for provider '{provider}' not found.")
        
        return model_class