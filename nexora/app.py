from typing import Optional
import logging
import os
import logging
import sys
import getch

from .config import Config
from .logic import Logic
from .input import Input
from .logger import (logger, console)
from .exceptions import ModelNotFoundException

class App:
    def __init__(self) -> None:
        self.config = Config()
        if not self.config.settings:
            print("Config file created at ~/.config/nexora/config.yaml. Please edit it before running Nexora again.")
            
            sys.exit(0)

    def start(self, permissive: bool, directory: Optional[str], model_name: Optional[str] = None, provider: Optional[str] = None, verbose: Optional[bool] = None, api_key: Optional[str] = None, preload_prompt: Optional[str] = None) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        console.setLevel(level)
        logger.setLevel(level)

        # Check for rc file
        rc_file_path = os.path.join(directory if directory else '.', '.luminos_rc.yaml')

        if os.path.exists(rc_file_path):
            while True:
                print(f".luminos_rc.yaml detected in {directory or '.'}. Would you like to use it? (y/n): ", end='', flush=True)
                user_input = getch.getch().lower()
                print(user_input)  # echo the keypress
                if user_input in ['y']:
                    self.config.load_rc_config()
                    break
                elif user_input in ['n']:
                    break
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.")

        model_name = model_name or self.config.settings.get("defaults", {}).get("model")
        provider = provider or self.config.settings.get("defaults", {}).get("provider")

        if not model_name or not provider:
            logger.error("Default model name and/or provider not specified in configuration. Exiting.")
            sys.exit(1)

        try:
            model_class = self.config.get_model_class(provider, model_name)
        except ModelNotFoundException as e:
            logger.error(f"Error getting model class: {e}")
            sys.exit(1)

        # Fetch and merge settings
        provider_settings = self.config.settings.get('providers', {}).get(provider, {})
        model_settings = provider_settings.get('models', {}).get(model_name, {})
        effective_settings = {**provider_settings, **model_settings}

        # Instantiate model and dynamically set attributes
        model_instance = model_class()
        for attr, value in effective_settings.items():
            if hasattr(model_instance, attr):
                setattr(model_instance, attr, value)

        # Set model & API key if passed as an argument
        if api_key:
            model_instance.api_key = api_key

        if model_name:
            # Only set it if model is None
            if not model_instance.model:
                model_instance.model = model_name

        # Logic handles processes including model
        self.logic = Logic(self, model=model_instance)

        # Start handling inputs with optional preload prompt
        input_handler = Input(permissive=permissive, directory=directory, logic=self.logic)
        input_handler.start(preload_prompt=preload_prompt)
