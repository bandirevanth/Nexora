import os
import importlib

class Commands:
    def __init__(self, input, logic):
        self.input = input
        self.logic = logic
        self.commands = {}
        self.load_commands()

    def load_commands(self):
        commands_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "commands.py":
                module_name = filename[:-3]
                module = importlib.import_module(f"nexora.commands.{module_name}")
                command_class = getattr(module, module_name.capitalize())
                self.commands[module_name] = command_class()

    def execute(self, user_input):
        parts = user_input[1:].split()
        command_name = parts[0]
        args = parts[1:]

        try:
            if command_name == 'help':  # Handle /help command separately
                return self.help()

            if command_name in self.commands:
                command = self.commands[command_name]
                return command.run(*args)
            else:
                return f"Unknown command: {command_name}"
        except Exception as e:
            logger.error(f"Exception in command '{command_name}': {e}")
            return f"Error executing command: {str(e)}"


    def help(self):
        help_text = 'Available commands:\n'
        for command_name, command_class in self.commands.items():
            docstring = command_class.run.__doc__.strip() if command_class.run.__doc__ else 'No documentation available.'
            help_text += f'/{command_name}: {docstring}\n'

        return help_text