from docstring_parser import parse

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings

from uuid import uuid4

import os

# Define custom styles for the permission request and key bindings
style = Style.from_dict({
    'permission': 'bg:ansiyellow ansiwhite bold',
    'action': 'bg:ansiblue ansiwhite bold',
    'preview': 'bg:ansigreen ansiwhite bold',
    'normal': '' # Default text style
})

bindings = KeyBindings()

@bindings.add('y')
def accept(event):
    event.app.exit('Y')

@bindings.add('n')
def decline(event):
    event.app.exit('N')

@bindings.add('p')
def preview(event):
    event.app.exit('P')

@bindings.add('c-q')
def exit_app(event):
    event.app.exit()

class BaseTool:
    name = "basetool"

    @property
    def __func__(self):
        """Converts the BaseTool system to the OpenAI-friendly 'functions' methodology."""
        func = []

        for attr in dir(self):
            if attr.startswith("__"):
                continue

            method = getattr(self, attr)

            if not callable(method) or not method.__doc__:
                continue

            docstring = parse(method.__doc__)

            if not docstring.short_description.startswith("openai.function: "):
                continue

            properties = {}

            for param in docstring.params:
                if param.type_name == "int":
                    type_name = "number"
                elif param.type_name == "str":
                    type_name = "string"
                elif param.type_name == "bool":
                    type_name = "boolean"
                elif param.type_name == "list":
                    type_name = "array"

                properties[param.arg_name] = {
                    "type": type_name,
                    "description": param.description
                }

                if type_name == "array":
                    properties[param.arg_name]["items"] = {
                        "type": "string"
                    }

            try:
                required = docstring.long_description.split(",")
            except AttributeError:
                required = []

            function = {
                "name": f"{self.name}_{attr}",
                "description": docstring.short_description[17:],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }

            func.append({
                "type": "function",
                "function": function
            })

        return func

    def safe(self, reason, preview=None):
        if os.getenv('ALWAYS_GRANT_PERMISSION', '0') == '1':
            return
        
        permission_request = FormattedText([
            ('class:permission', '\n🛑  Permission Requested!  🛑\n'),
            ('class:action', f'Action: {reason}\n'),
            ('class:preview', 'Preview changes with the editor? (P) ' if preview else ''),
            ('class:normal', 'Grant permission? (Y/N, P to preview, Ctrl+Q to quit): ')
        ])
        print_formatted_text(permission_request, style=style)

        user_input = prompt('', key_bindings=bindings).strip().upper()
        if user_input == 'Y':
            return True
        elif user_input == 'N':
            raise PermissionError("Permission denied by the user.")
        elif user_input == 'P' and preview:
            path = f"/tmp/{str(uuid4())}"

            with open(path, "w") as f:
                f.write(preview)

            os.system(f'{os.getenv("EDITOR", "nano")} {path}')
            
            os.remove(path)

            return self.safe(reason, preview)  # Re-prompt after preview
        else:
            self.safe(reason, preview) # Recursive call if any other key is pressed