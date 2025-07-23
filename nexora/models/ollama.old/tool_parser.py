from markdown2 import Markdown

import json
import re
from html.parser import HTMLParser

from .messages.tool_call import ToolCall

markdowner = Markdown()

class Function(object):
    name = None
    arguments = None

class CodeBlockParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.code_blocks = []
        self.current_code_block = None
        self.in_code_block = False

    def handle_starttag(self, tag, attrs):
        if tag == 'code':
            self.in_code_block = True
            self.current_code_block = []

    def handle_endtag(self, tag):
        if tag == 'code':
            self.in_code_block = False
            code_block = ''.join(self.current_code_block)

            if code_block.startswith('tool_call'):
                try:
                    json_data = '\n'.join(code_block.split('\n')[1:])
                    data = json.loads(json_data)
                    self.code_blocks.append(data)
                except json.JSONDecodeError:
                    pass
            self.current_code_block = None

    def handle_data(self, data):
        if self.in_code_block:
            self.current_code_block.append(data)

def parse_html(html_content):
    parser = CodeBlockParser()
    parser.feed(html_content)
    return parser.code_blocks

def tool_parser(msg):
    tool_calls = []

    m = markdowner.convert(msg)

    # Now you can iterate over the child elements (or attributes) of the root:
    for tool in parse_html(m):
        func = Function()
        func.name = tool["name"]
        func.arguments = tool["args"]

        use_id = tool["use_id"]

        tool_call = ToolCall(func, "function", use_id)

        tool_calls.append(tool_call)

    return tool_calls

if __name__ == "__main__":
    example_call = """I'll go ahead and retrieve that website for you.

```tool_call
{"name": "http", "args": {"method": "GET", "url": "https://example.com/"}, "use_id": 1}
```

```tool_call
{"name": "http", "args": {"method": "GET", "url": "https://example.com/"}, "use_id": 2}
```
"""

    tool_calls = tool_parser(example_call)

    for func in tool_calls:
        print(func.name, func.arguments)