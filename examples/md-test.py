from markdown2 import Markdown

import json
import re
from html.parser import HTMLParser

markdowner = Markdown()

tool_call = """I'll go ahead and retrieve that website for you.

```tool_call
{"function": "http", "args": {"method": "GET", "url": "https://example.com/"}}
```

```tool_call
{"function": "http", "args": {"method": "GET", "url": "https://example.com/"}}
```
"""

m = markdowner.convert(tool_call)

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

print(parse_html(m))