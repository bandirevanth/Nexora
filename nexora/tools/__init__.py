from nexora.tools.fileio import FileIO
from nexora.tools.http import HTTP
from nexora.tools.shell import Shell
from nexora.tools.image import ImageTool

class Tools:
    def __init__(self, ToolReturn):
        self.ToolReturn = ToolReturn
        self._tools = ("Shell", "FileIO", "HTTP", "ImageTool")
        # self._tools = [Shell, FileIO, HTTP, ImageTool]

    @property
    def tools(self):
        tools = []

        for tool in self._tools:
            tools.append(globals()[tool])

        return tools
    
    @tools.setter
    def tools(self, tools):
        self._tools = tools

    def call(self, name, call_id, app, kwargs):
        try:
            tool_name, function_name = name.split("_")
        except ValueError:
            raise ValueError(f"Unable to unpack method {name}")

        for tool in self.tools:
            if tool.name == tool_name and hasattr(tool, function_name):
                # Create tool context
                tool_cxt = tool()

                tool_cxt.app = app

                # Call the function and return the result
                try:
                    kwargs_f = ("%s=%r" % (k, v) for k, v in kwargs.items())
                    kwargs_f = ", ".join(kwargs_f)

                    print(f"\033[92m* {name}({kwargs_f})\033[0m")
                    
                    content = getattr(tool_cxt, function_name)(**kwargs)

                    tool_return = self.ToolReturn(
                        content=content,
                        id=call_id,
                        name=name,
                    )
                except Exception as e:
                    # Log and return error message to the LLM
                    print(f"func returned error: {e}")
                    err = f"Tool call failed with the following error: {e}"
                    
                    tool_return = self.ToolReturn(
                        content=err,
                        id=call_id,
                        name=name,
                        error=True
                    )
                
                return tool_return

        raise NotImplementedError(f"Couldn't find the method {name}")
    
    @property
    def __obj__(self):
        l = []
        for tool in self.tools:
            tool = tool()
            l += tool.__func__

        return l

    @property
    def list_methods(self):
        methods = []

        for Tool in self.tools:
            for attr in dir(Tool):
                val = getattr(Tool, attr)

                if not callable(val):
                    continue

                if not val.__doc__:
                    continue

                if not val.__doc__.startswith("openai.function: "):
                    continue

                methods.append((Tool, val))

        return methods
