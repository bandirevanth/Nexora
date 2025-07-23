from nexora.messages.base_message import BaseMessage

class Assistant(BaseMessage):
    def __init__(self, content: str, tool_calls=None, finish_reason=None):
        super().__init__(content)

        if not tool_calls:
            tool_calls = []
            
        self.tool_calls = tool_calls
        self._finish_reason = finish_reason
        self.model = None

    @property
    def finish_reason(self):
        return self._finish_reason

    def serialize(self) -> dict:
        obj = {
            'role': 'assistant', 
            'content': self.content
        }

        if len(self.tool_calls) > 0:
            obj["tool_calls"] = [tool_call.serialize() for tool_call in self.tool_calls]

        return obj