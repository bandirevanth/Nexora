from nexora.messages.base_message import BaseMessage

class Response(BaseMessage):
    def __init__(self, content: str, model: str, tool_calls: list = None, finish_reason = None):
        super().__init__(content)
        self.model = model
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason

    def serialize(self) -> dict:
        return {
            'content': self.content,
            'model': self.model,
            'tool_calls': [call.serialize() for call in self.tool_calls]
        }
