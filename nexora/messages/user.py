from nexora.messages.base_message import BaseMessage

class User(BaseMessage):
    def serialize(self) -> dict:
        return {'role': 'user', 'content': self.content}
