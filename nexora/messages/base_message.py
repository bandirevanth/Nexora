class BaseMessage:
    def __init__(self, content: str):
        self.content = content

    def serialize(self) -> dict:
        raise NotImplementedError("This method should be implemented by subclasses.")
