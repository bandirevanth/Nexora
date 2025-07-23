from nexora.messages.assistant import Assistant

class Assistant(Assistant):
    @property
    def finish_reason(self):
        if len(self.tool_calls) > 0:
            return "tool_calls"
        else:
            return "complete"
    
    def serialize(self) -> dict:
        if not self.content:
            content = "<blank message>"
        else:
            content = self.content

        content = [
            {"type": "text", "text": content}
        ]

        if len(self.tool_calls) > 0:

            for tool_call in self.tool_calls:

                content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.content.name,
                    "input": {}
                })
                
        
        obj = {
            'role': 'assistant', 
            'content': content
        }

        return obj