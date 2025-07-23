from nexora.models.base_model import BaseModel
from nexora.logger import logger
from nexora.exceptions import ModelReturnError

from nexora.messages.assistant import Assistant
from nexora.messages.tool_call import ToolCall
from nexora.messages.tool_return import ToolReturn
from nexora.messages.response import Response

from .system_prompt import SYSTEM_PROMPT

from litellm import completion

import json

class Ollama(BaseModel):
    provider = "ollama"
    model = None

    has_tools = True
    has_vision = True

    api_base="http://localhost:11434"

    def __init__(self):
        super().__init__()

    # @property
    # def system_prompt_template(self):
    #     tool_prompt = ""

    #     for tool in self.tools.__obj__:
    #         name = tool["function"]["name"]
    #         description = tool["function"]["description"]
    #         parameters = json.dumps(tool["function"]["parameters"])

    #         tool_prompt += f"**{name}**{description}\nJSON Schema: {parameters}\n"

    #     tool_prompt = tool_prompt.replace("{", "{{")
    #     tool_prompt = tool_prompt.replace("}", "}}")

    #     return SYSTEM_PROMPT + "\n\n" + tool_prompt

    def generate_response(self):
        serialized_messages = [message.serialize() for message in self.messages]

        try:
            response = completion(
                model=f"ollama_chat/{self.model}",
                messages=serialized_messages,
                api_base=self.api_base,
                tools=self.tools.__obj__
            )

            logger.debug(response)
        except Exception as e:
            logger.error(f"Error while making request to Ollama: {e}")
            raise ModelReturnError(f"Error making request to Ollama: {e}")
        
        choice = response.choices[0]
        
        content = choice.message["content"]
        finish_reason = choice.finish_reason

        # # Parse tool calls from the response
        # try:
        #     tool_calls = tool_parser(content)
        # except Exception as e:
        #     logger.error(f"Error while parsing for potential tool calls {e}")
        #     logger.debug(content)
        #     raise ModelReturnError(f"Error while parsing for potential tool calls: {e}")

        tool_calls = []

        if finish_reason == "tool_calls":
            tool_calls_data = response.choices[0].message.tool_calls

            tool_calls = [
                ToolCall(content=data.function, id=data.id, type=data.type) for data in tool_calls_data
            ]

            msg = Assistant(content)
            msg.tool_calls = tool_calls

            self.add_message(msg)
        else:
            self.add_message(Assistant(content))
            
        return Response(content=content, model=self.model, tool_calls=tool_calls, finish_reason=finish_reason)
