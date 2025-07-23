from nexora.models.openai.base_openai import BaseOpenAI

class GPT4o(BaseOpenAI):
    has_vision = True
    model = "gpt-4o"