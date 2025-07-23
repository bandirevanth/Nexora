from nexora.models.openai.base_openai import BaseOpenAI

class GPT4(BaseOpenAI):
    has_vision = True
    model = "gpt-4-turbo"