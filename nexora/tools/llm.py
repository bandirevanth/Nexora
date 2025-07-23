from .basetool import BaseTool

class LLM(BaseTool):
    name = "llm"

    def switch_model(self, model:
        """openai.function: Allows the LLM to switch LLMs. Valid models: gpt-4-0125-preview, gpt-3.5-turbo-0125

        model

        :param str model: The model name to switch to (e.g., gpt-4-0125-preview, gpt-3.5-turbo-0125).
        """
        if model not in ['gpt-4-0125-preview', 'gpt-3.5-turbo-0125']:
            return {"error": "Invalid model name provided."}
            
        # This is where the actual switching logic would go, simulated as a placeholder.
        return {"success": True, "message": f"Switched to model {model} successfully."}
