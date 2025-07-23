# Version Goals

# Next
- Model 'gpt-3.5-turbo' for provider 'openai' not found
- Catch and handle Anthropic anthropic.InternalServerError cleanly


## 0.15
- [ ] Save sessions in ~/.config/nexora, associate them with current directory (absolute path), and prompt for recalling previous sessions upon launch. Show a list of old sessions and the option to either recall one or to proceed a fresh session
- [ ] Better CLI elements
    - [ ] BUG FIX: Not accepting multi-line inputs at prompt, EOFs too soon and submits to AI
    - [ ] Spinning animation while LLM is generating
    - [ ] Ctrl+C during generation should interrupt the generation properly, returning you to a prompt
    - [ ] Up/down arrows should give you prompt history
    - [ ] Better permission prompt
        - [ ] Ability to deny permission with REASON, so you can explain to the LLM why you didn't give it the ability to do that action
        - [ ] Better styling, cleaner, no weird colours. Just RED background and white text

## 0.16
- [ ] Vision processing for supported models
- [ ] Google (and Google Gemini) model support
    - [ ] `google` will be the provider, three models will inherit it: `gemini-1.0-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`
    - [ ] See other models' implementations to see how it should be implemented. Specifically, see how Ollama is implemented since it uses LiteLLM, and we should use LiteLLM to implement Google. See example in 'examples/gemini.py' for an example

## 0.17
- [ ] Token Optimizations; 
    - [ ] Automatically compress and reduce redundant lines in scrollback to reduce input tokens
    - [ ] Use the current LLM to compress conversations when hard token limits are reached

# Unsorted Goals
- [ ] Better code structure, more organized
- [ ] Better handling of token overflows
- [ ] Two --verbose levels, for better control of debugging
- [ ] Consider migrating to using litellm for ALL LLM-calling, reducing code complexity and increasing reusability