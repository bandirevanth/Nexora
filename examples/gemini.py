from litellm import completion

# Ignore this silly way of getting the API key - it's just for this example so I'm not hard-coding a sensitive API key in this code 
with open("/tmp/gemini.txt", "r") as f:
    api_key = f.read()

# help(completion)

response = completion(
    model="gemini/gemini-pro", 
    messages=[{"role": "user", "content": "Howdy"}],
    api_key=api_key,
    tools=[]
)

print(response)

