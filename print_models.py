from gpt4all import GPT4All
import json

# Fetch the list of all available models from Nomic's official registry
models = GPT4All.list_models()
for model in models:
    print(f"Name: {model['name']} | Filename: {model['filename']}")