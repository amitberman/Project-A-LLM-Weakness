# LLM Orchestration & Prompt Influence Project

This repository is based on the open-source GPT4All project.

## About
This project extends GPT4All with an orchestration layer in order to study
how system prompts and developer prompts affect LLM outputs.

## Based on GPT4All
Original project:
- Name: GPT4All
- Repository: https://github.com/nomic-ai/gpt4all
- License: Apache 2.0

The original README files are preserved in this repository.

## Our Extensions
- The project implementation is located in `assistant.py`
- **When running `assistant.py`, the model’s foundation weights will be downloaded automatically (3-4 GB).**
- Controlled system & developer prompt injection
- Prompt-sensitivity experiments
- Logging and evaluation pipeline
- **Conversation documentation stored as JSON files in the `session/` directory**, including full user–assistant interaction histories for analysis and evaluation



## License
This project follows the license of the original GPT4All project.
