# LLM Orchestration & Prompt Influence Project

This repository is based on the open-source GPT4All project.

## About
This project extends GPT4All with an orchestration layer in order to study
how system prompts and developer prompts affect LLM outputs.

## Our Extensions
- The project implementation is located in `assistant.py`
- **When running `assistant.py`, the model’s foundation weights will be downloaded automatically (3-4 GB).**
- Controlled system & developer prompt injection
- Prompt-sensitivity experiments
- **Conversation documentation stored as JSON files in the `session/` directory**, including full user–assistant interaction histories for analysis and evaluation
- Evaluation experiments based on the HumanEval question set

## HumanEval Question Set (Third-Party)
This project uses evaluation questions from the open-source **HumanEval** repository:
- Repository: https://github.com/openai/human-eval

All rights to the original HumanEval tasks and questions belong to their
respective authors. The HumanEval content is used in this project in
accordance with its original license.

The original HumanEval LICENSE file is preserved in this repository, and all
HumanEval-related files retain their original copyright and licensing terms.

This repository does not claim ownership over the HumanEval questions.
The project’s contribution includes the orchestration layer, prompt
injection setup, experiment pipeline, and analysis of model-generated
outputs produced when running the evaluation.

## Based on GPT4All
Original project:
- Name: GPT4All
- Repository: https://github.com/nomic-ai/gpt4all
- License: Apache 2.0

The original README files and LICENSE of GPT4All are preserved in this repository.

## License
This project follows the license of the original GPT4All project.

