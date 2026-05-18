# LLM Orchestration & Prompt Influence Project

This repository is based on the open-source GPT4All project.

## About
This project extends GPT4All with an orchestration layer in order to study
how system prompts and developer prompts affect LLM outputs.

## Our Extensions
- The project implementation is located in `our_project_tests`
- **When running for the first time, the model’s foundation weights will be downloaded automatically (3-4 GB).**
- **results of each test are in `results` directory saved as <name_of_the_test.jsonl>.**
- Evaluation experiments based on the HumanEval question set.
- the benchmark is done by `par_evaluation.py`

## running instructions
- Insure Python script in run_eval.slurm is the path to your file name
- bash : sbatch slurm_running_files/<name_of_file>.slurm
- bash : evaluate_functional_correctness <name_of_file>.jsonl to get the Correctness Analysis
- bash : python par_evaluation.py results/<result_file#1>.jsonl results/<result_file#2>.jsonl to get result comaparison according to PAR Metric
- for more options see :  evaluate_functional_correctness --help



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

