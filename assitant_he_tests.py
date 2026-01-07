from gpt4all import GPT4All
from human_eval.data import write_jsonl, read_problems
import tqdm 

MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_0.gguf"

print("Loading model...")

model = GPT4All(MODEL_NAME, device="cpu") 
print("Model loaded.")

def generate_one_completion(prompt: str) -> str:
    """
    Generates completion for a coding task.
    """
    
    response = model.generate(
        prompt, 
        max_tokens=512, 
        temp=0.1  
    )
    return response

def run_evaluation():
    samples = []
    problems = read_problems()

    for task_id in tqdm.tqdm(problems):
        prompt = problems[task_id]["prompt"]
        
        completion = generate_one_completion(prompt)
        
        samples.append(dict(task_id=task_id, completion=completion))

    write_jsonl("samples.jsonl", samples)


if __name__ == "__main__":
    run_evaluation()
