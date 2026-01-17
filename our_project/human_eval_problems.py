from human_eval.data import write_jsonl, read_problems

problems = read_problems()

idx_list = list(problems.keys())

print(problems[idx_list[0]]["prompt"]) 
print(problems[idx_list[-1]]["prompt"]) 