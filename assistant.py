from gpt4all import GPT4All
import json
from datetime import datetime
import shutil
MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_0.gguf"  # auto-downloads from GPT4All CDN

SYSTEM_PROMPT = """
You are a helpful, honest and concise AI assistant.
You run on a CPU-only laptop (no GPU).
If the user pastes code, help them understand, debug or improve it.
"""

def main():
    # Force CPU (you’ve already disabled CUDA with env vars, but this is explicit)
    model = GPT4All(MODEL_NAME, device="cpu")

    print(f"Loaded model: {MODEL_NAME}")
    print("Type 'exit' to quit.\n")
    
    conv_dict = {"time + date" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    # Keeps conversation context
    with model.chat_session(system_prompt=SYSTEM_PROMPT):
        i = 0
        while True:
            try:
                user_input = input("You: ").strip()
                conv_dict[f"user_input#{i}"] = user_input
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if user_input.lower() in {"exit", "quit"}:
                print("Bye 👋")
                break

            if not user_input:
                continue

            # 👉 This is where you “send” text/code to the LLM
            response = model.generate(
                user_input,
                max_tokens=512,
                temp=0.7,
            )
            conv_dict[f"assitant_response#{i}"] = response
            i+=1
            print(f"Assistant: {response}\n")
        chat_file = "session_"+ conv_dict["time + date"] + ".json"
        with open(chat_file , "w") as doc:
            json.dump(conv_dict , doc , indent= 4)
        shutil.move(chat_file, "session/" + chat_file)
if __name__ == "__main__":
    main()
