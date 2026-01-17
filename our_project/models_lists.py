from gpt4all import GPT4All
import pandas as pd

def list_gpt4all_models():
    # Fetch the list of models using the native static method
    models = GPT4All.list_models()
    
    # Optional: Convert to DataFrame for cleaner viewing in a notebook/terminal
    df = pd.DataFrame(models)
    
    # Filter for relevant columns to avoid clutter
    display_cols = ['name', 'filename', 'parameters', 'quant', 'ramrequired', 'description']
    return df[display_cols]

# Execution
if __name__ == "__main__":
    try:
        df_models = list_gpt4all_models()
        print(f"Found {len(df_models)} models:")
        print(df_models.to_markdown(index=False)) # Markdown table for readability
    except Exception as e:
        print(f"Error fetching models: {e}")