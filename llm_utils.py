# llm_utils.py

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
