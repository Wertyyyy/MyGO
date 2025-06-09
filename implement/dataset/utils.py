def load_prompt(prompt_path: str):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()