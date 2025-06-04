import re

def reward(completion, solution):
    # Reward = 1.0 if the completion matches the <think>...</think><answer>...</answer> format
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.fullmatch(pattern, completion, re.DOTALL) else 0.0
