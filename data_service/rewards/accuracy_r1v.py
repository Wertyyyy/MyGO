from math_verify import parse, verify
import re


def reward(completion, solution):
    # Reward = 1.0 if the reasoning/answer matches the ground truth, else 0.0
    ans = parse(completion)
    if verify(ans, parse(solution)):
        return 1.0

    try:
        sol_match = re.search(r"<answer>(.*?)</answer>", solution)
        gt = sol_match.group(1).strip() if sol_match else solution.strip()
        cont_match = re.search(r"<answer>(.*?)</answer>", completion)
        student_ans = cont_match.group(1).strip() if cont_match else completion.strip()
        if student_ans == gt:
            return 1.0
        return 0.0
    except:
        return 0.0
