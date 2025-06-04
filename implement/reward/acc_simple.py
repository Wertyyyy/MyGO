import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reward(completion, solution):
    model_answer = re.findall("(\\-?[0-9\\.\\,]+)", completion)
    if len(model_answer) == 0:
        return 0.0
    else:
        invalid_str = ["", "."]
        # find the last number that is not '.'
        for final_answer in reversed(model_answer):
            if final_answer not in invalid_str:
                break
        if final_answer == solution:
            return 1.0
        else:
            return 0.0

if __name__ == "__main__":
    print(reward("Jill watched a show that was 30 minutes long, and then watched another show that was 4 times longer. So the length of the second show is 4 * 30 = 120 minutes. The total time she spent watching shows is 30 + 120 = 150 minutes.", "150"))