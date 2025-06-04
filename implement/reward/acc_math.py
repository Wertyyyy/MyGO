from math_verify import parse, verify
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reward(completion, solution):
    try:
        ans = parse(completion)
        return float(verify(ans, parse(solution)))
    except Exception as e:
        logger.error(f"Error in reward function {__name__}: {e}")
        return 0.0

if __name__ == "__main__":
    print(reward("<think> Jill watched a show that was 30 minutes long, and then watched another show that was 4 times longer. So the length of the second show is 4 * 30 = 120 minutes. The total time she spent watching shows is 30 + 120 = 150 minutes. </think><answer>150 minutes</answer>", "150"))