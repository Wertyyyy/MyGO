import re
import string


def reward(completion, solution):
    """
    Reward function for multiple choice questions.

    Args:
        completion (str): The model's completion/response
        solution (str): The correct answer (should be A, B, C, or D)

    Returns:
        float: 1.0 if the completion contains the correct answer, 0.0 otherwise
    """
    # Normalize the solution to ensure it's a single letter
    if not solution or not isinstance(solution, str):
        return 0.0

    solution = solution.strip().upper()
    if solution not in string.ascii_uppercase:
        return 0.0

    # Clean the completion
    if not completion or not isinstance(completion, str):
        return 0.0

    completion = completion.strip()

    # Strategy 1: Check for exact answer in <answer> tags
    answer_match = re.search(
        r"<answer>\s*([A-D])\s*</answer>", completion, re.IGNORECASE
    )
    if answer_match:
        predicted_answer = answer_match.group(1).upper()
        return 1.0 if predicted_answer == solution else 0.0

    # Strategy 2: Look for the answer pattern at the end of completion
    # Common patterns: "Answer: A", "The answer is B", "选择 C", etc.
    end_patterns = [
        r"(?:answer|答案|选择|选项)(?:\s*is|\s*:|：)?\s*([A-D])",
        r"([A-D])\s*(?:is\s+(?:the\s+)?(?:correct\s+)?answer|是(?:正确|答案))",
        r"(?:therefore|所以|因此|故),?\s*(?:the\s+answer\s+is\s+)?([A-D])",
        r"^([A-D])$",  # Just the letter alone
        r"([A-D])\s*$",  # Letter at the end
    ]

    for pattern in end_patterns:
        matches = re.findall(pattern, completion, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Take the last match (most likely to be the final answer)
            predicted_answer = matches[-1].upper()
            return 1.0 if predicted_answer == solution else 0.0

    # Strategy 3: Look for any occurrence of the correct letter
    # This is more lenient and might have false positives, but covers edge cases
    solution_pattern = rf"\b{solution}\b"
    if re.search(solution_pattern, completion.upper()):
        # Additional check: make sure it's not just part of the question text
        # Count all letter occurrences
        all_letters = re.findall(r"\b[A-D]\b", completion.upper())
        if all_letters:
            # If the solution letter appears more frequently than others,
            # or if it's the last mentioned letter, consider it correct
            last_letter = all_letters[-1]
            if last_letter == solution:
                return 1.0

            # Count frequency of each letter
            from collections import Counter

            letter_counts = Counter(all_letters)
            if letter_counts[solution] > 0:
                # Check if it's the most frequent or tied for most frequent
                max_count = max(letter_counts.values())
                if letter_counts[solution] == max_count:
                    # If tied, prefer the solution
                    return 1.0

    return 0.0


def extract_answer(completion):
    """
    Helper function to extract the predicted answer from completion.

    Args:
        completion (str): The model's completion/response

    Returns:
        str or None: The extracted answer letter (A, B, C, or D) or None if not found
    """
    if not completion or not isinstance(completion, str):
        return None

    completion = completion.strip()

    # Strategy 1: Check for answer in <answer> tags
    answer_match = re.search(
        r"<answer>\s*([A-D])\s*</answer>", completion, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).upper()

    # Strategy 2: Look for answer patterns
    end_patterns = [
        r"(?:answer|答案|选择|选项)(?:\s*is|\s*:|：)?\s*([A-D])",
        r"([A-D])\s*(?:is\s+(?:the\s+)?(?:correct\s+)?answer|是(?:正确|答案))",
        r"(?:therefore|所以|因此|故),?\s*(?:the\s+answer\s+is\s+)?([A-D])",
        r"([A-D])\s*$",  # Letter at the end
    ]

    for pattern in end_patterns:
        matches = re.findall(pattern, completion, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    # Strategy 3: Find all letters and return the last one
    all_letters = re.findall(r"\b[A-D]\b", completion.upper())
    if all_letters:
        return all_letters[-1]

    return None
