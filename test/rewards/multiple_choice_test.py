import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service.rewards.multiple_choice import reward, extract_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic reward function functionality"""
    logger.info("Testing basic functionality...")
    
    # Test correct answers
    test_cases = [
        # (completion, solution, expected_reward)
        ("<answer>A</answer>", "A", 1.0),
        ("<answer> B </answer>", "B", 1.0),
        ("<answer> C</answer>", "C", 1.0),
        ("<answer>D</answer>", "D", 1.0),
        
        # Test incorrect answers
        ("<answer>A </answer>", "B", 0.0),
        ("<answer>B</answer>", "C", 0.0),
        ("<answer>C</answer>", "D", 0.0),
        ("<answer>D</answer>", "A", 0.0),
    ]
    
    for completion, solution, expected in test_cases:
        result = reward(completion, solution)
        assert result == expected, f"Failed for completion='{completion}', solution='{solution}', expected={expected}, got={result}"
        logger.info(f"✓ {completion} -> {solution}: {result}")
    
    logger.info("✓ Basic functionality test passed")


def test_answer_patterns():
    """Test various answer patterns"""
    logger.info("Testing answer patterns...")
    
    test_cases = [
        # English patterns
        ("The answer is A", "A", 1.0),
        ("Answer: B", "B", 1.0),
        ("Therefore, C", "C", 1.0),
        ("So the answer is D", "D", 1.0),
        ("Answer is A.", "A", 1.0),
        ("The correct answer is B", "B", 1.0),
        
        # Chinese patterns
        ("答案是A", "A", 1.0),
        ("选择A", "A", 1.0),
        ("答案：B", "B", 1.0),
        ("所以答案是C", "C", 1.0),
        
        # Just the letter
        ("A", "A", 1.0),
        ("B", "B", 1.0),
        
        # Letter at the end
        ("Based on the analysis, the answer should be C", "C", 1.0),
        ("After careful consideration, D", "D", 1.0),
        
        # Wrong answers
        ("The answer is A", "B", 0.0),
        ("Answer: C", "D", 0.0),
    ]
    
    for completion, solution, expected in test_cases:
        result = reward(completion, solution)
        assert result == expected, f"Failed for completion='{completion}', solution='{solution}', expected={expected}, got={result}"
        logger.info(f"✓ '{completion}' -> {solution}: {result}")
    
    logger.info("✓ Answer patterns test passed")


def test_complex_completions():
    """Test complex completions with reasoning"""
    logger.info("Testing complex completions...")
    
    test_cases = [
        # Long reasoning with correct answer
        ("""
        Let me analyze this question step by step.
        
        Looking at the options:
        A. This seems incorrect because...
        B. This could be right, but...
        C. This is clearly the correct answer because it matches the criteria.
        D. This is wrong because...
        
        <answer>C</answer>
        """, "C", 1.0),
        
        # Reasoning without tags
        ("""
        To solve this problem, I need to consider each option:
        
        Option A: Not suitable
        Option B: Partially correct but not the best
        Option C: This is wrong
        Option D: This matches all requirements perfectly
        
        Therefore, the answer is D.
        """, "D", 1.0),
        
        # Multiple letters mentioned, but clear final answer
        ("""
        Let's examine options A, B, C, and D.
        A is wrong. B has some merit. C is also incorrect.
        But D is clearly the best choice.
        
        Answer: D
        """, "D", 1.0),
        
        # Wrong final answer
        ("""
        Analyzing the options:
        A seems good, B is better, C is okay, D is poor.
        
        I choose A.
        """, "B", 0.0),
    ]
    
    for completion, solution, expected in test_cases:
        result = reward(completion, solution)
        assert result == expected, f"Failed for complex completion, solution='{solution}', expected={expected}, got={result}"
        logger.info(f"✓ Complex completion -> {solution}: {result}")
    
    logger.info("✓ Complex completions test passed")


def test_extract_answer_function():
    """Test the extract_answer helper function"""
    logger.info("Testing extract_answer function...")
    
    test_cases = [
        ("<answer>A</answer>", "A"),
        ("The answer is B", "B"),
        ("Answer: C", "C"),
        ("Therefore, D", "D"),
        ("Just A", "A"),
        ("No answer here", None),
        ("Multiple letters A B C but final answer is D", "D"),
        ("", None),
        (None, None),
    ]
    
    for completion, expected in test_cases:
        result = extract_answer(completion)
        assert result == expected, f"Failed for completion='{completion}', expected={expected}, got={result}"
        logger.info(f"✓ '{completion}' -> {result}")
    
    logger.info("✓ Extract answer function test passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    logger.info("Testing edge cases...")
    
    test_cases = [
        # Invalid inputs
        ("", "A", 0.0),
        (None, "A", 0.0),
        ("A", "", 0.0),
        ("A", None, 0.0),
        ("A", "E", 0.0),  # Invalid solution
        ("A", "1", 0.0),  # Invalid solution
        
        # Case sensitivity
        ("<answer>a</answer>", "A", 1.0),
        ("<answer>A</answer>", "a", 1.0),
        ("answer is a", "A", 1.0),
        
        # Whitespace
        ("<answer> A </answer>", "A", 1.0),
        ("  A  ", "A", 1.0),
        ("Answer:   B   ", "B", 1.0),
        
        # No clear answer
        ("This is a difficult question with no clear answer mentioned", "A", 0.0),
        ("I think all options are wrong", "B", 0.0),
    ]
    
    for completion, solution, expected in test_cases:
        result = reward(completion, solution)
        assert result == expected, f"Failed for completion='{completion}', solution='{solution}', expected={expected}, got={result}"
        logger.info(f"✓ Edge case: '{completion}' -> {solution}: {result}")
    
    logger.info("✓ Edge cases test passed")


def test_scienceqa_format():
    """Test with ScienceQA-style completions"""
    logger.info("Testing ScienceQA format...")
    
    test_cases = [
        # Typical ScienceQA response
        ("""
        To answer this question, I need to analyze each option:

        A. takin - This is an animal but not specifically adapted for climbing trees
        B. red-handed tamarin - This is a primate that lives in rainforests and is well adapted for climbing trees
        C. elephant - Large ground animals, not tree climbers
        D. whale - Marine animal, not related to trees

        Based on the analysis, red-handed tamarins are primates that spend most of their time in trees and have adaptations like long limbs and strong grip for climbing.

        <answer>B</answer>
        """, "B", 1.0),
        
        # Short answer
        ("""
        Looking at the image and the question about which animal is adapted for climbing trees.
        
        The answer is B.
        """, "B", 1.0),
        
        # Chinese response
        ("""
        根据题目，我需要分析哪个动物最适合爬树：
        
        A. 羚牛 - 地面动物
        B. 红手绢猴 - 灵长类，擅长爬树
        C. 大象 - 大型地面动物
        D. 鲸鱼 - 海洋动物
        
        答案是B
        """, "B", 1.0),
    ]
    
    for completion, solution, expected in test_cases:
        result = reward(completion, solution)
        assert result == expected, f"Failed for ScienceQA completion, solution='{solution}', expected={expected}, got={result}"
        logger.info(f"✓ ScienceQA format -> {solution}: {result}")
    
    logger.info("✓ ScienceQA format test passed")


def main():
    """Run all tests"""
    logger.info("Starting multiple choice reward function tests...")
    
    try:
        test_basic_functionality()
        test_answer_patterns()
        test_complex_completions()
        test_extract_answer_function()
        test_edge_cases()
        test_scienceqa_format()
        
        logger.info("🎉 All multiple choice reward function tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 