from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
    # Count occurrences of each number
    count = {}
    for num in numbers:
        count[num] = count.get(num, 0) + 1
    
    # Filter out numbers that occur more than once, preserving order
    result = []
    for num in numbers:
        if count[num] == 1:
            result.append(num)
    
    return result

# Test the function
if __name__ == "__main__":
    test_input = [1, 2, 3, 2, 4]
    result = remove_duplicates(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {result}")
    print(f"Expected: [1, 3, 4]")
    print(f"Match: {result == [1, 3, 4]}")