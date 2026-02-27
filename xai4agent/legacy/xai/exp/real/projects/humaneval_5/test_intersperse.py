from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if len(numbers) <= 1:
        return numbers[:]
    
    result = []
    for i, num in enumerate(numbers):
        result.append(num)
        # Add delimiter after each element except the last one
        if i < len(numbers) - 1:
            result.append(delimeter)
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1: empty list
    result1 = intersperse([], 4)
    print(f"intersperse([], 4) = {result1}")
    assert result1 == []
    
    # Test case 2: normal case
    result2 = intersperse([1, 2, 3], 4)
    print(f"intersperse([1, 2, 3], 4) = {result2}")
    assert result2 == [1, 4, 2, 4, 3]
    
    # Additional test cases
    # Single element
    result3 = intersperse([5], 0)
    print(f"intersperse([5], 0) = {result3}")
    assert result3 == [5]
    
    # Two elements
    result4 = intersperse([1, 2], 9)
    print(f"intersperse([1, 2], 9) = {result4}")
    assert result4 == [1, 9, 2]
    
    print("All tests passed!")