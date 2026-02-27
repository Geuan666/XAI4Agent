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