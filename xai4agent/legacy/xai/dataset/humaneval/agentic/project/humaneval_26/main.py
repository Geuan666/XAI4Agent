from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
    # 统计每个数字的出现次数
    count = {}
    for num in numbers:
        count[num] = count.get(num, 0) + 1

    # 保留只出现一次的数字，保持原有顺序
    result = []
    for num in numbers:
        if count[num] == 1:
            result.append(num)

    return result
