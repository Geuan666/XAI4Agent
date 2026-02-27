from typing import List


def rescale_to_unit(numbers: List[float]) -> List[float]:
    """ Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if not numbers:
        return []

    min_val = min(numbers)
    max_val = max(numbers)

    # 如果所有数都相同，则返回全0列表
    if min_val == max_val:
        return [0.0] * len(numbers)

    # 应用线性变换公式：(x - min) / (max - min)
    return [(x - min_val) / (max_val - min_val) for x in numbers]
