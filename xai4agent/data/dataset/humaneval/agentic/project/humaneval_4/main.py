from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    if not numbers:
        return 0.0

    # 计算均值
    mean = sum(numbers) / len(numbers)

    # 计算每个元素与均值的绝对差
    absolute_deviations = [abs(x - mean) for x in numbers]

    # 返回平均绝对偏差
    return sum(absolute_deviations) / len(absolute_deviations)
