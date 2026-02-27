from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """
    # 创建数字单词到数值的映射
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }

    # 如果输入为空，直接返回空字符串
    if not numbers:
        return ''

    # 将输入分割成单词列表
    words = numbers.split()

    # 将单词转换为对应的数值
    nums = [word_to_num[word] for word in words]

    # 对数值进行排序
    nums.sort()

    # 将排序后的数值转换回单词
    num_to_word = {v: k for k, v in word_to_num.items()}
    sorted_words = [num_to_word[num] for num in nums]

    # 重新组合成字符串
    return ' '.join(sorted_words)
