def histogram(test):
    """Given a string representing a space separated lowercase letters, return a dictionary
    of the letter with the most repetition and containing the corresponding count.
    If several letters have the same occurrence, return all of them.

    Example:
    histogram('a b c') == {'a': 1, 'b': 1, 'c': 1}
    histogram('a b b a') == {'a': 2, 'b': 2}
    histogram('a b c a b') == {'a': 2, 'b': 2}
    histogram('b b b b a') == {'b': 4}
    histogram('') == {}

    """
    # 处理空字符串的情况
    if not test:
        return {}

    # 分割字符串得到字母列表
    letters = test.split()

    # 统计每个字母的出现次数
    count_dict = {}
    for letter in letters:
        count_dict[letter] = count_dict.get(letter, 0) + 1

    # 找到最大出现次数
    max_count = max(count_dict.values())

    # 返回具有最大出现次数的所有字母
    return {letter: count for letter, count in count_dict.items() if count == max_count}
