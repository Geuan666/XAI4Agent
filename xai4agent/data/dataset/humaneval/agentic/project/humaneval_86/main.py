def anti_shuffle(s):
    """
    Write a function that takes a string and returns an ordered version of it.
    Ordered version of string, is a string where all words (separated by space)
    are replaced by a new word where all the characters arranged in
    ascending order based on ascii value.
    Note: You should keep the order of words and blank spaces in the sentence.

    For example:
    anti_shuffle('Hi') returns 'Hi'
    anti_shuffle('hello') returns 'ehllo'
    anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'
    """
    if not s:
        return s

    # 分割字符串为单词，保持空格位置
    words = s.split(' ')

    # 对每个单词内的字符进行排序
    sorted_words = []
    for word in words:
        # 将单词中的字符排序
        sorted_chars = sorted(word)
        # 重新组合成字符串
        sorted_word = ''.join(sorted_chars)
        sorted_words.append(sorted_word)

    # 用空格连接所有单词
    return ' '.join(sorted_words)
