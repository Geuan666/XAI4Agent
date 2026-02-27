def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.

    For example:
    >>> is_bored("Hello world")
    0
    >>> is_bored("The sky is blue. The sun is shining. I love this weather")
    1
    """
    if not S:
        return 0

    # 分割句子
    import re
    sentences = re.split(r'[.?!]+', S)

    # 统计以"I"开头的句子数量
    count = 0
    for sentence in sentences:
        # 去除前后空白字符
        sentence = sentence.strip()
        # 检查是否以"I"开头且后面是空格或结尾（完整单词）
        if sentence and sentence.startswith('I') and (len(sentence) == 1 or sentence[1].isspace()):
            count += 1

    return count
