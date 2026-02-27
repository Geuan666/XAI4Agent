def get_closest_vowel(word):
    """You are given a word. Your task is to find the closest vowel that stands between
    two consonants from the right side of the word (case sensitive).

    Vowels in the beginning and ending doesn't count. Return empty string if you didn't
    find any vowel met the above condition.

    You may assume that the given string contains English letter only.

    Example:
    get_closest_vowel("yogurt") ==> "u"
    get_closest_vowel("FULL") ==> "U"
    get_closest_vowel("quick") ==> ""
    get_closest_vowel("ab") ==> ""
    """
    vowels = "aeiouAEIOU"

    # 从右往左遍历，跳过第一个和最后一个字符
    for i in range(len(word) - 2, 0, -1):
        # 检查当前位置是否是元音
        if word[i] in vowels:
            # 检查该元音前后是否都是辅音
            if word[i-1] not in vowels and word[i+1] not in vowels:
                return word[i]

    return ""
