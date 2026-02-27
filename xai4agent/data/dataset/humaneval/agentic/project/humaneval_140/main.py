def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores,
    and if a string has more than 2 consecutive spaces,
    then replace all consecutive spaces with -

    fix_spaces("Example") == "Example"
    fix_spaces("Example 1") == "Example_1"
    fix_spaces(" Example 2") == "_Example_2"
    fix_spaces(" Example   3") == "_Example-3"
    """
    import re
    # 先替换3个或以上连续空格为-
    result = re.sub(r' {3,}', '-', text)
    # 再替换剩余的单个空格为_
    result = result.replace(' ', '_')
    return result
