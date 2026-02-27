from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # 移除所有空格
    paren_string = paren_string.replace(' ', '')

    result = []
    current_group = ""
    depth = 0

    for char in paren_string:
        if char == '(':
            current_group += char
            depth += 1
        elif char == ')':
            current_group += char
            depth -= 1

            # 当深度回到0时，表示找到了一个完整的组
            if depth == 0:
                result.append(current_group)
                current_group = ""

    return result
