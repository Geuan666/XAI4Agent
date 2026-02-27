def is_nested(string):
    '''
    Create a function that takes a string as input which contains only square brackets.
    The function should return True if and only if there is a valid subsequence of brackets
    where at least one bracket in the subsequence is nested.

    is_nested('[[]]') ➞ True
    is_nested('[]]]]]]][[[[[]') ➞ False
    is_nested('[][]') ➞ False
    is_nested('[]') ➞ False
    is_nested('[[][]]') ➞ True
    is_nested('[[]][[') ➞ True
    '''
    # Track the maximum nesting depth reached
    depth = 0
    max_depth = 0
    valid = True

    for char in string:
        if char == '[':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ']':
            depth -= 1
            if depth < 0:
                valid = False
                break

    # Valid nested structure exists if:
    # 1. The string is valid (no negative depth)
    # 2. We reached a nesting level of at least 2 (meaning we had nested brackets)
    return valid and max_depth >= 2
