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
    # We need to find if there's a valid subsequence with nested brackets
    # A nested bracket structure means we have at least one pair of brackets inside another pair
    
    # Let's track the nesting level as we go through the string
    depth = 0
    max_depth = 0
    
    for char in string:
        if char == '[':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ']':
            depth -= 1
            # If we go negative, it's invalid
            if depth < 0:
                return False
    
    # If we never went deeper than 1, there's no nesting
    # But we also need to make sure the brackets are balanced
    if depth != 0:
        return False
        
    # If we reached a maximum depth > 1, then we had nesting
    return max_depth > 1