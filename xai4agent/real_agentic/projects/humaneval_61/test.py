def correct_bracketing(brackets: str):
    """ brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
    count = 0
    for bracket in brackets:
        if bracket == '(':
            count += 1
        elif bracket == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0

# Test the examples
print("Testing correct_bracketing function:")
print(f'correct_bracketing("(") = {correct_bracketing("(")}')  # Should be False
print(f'correct_bracketing("()") = {correct_bracketing("()")}');  # Should be True
print(f'correct_bracketing("(()())") = {correct_bracketing("(()())")}');  # Should be True
print(f'correct_bracketing(")(()") = {correct_bracketing(")(()")}');  # Should be False