def match_parens(lst):
    '''
    You are given a list of two strings, both strings consist of open
    parentheses '(' or close parentheses ')' only.
    Your job is to check if it is possible to concatenate the two strings in
    some order, that the resulting string will be good.
    A string S is considered to be good if and only if all parentheses in S
    are balanced. For example: the string '(())()' is good, while the string
    '())' is not.
    Return 'Yes' if there's a way to make a good string, and return 'No' otherwise.

    Examples:
    match_parens(['()(', ')']) == 'Yes'
    match_parens([')', ')']) == 'No'
    '''
    def is_balanced(s):
        """Check if a parentheses string is balanced"""
        balance = 0
        for char in s:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            # If at any point we have more closing than opening, it's invalid
            if balance < 0:
                return False
        # All parentheses should be matched
        return balance == 0

    # Try both possible concatenations
    s1 = lst[0] + lst[1]
    s2 = lst[1] + lst[0]

    if is_balanced(s1) or is_balanced(s2):
        return 'Yes'
    else:
        return 'No'
