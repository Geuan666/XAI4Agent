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
        # Check if a string has balanced parentheses
        balance = 0
        for char in s:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            # If balance goes negative, it means we have more closing than opening so far
            if balance < 0:
                return False
        # Final balance should be zero for balanced parentheses
        return balance == 0
    
    s1, s2 = lst[0], lst[1]
    
    # Try both concatenations
    if is_balanced(s1 + s2) or is_balanced(s2 + s1):
        return 'Yes'
    else:
        return 'No'