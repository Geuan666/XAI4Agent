from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Remove all spaces from the input string
    paren_string = paren_string.replace(' ', '')
    
    result = []
    start = 0
    balance = 0
    current_group = ""
    
    for i, char in enumerate(paren_string):
        if char == '(':
            if balance == 0:
                # Start of a new group
                start = i
            balance += 1
        elif char == ')':
            balance -= 1
            if balance == 0:
                # End of a group
                result.append(paren_string[start:i+1])
    
    return result