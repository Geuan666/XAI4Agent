def reverse_delete(s,c):
    """Task
    We are given two strings s and c, you have to deleted all the characters in s that are equal to any character in c
    then check if the result string is palindrome.
    A string is called palindrome if it reads the same backward as forward.
    You should return a tuple containing the result string and True/False for the check.
    Example
    For s = "abcde", c = "ae", the result should be ('bcd',False)
    For s = "abcdef", c = "b"  the result should be ('acdef',False)
    For s = "abcdedcba", c = "ab", the result should be ('cdedc',True)
    """
    # Remove all characters in s that are present in c
    result = ''.join(char for char in s if char not in c)
    
    # Check if the result is a palindrome
    is_palindrome = result == result[::-1]
    
    return (result, is_palindrome)

# Test cases
print("Test 1:", reverse_delete("abcde", "ae"))  # Expected: ('bcd', False)
print("Test 2:", reverse_delete("abcdef", "b"))  # Expected: ('acdef', False)
print("Test 3:", reverse_delete("abcdedcba", "ab"))  # Expected: ('cdedc', True)