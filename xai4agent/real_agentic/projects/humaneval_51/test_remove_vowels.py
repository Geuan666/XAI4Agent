def remove_vowels(text):
    """
    remove_vowels is a function that takes string and returns string without vowels.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\nghijklm")
    'bcdf\nghjklm'
    >>> remove_vowels('abcdef')
    'bcdf'
    >>> remove_vowels('aaaaa')
    ''
    >>> remove_vowels('aaBAA')
    'B'
    >>> remove_vowels('zbcd')
    'zbcd'
    """
    vowels = "aeiouAEIOU"
    result = ""
    for char in text:
        if char not in vowels:
            result += char
    return result

# Test cases
print("Testing remove_vowels function:")
print(f"remove_vowels('') = '{remove_vowels('')}'")
print(f"remove_vowels('abcdef\\nghijklm') = '{remove_vowels('abcdef\\nghijklm')}'")
print(f"remove_vowels('abcdef') = '{remove_vowels('abcdef')}'")
print(f"remove_vowels('aaaaa') = '{remove_vowels('aaaaa')}'")
print(f"remove_vowels('aaBAA') = '{remove_vowels('aaBAA')}'")
print(f"remove_vowels('zbcd') = '{remove_vowels('zbcd')}'")