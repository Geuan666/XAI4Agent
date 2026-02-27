"""
Add more test cases.
"""

def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word.

    Example:
    >>> vowels_count("abcde")
    2
    >>> vowels_count("ACEDY")
    3
    """
    if not s:
        return 0
    
    # Convert to lowercase for case insensitivity
    s = s.lower()
    
    # Define regular vowels
    vowels = set('aeiou')
    
    # Count regular vowels
    count = sum(1 for char in s if char in vowels)
    
    # Check if 'y' is at the end of the word
    if s.endswith('y'):
        count += 1
    
    return count