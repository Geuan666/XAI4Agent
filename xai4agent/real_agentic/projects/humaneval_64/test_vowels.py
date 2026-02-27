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

# Test the provided examples
print("Testing provided examples:")
print(f'vowels_count("abcde") = {vowels_count("abcde")}')  # Expected: 2
print(f'vowels_count("ACEDY") = {vowels_count("ACEDY")}')  # Expected: 3

# Test additional cases
print("\nTesting additional cases:")
print(f'vowels_count("") = {vowels_count("")}')  # Expected: 0
print(f'vowels_count("xyz") = {vowels_count("xyz")}')  # Expected: 1 (y at end)
print(f'vowels_count("happy") = {vowels_count("happy")}')  # Expected: 2 (a, y at end)
print(f'vowels_count("gym") = {vowels_count("gym")}')  # Expected: 1 (y at end)
print(f'vowels_count("yellow") = {vowels_count("yellow")}')  # Expected: 2 (e, o, no y at end)
print(f'vowels_count("rhythm") = {vowels_count("rhythm")}')  # Expected: 0 (y not at end)
print(f'vowels_count("fly") = {vowels_count("fly")}')  # Expected: 1 (y at end)
print(f'vowels_count("gry") = {vowels_count("gry")}')  # Expected: 1 (y at end)