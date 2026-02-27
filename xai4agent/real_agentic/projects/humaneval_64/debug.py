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
    print(f"Input string: '{s}'")
    
    # Define regular vowels
    vowels = set('aeiou')
    print(f"Regular vowels: {vowels}")
    
    # Count regular vowels
    count = sum(1 for char in s if char in vowels)
    print(f"Regular vowels count: {count}")
    
    # Check if 'y' is at the end of the word
    ends_with_y = s.endswith('y')
    print(f"Ends with 'y': {ends_with_y}")
    
    if ends_with_y:
        count += 1
        print("Added 1 for ending 'y'")
    
    print(f"Total count: {count}")
    return count

# Test
print("Testing 'xyz':")
result = vowels_count("xyz")
print(f"Result: {result}\n")

print("Testing 'happy':")
result = vowels_count("happy")
print(f"Result: {result}\n")

print("Testing 'gym':")
result = vowels_count("gym")
print(f"Result: {result}\n")