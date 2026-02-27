def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
    Examples
    split_words("Hello world!") ➞ ["Hello", "world!"]
    split_words("Hello,world!") ➞ ["Hello", "world!"]
    split_words("abcdef") == 3 
    '''
    # Check if there are any whitespaces
    if ' ' in txt:
        return txt.split(' ')
    
    # If no whitespaces, check for commas
    if ',' in txt:
        return txt.split(',')
    
    # If neither whitespaces nor commas, count lowercase letters with odd order
    count = 0
    for char in txt:
        if char.islower():
            order = ord(char) - ord('a')
            if order % 2 == 1:  # odd order
                count += 1
    return count

# Test cases
print("Test 1:", split_words("Hello world!"))
print("Test 2:", split_words("Hello,world!"))
print("Test 3:", split_words("abcdef"))

# Additional verification for the third case
# 'a'=0 (even), 'b'=1 (odd), 'c'=2 (even), 'd'=3 (odd), 'e'=4 (even), 'f'=5 (odd)
# So we expect 3 odd-order letters: b, d, f