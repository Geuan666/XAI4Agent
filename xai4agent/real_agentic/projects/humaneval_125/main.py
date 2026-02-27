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