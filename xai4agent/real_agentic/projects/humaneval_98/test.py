def count_upper(s):
    """
    Given a string s, count the number of uppercase vowels in even indices.
    
    For example:
    count_upper('aBCdEf') returns 1
    count_upper('abcdefg') returns 0
    count_upper('dBBE') returns 0
    """
    uppercase_vowels = set('AEIOU')
    count = 0
    for i in range(0, len(s), 2):  # step by 2 to get even indices
        if s[i] in uppercase_vowels:
            count += 1
    return count

# Test cases
print(count_upper('aBCdEf'))   # Should return 1
print(count_upper('abcdefg'))  # Should return 0
print(count_upper('dBBE'))     # Should return 0