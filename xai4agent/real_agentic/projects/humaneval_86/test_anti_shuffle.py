def anti_shuffle(s):
    """
    Write a function that takes a string and returns an ordered version of it.
    Ordered version of string, is a string where all words (separated by space)
    are replaced by a new word where all the characters arranged in
    ascending order based on ascii value.
    Note: You should keep the order of words and blank spaces in the sentence.

    For example:
    anti_shuffle('Hi') returns 'Hi'
    anti_shuffle('hello') returns 'ehllo'
    anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'
    """
    words = s.split(' ')
    result = []
    for word in words:
        # Sort characters in the word by ASCII value
        sorted_word = ''.join(sorted(word))
        result.append(sorted_word)
    return ' '.join(result)

# Test the examples from the docstring
print("Testing anti_shuffle function:")
print(f"anti_shuffle('Hi') = '{anti_shuffle('Hi')}'")
print(f"anti_shuffle('hello') = '{anti_shuffle('hello')}'")
print(f"anti_shuffle('Hello World!!!') = '{anti_shuffle('Hello World!!!')}'")

# Additional tests
print(f"anti_shuffle('') = '{anti_shuffle('')}'")
print(f"anti_shuffle('a') = '{anti_shuffle('a')}'")
print(f"anti_shuffle('Hello   World') = '{anti_shuffle('Hello   World')}'")