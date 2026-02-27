def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False
    '''
    if not txt:
        return False

    # Check if the last character is an alphabetical character
    if not txt[-1].isalpha():
        return False

    # If string has only one character and it's alphabetic, return True
    if len(txt) == 1:
        return True

    # Check if the character before the last one is a space
    # This means the last character is not part of a word
    return txt[-2] == ' '
