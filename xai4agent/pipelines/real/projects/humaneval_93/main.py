def encode(message):
    """
    Write a function that takes a message, and encodes in such a 
    way that it swaps case of all letters, replaces all vowels in 
    the message with the letter that appears 2 places ahead of that 
    vowel in the english alphabet. 
    Assume only letters. 
    
    Examples:
    >>> encode('test')
    'TGST'
    >>> encode('This is a message')
    'tHKS KS C MGSSCGG'
    """
    # Define vowels and their replacements (2 places ahead)
    vowel_replacements = {
        'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
        'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'
    }
    
    result = []
    for char in message:
        if char.isalpha():
            # First swap the case
            swapped_char = char.swapcase()
            
            # Then replace vowels if needed
            if swapped_char in vowel_replacements:
                encoded_char = vowel_replacements[swapped_char]
            else:
                encoded_char = swapped_char
            
            result.append(encoded_char)
        else:
            # Non-alphabetic characters remain unchanged
            result.append(char)
    
    return ''.join(result)