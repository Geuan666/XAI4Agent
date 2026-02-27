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
    # Define vowel mappings
    vowel_map = {
        'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
        'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'
    }

    result = []
    for char in message:
        # Swap case
        swapped_char = char.lower() if char.isupper() else char.upper()

        # Replace vowels
        if swapped_char in vowel_map:
            result.append(vowel_map[swapped_char])
        else:
            result.append(swapped_char)

    return ''.join(result)
