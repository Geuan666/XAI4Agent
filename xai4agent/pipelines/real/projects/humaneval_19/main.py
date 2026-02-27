from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """
    # Mapping from word to number
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    
    # Mapping from number to word
    num_to_word = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
    }
    
    # Split the input string into words
    words = numbers.split()
    
    # Convert words to numbers, sort them, then convert back to words
    numbers_list = [word_to_num[word] for word in words]
    numbers_list.sort()
    sorted_words = [num_to_word[num] for num in numbers_list]
    
    # Join the sorted words back into a string
    return ' '.join(sorted_words)