from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    # Define the mapping from note symbols to beats
    note_to_beats = {
        'o': 4,
        'o|': 2,
        '.|': 1
    }
    
    # Split the input string by spaces to get individual notes
    notes = music_string.split()
    
    # Convert each note to its corresponding beat count
    result = []
    for note in notes:
        if note in note_to_beats:
            result.append(note_to_beats[note])
        else:
            # Handle invalid notes if any (though not expected per problem description)
            raise ValueError(f"Unknown note symbol: {note}")
    
    return result


# Test the function with the provided example
if __name__ == "__main__":
    test_input = 'o o| .| o| o| .| .| .| .| o o'
    expected_output = [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    actual_output = parse_music(test_input)
    
    print(f"Input: {test_input}")
    print(f"Expected: {expected_output}")
    print(f"Actual: {actual_output}")
    
    if actual_output == expected_output:
        print("Test PASSED!")
    else:
        print("Test FAILED!")