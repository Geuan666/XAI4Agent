def get_closest_vowel(word):
    """You are given a word. Your task is to find the closest vowel that stands between 
    two consonants from the right side of the word (case sensitive).
    
    Vowels in the beginning and ending doesn't count. Return empty string if you didn't
    find any vowel met the above condition. 

    You may assume that the given string contains English letter only.

    Example:
    get_closest_vowel("yogurt") ==> "u"
    get_closest_vowel("FULL") ==> "U"
    get_closest_vowel("quick") ==> ""
    get_closest_vowel("ab") ==> ""
    """
    if len(word) < 3:
        return ""
    
    vowels = set('aeiouAEIOU')
    
    # Iterate from right to left, looking for vowels between consonants
    for i in range(len(word) - 2, 0, -1):  # Start from second last char, go to second char
        char = word[i]
        prev_char = word[i-1]
        next_char = word[i+1]
        
        # Check if current character is a vowel
        if char in vowels:
            # Check if previous and next characters are consonants
            if prev_char not in vowels and next_char not in vowels:
                return char
    
    return ""


# Test cases
test_cases = [
    ("yogurt", "u"),
    ("FULL", "U"),
    ("quick", ""),
    ("ab", ""),
]

all_passed = True
for word, expected in test_cases:
    result = get_closest_vowel(word)
    if result != expected:
        print(f"Failed for {word}: expected '{expected}', got '{result}'")
        all_passed = False
    else:
        print(f"Passed for {word}: '{result}'")

if all_passed:
    print("All tests passed!")