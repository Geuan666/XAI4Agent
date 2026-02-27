def words_string(s):
    """
    You will be given a string of words separated by commas or spaces. Your task is
    to split the string into words and return an array of the words.
    
    For example:
    words_string("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]
    words_string("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]
    """
    # Replace commas with spaces, then split by spaces and filter out empty strings
    cleaned = s.replace(',', ' ')
    parts = cleaned.split()
    return parts

# Test the function
print(words_string("Hi, my name is John"))
print(words_string("One, two, three, four, five, six"))