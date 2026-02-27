def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
    
    fix_spaces("Example") == "Example"
    fix_spaces("Example 1") == "Example_1"
    fix_spaces(" Example 2") == "_Example_2"
    fix_spaces(" Example   3") == "_Example-3"
    """
    # First, replace sequences of 3 or more spaces with a dash
    result = ""
    i = 0
    while i < len(text):
        if text[i] == ' ':
            # Check how many consecutive spaces we have
            space_count = 0
            j = i
            while j < len(text) and text[j] == ' ':
                space_count += 1
                j += 1
            
            # Replace based on count
            if space_count >= 3:
                result += '-'
            else:
                result += '_' * space_count
            i = j
        else:
            result += text[i]
            i += 1
    
    return result

# Test cases
print("Testing fix_spaces function:")
print(f'fix_spaces("Example") = "{fix_spaces("Example")}"')
print(f'fix_spaces("Example 1") = "{fix_spaces("Example 1")}"')
print(f'fix_spaces(" Example 2") = "{fix_spaces(" Example 2")}"')
print(f'fix_spaces(" Example   3") = "{fix_spaces(" Example   3")}"')

# Additional test cases
print(f'fix_spaces("Hello    World") = "{fix_spaces("Hello    World")}"')
print(f'fix_spaces("A  B   C") = "{fix_spaces("A  B   C")}"')