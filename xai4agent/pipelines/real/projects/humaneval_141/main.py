def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns
    'Yes' if the the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions 
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the dot should not be empty, and it starts with a letter from 
    the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check("example.txt") # => 'Yes'
    file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
    """
    # Check if there's exactly one dot
    if file_name.count('.') != 1:
        return 'No'
    
    # Split into name and extension
    name_part, ext_part = file_name.split('.')
    
    # Check if name part is not empty
    if len(name_part) == 0:
        return 'No'
    
    # Check if name starts with a Latin letter
    if not (name_part[0].isalpha() and (name_part[0].islower() or name_part[0].isupper())):
        return 'No'
    
    # Check if extension is valid
    valid_extensions = ['txt', 'exe', 'dll']
    if ext_part not in valid_extensions:
        return 'No'
    
    # Check if there are more than three digits
    digit_count = sum(1 for char in file_name if char.isdigit())
    if digit_count > 3:
        return 'No'
    
    # If all checks passed
    return 'Yes'