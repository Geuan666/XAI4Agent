def check_dict_case(dict):
    """
    Given a dictionary, return True if all keys are strings in lower 
    case or all keys are strings in upper case, else return False.
    The function should return False is the given dictionary is empty.
    Examples:
    check_dict_case({"a":"apple", "b":"banana"}) should return True.
    check_dict_case({"a":"apple", "A":"banana", "B":"banana"}) should return False.
    check_dict_case({"a":"apple", 8:"banana", "a":"apple"}) should return False.
    check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}) should return False.
    check_dict_case({"STATE":"NC", "ZIP":"12345" }) should return True.
    """
    # Return False if dictionary is empty
    if not dict:
        return False
    
    # Check if all keys are strings
    if not all(isinstance(key, str) for key in dict.keys()):
        return False
    
    # Get all string keys
    string_keys = [key for key in dict.keys() if isinstance(key, str)]
    
    # Check if all keys are lowercase or all keys are uppercase
    all_lower = all(key == key.lower() for key in string_keys)
    all_upper = all(key == key.upper() for key in string_keys)
    
    return all_lower or all_upper

# Test cases
print("Test 1:", check_dict_case({"a":"apple", "b":"banana"}))  # Should return True
print("Test 2:", check_dict_case({"a":"apple", "A":"banana", "B":"banana"}))  # Should return False
print("Test 3:", check_dict_case({"a":"apple", 8:"banana", "a":"apple"}))  # Should return False
print("Test 4:", check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}))  # Should return False
print("Test 5:", check_dict_case({"STATE":"NC", "ZIP":"12345" }))  # Should return True
print("Test 6:", check_dict_case({}))  # Should return False (empty dict)