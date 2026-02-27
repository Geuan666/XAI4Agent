from main import separate_paren_groups

# Test the example from the docstring
result = separate_paren_groups('( ) (( )) (( )( ))')
print("Result:", result)
print("Expected: ['()', '(())', '(()())']")
print("Match:", result == ['()', '(())', '(()())'])

# Additional tests
print("\nAdditional tests:")
print("Empty string:", separate_paren_groups(""))
print("Single group:", separate_paren_groups("()"))
print("Nested groups:", separate_paren_groups("(()())"))