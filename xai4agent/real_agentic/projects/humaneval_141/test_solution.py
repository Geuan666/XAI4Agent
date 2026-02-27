import main

# Test cases from the docstring
test_cases = [
    ("example.txt", "Yes"),
    ("1example.dll", "No"),
]

all_passed = True
for file_name, expected in test_cases:
    result = main.file_name_check(file_name)
    if result == expected:
        print(f"✓ {file_name} -> {result}")
    else:
        print(f"✗ {file_name} -> {result} (expected {expected})")
        all_passed = False

if all_passed:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed!")