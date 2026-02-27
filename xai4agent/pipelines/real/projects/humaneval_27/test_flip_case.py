from main import flip_case

# Test the function with the provided example
result = flip_case('Hello')
print(f"flip_case('Hello') = '{result}'")

# Additional tests
print(f"flip_case('WORLD') = '{flip_case('WORLD')}'")
print(f"flip_case('PyThOn') = '{flip_case('PyThOn')}'")
print(f"flip_case('Hello123World!') = '{flip_case('Hello123World!')}'")