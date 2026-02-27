from main import skjkasdkd

# Test cases from the docstring
test_cases = [
    ([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3], 10),
    ([1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1], 25),
    ([1,3,1,32,5107,34,83278,109,163,23,2323,32,30,1,9,3], 13),
    ([0,724,32,71,99,32,6,0,5,91,83,0,5,6], 11),
    ([0,81,12,3,1,21], 3),
    ([0,8,1,2,1,7], 7)
]

print("Testing skjkasdkd function:")
for i, (input_list, expected) in enumerate(test_cases):
    result = skjkasdkd(input_list)
    status = "✓" if result == expected else "✗"
    print(f"Test {i+1}: {status} Input: {input_list}")
    print(f"  Expected: {expected}, Got: {result}")
    if result != expected:
        print(f"  ERROR: Mismatch!")
    print()