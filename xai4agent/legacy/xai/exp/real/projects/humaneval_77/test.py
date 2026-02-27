def iscube(a):
    '''
    Write a function that takes an integer a and returns True 
    if this ingeger is a cube of some integer number.
    Note: you may assume the input is always valid.
    Examples:
    iscube(1) ==> True
    iscube(2) ==> False
    iscube(-1) ==> True
    iscube(64) ==> True
    iscube(0) ==> True
    iscube(180) ==> False
    '''
    # Handle special cases
    if a == 0:
        return True
    
    # For positive numbers
    if a > 0:
        # Calculate cube root and check if it's an integer
        cube_root = round(a ** (1/3))
        # Verify by cubing the result
        return cube_root ** 3 == a
    
    # For negative numbers
    else:
        # For negative numbers, we need to check if the absolute value is a perfect cube
        abs_a = -a
        cube_root = round(abs_a ** (1/3))
        # Verify by cubing the result
        return cube_root ** 3 == abs_a

# Test cases
test_cases = [1, 2, -1, 64, 0, 180]
expected = [True, False, True, True, True, False]

for i, (test, exp) in enumerate(zip(test_cases, expected)):
    result = iscube(test)
    print(f"iscube({test}) ==> {result} {'✓' if result == exp else '✗'}")