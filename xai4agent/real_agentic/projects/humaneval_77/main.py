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
        # Calculate cube root
        cube_root = round(a ** (1/3))
        # Check if cube of this root equals original number
        return cube_root ** 3 == a
    
    # For negative numbers
    else:
        # Take absolute value and check if it's a perfect cube
        abs_a = -a
        cube_root = round(abs_a ** (1/3))
        # Check if cube of this root equals original number
        return cube_root ** 3 == abs_a