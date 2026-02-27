def right_angle_triangle(a, b, c):
    '''
    Given the lengths of the three sides of a triangle. Return True if the three
    sides form a right-angled triangle, False otherwise.
    A right-angled triangle is a triangle in which one angle is right angle or 
    90 degree.
    Example:
    right_angle_triangle(3, 4, 5) == True
    right_angle_triangle(1, 2, 3) == False
    '''
    # Sort the sides to identify the largest side (potential hypotenuse)
    sides = sorted([a, b, c])
    
    # Check if it's a valid triangle (all sides positive)
    if sides[0] <= 0 or sides[1] <= 0 or sides[2] <= 0:
        return False
    
    # Apply Pythagorean theorem: a^2 + b^2 = c^2
    # where c is the longest side (hypotenuse)
    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-9