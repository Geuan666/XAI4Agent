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

    # Check if all sides are positive
    if any(side <= 0 for side in sides):
        return False

    # Apply Pythagorean theorem: a^2 + b^2 = c^2
    # Using the sorted sides, the largest is at index 2
    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-9
