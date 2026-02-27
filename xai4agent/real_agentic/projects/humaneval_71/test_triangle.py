def triangle_area(a, b, c):
    '''
    Given the lengths of the three sides of a triangle. Return the area of
    the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
    Otherwise return -1
    Three sides make a valid triangle when the sum of any two sides is greater 
    than the third side.
    Example:
    triangle_area(3, 4, 5) == 6.00
    triangle_area(1, 2, 10) == -1
    '''
    # Check triangle inequality
    if a + b <= c or a + c <= b or b + c <= a:
        return -1
    
    # Calculate area using Heron's formula
    # Semi-perimeter
    s = (a + b + c) / 2
    # Area = sqrt(s * (s-a) * (s-b) * (s-c))
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return round(area, 2)

# Test cases
print("Testing triangle_area(3, 4, 5):")
result1 = triangle_area(3, 4, 5)
print(f"Result: {result1}")
print(f"Expected: 6.00")
print(f"Match: {result1 == 6.00}")

print("\nTesting triangle_area(1, 2, 10):")
result2 = triangle_area(1, 2, 10)
print(f"Result: {result2}")
print(f"Expected: -1")
print(f"Match: {result2 == -1}")