import math

def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees
    a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    # For a linear polynomial (degree 1), we can solve analytically
    if len(xs) == 2:
        # ax + b = 0 => x = -b/a
        a, b = xs[1], xs[0]
        return -b / a
    
    # For higher degree polynomials, we'll use a simple numerical approach
    # Try some values around 0 to find a sign change
    for i in range(-100, 101):
        test_x = i / 10.0
        if abs(poly(xs, test_x)) < 1e-10:  # Found exact zero
            return test_x
        if i > -100:
            prev_x = (i - 1) / 10.0
            prev_val = poly(xs, prev_x)
            curr_val = poly(xs, test_x)
            # Check for sign change
            if prev_val * curr_val < 0:
                # Use bisection method between prev_x and test_x
                left, right = prev_x, test_x
                for _ in range(50):  # Limit iterations
                    mid = (left + right) / 2
                    mid_val = poly(xs, mid)
                    if abs(mid_val) < 1e-10:
                        return mid
                    if prev_val * mid_val < 0:
                        right = mid
                        curr_val = mid_val
                    else:
                        left = mid
                        prev_val = mid_val
                return (left + right) / 2
    
    # If no sign change found, try a more systematic approach
    # Try to find a root by testing a wider range
    for i in range(-1000, 1001):
        test_x = i / 100.0
        if abs(poly(xs, test_x)) < 1e-8:
            return test_x
    
    # If still no root found, return a default value (shouldn't happen with valid inputs)
    return 0.0

# Test cases
print("Testing find_zero([1, 2]):")
result1 = find_zero([1, 2])
print(f"Result: {result1}")
print(f"Expected: -0.5")
print(f"Test passed: {abs(result1 - (-0.5)) < 1e-10}")

print("\nTesting find_zero([-6, 11, -6, 1]):")
result2 = find_zero([-6, 11, -6, 1])
print(f"Result: {result2}")
print(f"Expected: 1.0")
print(f"Test passed: {abs(result2 - 1.0) < 1e-10}")