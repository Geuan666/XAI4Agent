def derivative(xs: list):
    """ xs represent coefficients of a polynomial.
    xs[0] + xs[1] * x + xs[2] * x^2 + ....
     Return derivative of this polynomial in the same form.
    >>> derivative([3, 1, 2, 4, 5])
    [1, 4, 12, 20]
    >>> derivative([1, 2, 3])
    [2, 6]
    """
    if len(xs) <= 1:
        return []
    
    result = []
    for i in range(1, len(xs)):
        result.append(i * xs[i])
    
    return result

# Test cases
print("Test 1:", derivative([3, 1, 2, 4, 5]))
print("Test 2:", derivative([1, 2, 3]))

# Additional test cases
print("Test 3:", derivative([5]))  # Should return []
print("Test 4:", derivative([]))   # Should return []