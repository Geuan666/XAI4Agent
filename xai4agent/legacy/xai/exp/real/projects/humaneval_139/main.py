def special_factorial(n):
    """The Brazilian factorial is defined as:
    brazilian_factorial(n) = n! * (n-1)! * (n-2)! * ... * 1!
    where n > 0

    For example:
    >>> special_factorial(4)
    288

    The function will receive an integer as input and should return the special
    factorial of this integer.
    """
    # Helper function to calculate factorial
    def factorial(x):
        if x <= 1:
            return 1
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result
    
    # Calculate the special factorial
    result = 1
    for i in range(1, n + 1):
        result *= factorial(i)
    
    return result