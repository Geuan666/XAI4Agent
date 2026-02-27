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
    # Special case for linear equation (degree 1)
    if len(xs) == 2:
        # f(x) = a + bx = 0 => x = -a/b
        a, b = xs[0], xs[1]
        return -a / b

    # For higher degree polynomials, use numerical method (Newton-Raphson)
    # Start with an initial guess
    x = 0.0

    # Newton-Raphson method: x_new = x - f(x)/f'(x)
    # Derivative of polynomial
    def derivative(xs):
        return [i * xs[i] for i in range(1, len(xs))]

    # Maximum iterations to prevent infinite loop
    max_iter = 1000
    tolerance = 1e-10

    for _ in range(max_iter):
        fx = poly(xs, x)
        if abs(fx) < tolerance:
            return x

        # Calculate derivative at x
        dxs = derivative(xs)
        dfx = poly(dxs, x)

        # Avoid division by zero
        if abs(dfx) < 1e-15:
            # Try a different starting point
            x = 1.0 if x == 0.0 else x * 2
            continue

        # Newton-Raphson update
        x_new = x - fx / dfx
        if abs(x_new - x) < tolerance:
            return x_new
        x = x_new

    # If we reach here, return the last computed value
    return x
