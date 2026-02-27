def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.
    """
    if n == 1:
        return 1
    else:
        # Numbers starting with 1: 1 followed by (n-1) digits = 10^(n-1)
        start_with_1 = 10 ** (n - 1)
        # Numbers ending with 1: (n-1) digits followed by 1 = 9 * 10^(n-2)
        # First digit can't be 0, so 9 choices for first digit, then (n-2) middle digits
        end_with_1 = 9 * (10 ** (n - 2))
        # Numbers both starting and ending with 1: 1 followed by (n-2) digits followed by 1 = 10^(n-2)
        both = 10 ** (n - 2)
        # Apply inclusion-exclusion principle
        return start_with_1 + end_with_1 - both