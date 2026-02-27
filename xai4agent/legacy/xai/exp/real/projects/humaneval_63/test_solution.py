def fibfib(n: int):
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """
    if n == 0 or n == 1:
        return 0
    elif n == 2:
        return 1
    
    # For n > 2, use iterative approach with three variables
    a, b, c = 0, 0, 1  # fibfib(0), fibfib(1), fibfib(2)
    
    for i in range(3, n + 1):
        next_val = a + b + c
        a, b, c = b, c, next_val
    
    return c

# Test the provided examples
print("fibfib(1):", fibfib(1))
print("fibfib(5):", fibfib(5))
print("fibfib(8):", fibfib(8))

# Let's also print a few more values to verify the sequence
print("Sequence:", [fibfib(i) for i in range(10)])