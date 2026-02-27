def fib4(n: int):
    """The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
    """
    if n == 0 or n == 1 or n == 3:
        return 0
    elif n == 2:
        return 2
    
    # Initialize the first four values
    a, b, c, d = 0, 0, 2, 0  # fib4(0), fib4(1), fib4(2), fib4(3)
    
    # Iteratively compute fib4(n)
    for i in range(4, n + 1):
        next_val = a + b + c + d
        # Shift the window
        a, b, c, d = b, c, d, next_val
    
    return d

# Test the function with the provided examples
if __name__ == "__main__":
    print(f"fib4(5) = {fib4(5)}")  # Should be 4
    print(f"fib4(6) = {fib4(6)}")  # Should be 8
    print(f"fib4(7) = {fib4(7)}")  # Should be 14
    print(f"fib4(0) = {fib4(0)}")  # Should be 0
    print(f"fib4(1) = {fib4(1)}")  # Should be 0
    print(f"fib4(2) = {fib4(2)}")  # Should be 2
    print(f"fib4(3) = {fib4(3)}")  # Should be 0