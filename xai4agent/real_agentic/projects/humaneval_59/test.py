def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    largest = 1
    divisor = 2
    
    while divisor * divisor <= n:
        while n % divisor == 0:
            largest = divisor
            n //= divisor
        divisor += 1
    
    if n > 1:
        largest = n
    
    return largest

# Test cases
print(largest_prime_factor(13195))  # Should output 29
print(largest_prime_factor(2048))   # Should output 2