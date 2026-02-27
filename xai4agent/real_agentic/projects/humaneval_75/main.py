def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less then 100. 
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Find all prime factors with their multiplicities
    prime_factors = []
    temp_a = a
    d = 2
    
    while d * d <= temp_a:
        while temp_a % d == 0:
            prime_factors.append(d)
            temp_a //= d
        d += 1
    
    if temp_a > 1:
        prime_factors.append(temp_a)
    
    # Check if we have exactly 3 prime factors (counting multiplicity)
    return len(prime_factors) == 3