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
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True

    # 特殊情况：小于2的数不是质数乘积
    if a < 2:
        return False

    # 找到所有可能的三个质数的组合
    primes = [i for i in range(2, 100) if is_prime(i)]

    # 检查是否存在恰好三个质数的乘积等于a
    count = 0
    for i in range(len(primes)):
        if a % primes[i] == 0:
            remaining = a // primes[i]
            for j in range(i, len(primes)):
                if remaining % primes[j] == 0:
                    third = remaining // primes[j]
                    if third >= primes[j] and is_prime(third):
                        if primes[i] * primes[j] * third == a:
                            count += 1
                            # 如果找到一种组合，就返回True
                            return True

    return False
