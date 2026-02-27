from main import factorize

# Test the examples from the docstring
print("Testing factorize(8):", factorize(8))
print("Testing factorize(25):", factorize(25))
print("Testing factorize(70):", factorize(70))

# Additional tests
print("Testing factorize(12):", factorize(12))  # 12 = 2² × 3
print("Testing factorize(17):", factorize(17))  # 17 is prime
print("Testing factorize(1):", factorize(1))    # 1 has no prime factors