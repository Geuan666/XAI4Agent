from main import special_factorial

# Test the example case
result = special_factorial(4)
print(f"special_factorial(4) = {result}")

# Let's also verify step by step:
# 4! = 24
# 3! = 6  
# 2! = 2
# 1! = 1
# Result = 24 * 6 * 2 * 1 = 288

print("Expected: 288")
print(f"Match: {result == 288}")