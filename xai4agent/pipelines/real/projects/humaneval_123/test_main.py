from main import get_odd_collatz

# Test the provided example
result = get_odd_collatz(5)
print(f"get_odd_collatz(5) = {result}")

# Additional tests
print(f"get_odd_collatz(3) = {get_odd_collatz(3)}")
print(f"get_odd_collatz(1) = {get_odd_collatz(1)}")
print(f"get_odd_collatz(10) = {get_odd_collatz(10)}")