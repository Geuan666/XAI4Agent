def digits(n):
    """Given a positive integer n, return the product of the odd digits.
    Return 0 if all digits are even.
    For example:
    digits(1)  == 1
    digits(4)  == 0
    digits(235) == 15
    """
    product = 1
    has_odd = False
    
    for digit_char in str(n):
        digit = int(digit_char)
        if digit % 2 == 1:  # odd digit
            product *= digit
            has_odd = True
    
    return product if has_odd else 0

# Test cases
print("Testing digits function:")
print(f"digits(1) = {digits(1)}")      # Should be 1
print(f"digits(4) = {digits(4)}")      # Should be 0
print(f"digits(235) = {digits(235)}")  # Should be 15

# Additional test cases
print(f"digits(2468) = {digits(2468)}")  # All even, should be 0
print(f"digits(1357) = {digits(1357)}")  # All odd, should be 1*3*5*7 = 105