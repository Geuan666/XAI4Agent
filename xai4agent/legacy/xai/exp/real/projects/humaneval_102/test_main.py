def choose_num(x, y):
    """This function takes two positive numbers x and y and returns the
    biggest even integer number that is in the range [x, y] inclusive. If 
    there's no such number, then the function should return -1.

    For example:
    choose_num(12, 15) = 14
    choose_num(13, 12) = -1
    """
    # If x > y, there's no valid range
    if x > y:
        return -1
    
    # Start from y and go backwards to find the largest even number
    for num in range(y, x - 1, -1):
        if num % 2 == 0:
            return num
    
    # If no even number found, return -1
    return -1

# Test cases
print("Testing choose_num function:")
print(f"choose_num(12, 15) = {choose_num(12, 15)}")  # Should be 14
print(f"choose_num(13, 12) = {choose_num(13, 12)}")  # Should be -1
print(f"choose_num(1, 3) = {choose_num(1, 3)}")      # Should be 2
print(f"choose_num(1, 1) = {choose_num(1, 1)}")      # Should be -1
print(f"choose_num(2, 2) = {choose_num(2, 2)}")      # Should be 2