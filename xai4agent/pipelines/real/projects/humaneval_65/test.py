def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    """
    # Convert to string to work with digits
    s = str(abs(x))
    n = len(s)
    
    # If shift is greater than number of digits, return reversed
    if shift > n:
        return s[::-1]
    
    # If shift is 0 or equal to number of digits, return original
    if shift == 0 or shift == n:
        return s
    
    # Normalize shift to be within range [0, n)
    shift = shift % n
    
    # Circular right shift: take last 'shift' digits and put them at the front
    shifted = s[-shift:] + s[:-shift]
    
    # Add back the negative sign if original was negative
    if x < 0:
        return "-" + shifted
    else:
        return shifted

# Test cases
print(circular_shift(12, 1))  # Expected: "21"
print(circular_shift(12, 2))  # Expected: "12"
print(circular_shift(1234, 1))  # Expected: "4123"
print(circular_shift(1234, 2))  # Expected: "3412"
print(circular_shift(1234, 3))  # Expected: "2341"
print(circular_shift(1234, 4))  # Expected: "1234"
print(circular_shift(1234, 5))  # Expected: "4321" (reversed since shift > num_digits)
print(circular_shift(-123, 1))  # Expected: "-312"