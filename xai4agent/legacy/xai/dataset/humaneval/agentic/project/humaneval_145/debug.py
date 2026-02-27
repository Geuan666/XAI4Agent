def order_by_points(nums):
    def digit_sum(n):
        return sum(int(digit) for digit in str(abs(n)))
    
    print("Input:", nums)
    for i, num in enumerate(nums):
        print(f"{num}: digit_sum = {digit_sum(num)}")
    
    # Create pairs of (digit_sum, original_index, value) and sort by digit_sum first,
    # then by original_index to maintain stability
    indexed_nums = [(digit_sum(num), i, num) for i, num in enumerate(nums)]
    print("Indexed nums:", indexed_nums)
    sorted_nums = sorted(indexed_nums, key=lambda x: (x[0], x[1]))
    print("Sorted indexed nums:", sorted_nums)
    
    # Extract just the values from the sorted pairs
    result = [num for _, _, num in sorted_nums]
    print("Result:", result)
    return result

# Test with the example
order_by_points([1, 11, -1, -11, -12])