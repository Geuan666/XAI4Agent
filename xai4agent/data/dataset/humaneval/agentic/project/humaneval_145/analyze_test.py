# 手动分析测试用例
test_cases = [
    ([1, 11, -1, -11, -12], [-1, -11, 1, -12, 11]),
    ([1234,423,463,145,2,423,423,53,6,37,3457,3,56,0,46], [0, 2, 3, 6, 53, 423, 423, 423, 1234, 145, 37, 46, 56, 463, 3457]),
    ([], []),
    ([1, -11, -32, 43, 54, -98, 2, -3], [-3, -32, -98, -11, 1, 2, 43, 54]),
    ([1,2,3,4,5,6,7,8,9,10,11], [1, 10, 2, 11, 3, 4, 5, 6, 7, 8, 9]),
    ([0,6,6,-76,-21,23,4], [-76, -21, 0, 4, 23, 6, 6])
]

def digit_sum(n):
    return sum(int(digit) for digit in str(abs(n)))

for i, (input_list, expected) in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    print(f"Input: {input_list}")
    print(f"Expected: {expected}")
    
    # Calculate digit sums
    for j, num in enumerate(input_list):
        print(f"  {num} (index {j}): digit_sum = {digit_sum(num)}")
        
    # Show what we think the sorted order should be
    indexed_nums = [(digit_sum(num), j, num) for j, num in enumerate(input_list)]
    indexed_nums.sort(key=lambda x: (x[0], x[1]))
    actual = [num for _, _, num in indexed_nums]
    print(f"Actual (our implementation): {actual}")
    print(f"Match: {actual == expected}")